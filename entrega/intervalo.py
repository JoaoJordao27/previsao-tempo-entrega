"""
Intervalo de previsão calibrado ("entre 24 e 33 min") com regressão quantílica conformalizada (CQR).

    python -m entrega.intervalo

Por que não basta a regressão quantílica pura: ela estima os percentis 10 e 90, mas não garante que 80% dos
casos reais caiam dentro. A calibração usa um pedaço separado dos dados (mais recente que o treino, mais antigo
que o teste) para medir o quanto o intervalo erra e ajustá-lo na medida certa.

Detalhe que importa: o tempo real é medido em minutos INTEIROS e o usuário vê limites inteiros. Calibrar em
minutos contínuos e arredondar depois muda a cobertura (aqui, de ~80% para ~92%), então a calibração é feita
já sobre os limites arredondados, que são os que a pessoa de fato vê.

Há um intervalo por modo (básico e completo, veja entrega/modelo.py), cada um com sua própria calibração.
"""

import json
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from entrega.dados import ALVO, RAIZ, carregar, dividir_por_data
from entrega.modelo import MODOS, PARAMS, ROTULO_MODO, arquivo_treino, categorias_de, matriz, treinar_modelo

COBERTURA = 0.80  # o intervalo deve conter o tempo real em 80% dos pedidos
SAIDA_METRICAS = RAIZ / "resultados" / "intervalo.json"
PASSO_FOLGA = 0.05


def ajuste_conformal(excessos: np.ndarray, cobertura: float = COBERTURA) -> float:
    """Versão contínua: percentil (com correção de amostra finita) dos excessos medidos na calibração."""
    n = len(excessos)
    nivel = min(1.0, math.ceil((n + 1) * cobertura) / n)
    return float(np.quantile(excessos, nivel, method="higher"))


def excessos(real: np.ndarray, baixo: np.ndarray, alto: np.ndarray) -> np.ndarray:
    """Quanto o valor real ficou fora do intervalo (negativo = quão dentro ele ficou)."""
    return np.maximum(baixo - real, real - alto)


def limites_inteiros(baixo: np.ndarray, alto: np.ndarray, folga: float) -> tuple[np.ndarray, np.ndarray]:
    """Aplica a folga e arredonda para fora: o mínimo para baixo, o máximo para cima (nunca menos que 1 min)."""
    baixo, alto = np.minimum(baixo, alto), np.maximum(baixo, alto)  # quantis podem se cruzar: reordena antes
    baixo, alto = baixo - folga, alto + folga
    baixo = np.minimum(baixo, alto)  # folga negativa pode encolher demais
    return np.maximum(np.floor(baixo), 1.0), np.ceil(alto)


def ajuste_inteiro(real: np.ndarray, baixo: np.ndarray, alto: np.ndarray, cobertura: float = COBERTURA) -> float:
    """Menor folga (em minutos, pode ser negativa) cujos limites INTEIROS cobrem a meta na calibração.

    A cobertura só cresce com a folga, então basta varrer em ordem. Usa a mesma correção de amostra finita
    da versão contínua: precisa de ceil((n+1)*cobertura)/n.
    """
    n = len(real)
    alvo = min(1.0, math.ceil((n + 1) * cobertura) / n)
    for folga in np.arange(-4.0, 8.0001, PASSO_FOLGA):
        lo, hi = limites_inteiros(baixo, alto, float(folga))
        if np.mean((real >= lo) & (real <= hi)) >= alvo:
            return round(float(folga), 2)
    return 8.0


def treinar_quantil(treino: pd.DataFrame, categorias: dict, alfa: float, atributos: list[str]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=alfa, **PARAMS).fit(
        matriz(treino, categorias, atributos), treino[ALVO])


def montar(treino: pd.DataFrame, categorias: dict, atributos: list[str]) -> dict:
    """Treina os dois quantis de um modo e calibra. `treino` tem a coluna `data` (usada para separar a calibração)."""
    ajuste_fit, calibra, _ = dividir_por_data(treino, frac_teste=0.2)
    baixo = treinar_quantil(ajuste_fit, categorias, (1 - COBERTURA) / 2, atributos)
    alto = treinar_quantil(ajuste_fit, categorias, 1 - (1 - COBERTURA) / 2, atributos)
    x = matriz(calibra, categorias, atributos)
    folga = ajuste_inteiro(calibra[ALVO].to_numpy(), baixo.predict(x), alto.predict(x))
    return {"atributos": atributos, "baixo": baixo, "alto": alto, "folga_min": folga}


def intervalo(modo: dict, categorias: dict, df: pd.DataFrame, pontual: np.ndarray | None = None):
    """(mínimo, máximo) inteiros, em minutos. Se vier a previsão pontual, o intervalo sempre a contém."""
    x = matriz(df, categorias, modo["atributos"])
    baixo, alto = limites_inteiros(modo["baixo"].predict(x), modo["alto"].predict(x), modo["folga_min"])
    if pontual is not None:
        baixo, alto = np.minimum(baixo, np.floor(pontual)), np.maximum(alto, np.ceil(pontual))
        baixo = np.maximum(baixo, 1.0)
    return baixo, alto


def resumo(real: np.ndarray, baixo: np.ndarray, alto: np.ndarray) -> dict:
    return {
        "cobertura_pct": round(float(((real >= baixo) & (real <= alto)).mean() * 100), 1),
        "largura_media_min": round(float((alto - baixo).mean()), 1),
        "chegou_depois_do_maximo_pct": round(float((real > alto).mean() * 100), 1),
        "chegou_antes_do_minimo_pct": round(float((real < baixo).mean() * 100), 1),
    }


def avaliar_modo(modo_nome: str, treino, teste, categorias, pontual_modelo) -> tuple[dict, dict]:
    """Devolve (pacote do modo pronto para salvar, métricas no teste)."""
    atributos = MODOS[modo_nome]
    real = teste[ALVO].to_numpy()
    ajuste_fit, _, _ = dividir_por_data(treino, frac_teste=0.2)
    x = matriz(teste, categorias, atributos)

    # referência: quantis 10% e 90% crus, só arredondados para fora (sem calibrar)
    cru = {"atributos": atributos, "folga_min": 0.0,
           "baixo": treinar_quantil(ajuste_fit, categorias, 0.1, atributos),
           "alto": treinar_quantil(ajuste_fit, categorias, 0.9, atributos)}
    b0 = cru["baixo"].predict(x), cru["alto"].predict(x)
    contino = resumo(real, np.minimum(*b0), np.maximum(*b0))  # sem arredondar: o que a regressão quantílica "promete"

    pacote = montar(treino, categorias, atributos)
    central = pontual_modelo.predict(x)
    lo0, hi0 = intervalo(cru, categorias, teste, central)
    lo1, hi1 = intervalo(pacote, categorias, teste, central)
    metricas = {
        "folga_conformal_min": pacote["folga_min"],
        "quantis_crus_continuos": contino,
        "quantis_crus_arredondados": resumo(real, lo0, hi0),
        "calibrado": resumo(real, lo1, hi1),
        "calibrado_por_transito": {
            t: resumo(real[m], lo1[m], hi1[m]) for t in sorted(teste["transito"].dropna().unique())
            for m in [(teste["transito"] == t).to_numpy()]},
        "calibrado_por_entregas_simultaneas": {
            str(int(e)): resumo(real[m], lo1[m], hi1[m]) for e in sorted(teste["entregas_simultaneas"].dropna().unique())
            for m in [(teste["entregas_simultaneas"] == e).to_numpy()]},
    }
    return pacote, metricas


def main() -> dict:
    from entrega.servico import salvar_pacotes  # import aqui evita ciclo (servico importa este módulo)

    df = carregar()
    treino, teste, _ = dividir_por_data(df)
    categorias = categorias_de(df)

    pacotes = {"categorias": categorias, "modos": {}}
    resultado = {"cobertura_alvo_pct": COBERTURA * 100, "modos": {}}
    for nome in MODOS:
        pontual = joblib.load(arquivo_treino(nome))
        pacote, met = avaliar_modo(nome, treino, teste, categorias, pontual)
        pacotes["modos"][nome] = {**pacote, "pontual": pontual}
        resultado["modos"][nome] = met

    SAIDA_METRICAS.parent.mkdir(exist_ok=True)
    salvar_pacotes(pacotes)
    SAIDA_METRICAS.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Meta: {COBERTURA * 100:.0f}% dos pedidos dentro do intervalo (limites inteiros)\n")
    for nome, met in resultado["modos"].items():
        print(f"=== modo {nome} · folga conformal {met['folga_conformal_min']:+.2f} min")
        for rot, chave in (("quantis crus, contínuos", "quantis_crus_continuos"), ("quantis crus, arredondados", "quantis_crus_arredondados"),
                           ("calibrado (inteiros)", "calibrado")):
            r = met[chave]
            print(f"  {rot:28s} cobertura {r['cobertura_pct']:5.1f}% · largura {r['largura_media_min']:4.1f} min · "
                  f"depois do máx {r['chegou_depois_do_maximo_pct']:4.1f}% · antes do mín {r['chegou_antes_do_minimo_pct']:4.1f}%")
        print("  por trânsito:", {k: v["cobertura_pct"] for k, v in met["calibrado_por_transito"].items()})
        print("  por entregas simultâneas:", {k: v["cobertura_pct"] for k, v in met["calibrado_por_entregas_simultaneas"].items()})
    return resultado


if __name__ == "__main__":
    main()

"""
Intervalo de previsão calibrado ("entre 24 e 33 min") com regressão quantílica conformalizada (CQR).

    python -m entrega.intervalo

Por que não basta a regressão quantílica pura: ela estima os percentis 10 e 90, mas não garante que 80% dos
casos reais caiam dentro. A calibração conformal usa um pedaço separado dos dados (mais recente que o treino,
mais antigo que o teste) para medir o quanto o intervalo erra e alargá-lo na medida certa.

Passos: treinar os quantis 10% e 90% -> medir no conjunto de calibração o "excesso" de cada pedido fora do
intervalo -> somar ao intervalo o percentil 80% desses excessos.
"""

import json
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from entrega.dados import ALVO, RAIZ, carregar, dividir_por_data
from entrega.modelo import ATRIBUTOS, PARAMS, categorias_de, matriz

COBERTURA = 0.80  # o intervalo deve conter o tempo real em 80% dos pedidos
SAIDA_MODELO = RAIZ / "modelos" / "intervalo.joblib"
SAIDA_METRICAS = RAIZ / "resultados" / "intervalo.json"


def ajuste_conformal(excessos: np.ndarray, cobertura: float = COBERTURA) -> float:
    """Percentil (com correção de amostra finita) dos excessos medidos na calibração."""
    n = len(excessos)
    nivel = min(1.0, math.ceil((n + 1) * cobertura) / n)
    return float(np.quantile(excessos, nivel, method="higher"))


def excessos(real: np.ndarray, baixo: np.ndarray, alto: np.ndarray) -> np.ndarray:
    """Quanto o valor real ficou fora do intervalo (negativo = quão dentro ele ficou)."""
    return np.maximum(baixo - real, real - alto)


def treinar_quantil(treino: pd.DataFrame, categorias: dict, alfa: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=alfa, **PARAMS).fit(matriz(treino, categorias), treino[ALVO])


def montar(treino: pd.DataFrame, categorias: dict) -> dict:
    """Treina os dois quantis e calibra. `treino` já vem ordenado no tempo por `data`."""
    ajuste_fit, calibra, _ = dividir_por_data(treino, frac_teste=0.2)
    baixo = treinar_quantil(ajuste_fit, categorias, (1 - COBERTURA) / 2)
    alto = treinar_quantil(ajuste_fit, categorias, 1 - (1 - COBERTURA) / 2)
    x = matriz(calibra, categorias)
    folga = ajuste_conformal(excessos(calibra[ALVO].to_numpy(), baixo.predict(x), alto.predict(x)))
    return {"baixo": baixo, "alto": alto, "folga_min": folga, "categorias": categorias}


def intervalo(pacote: dict, df: pd.DataFrame, pontual: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mínimo, máximo) em minutos. Se vier a previsão pontual, o intervalo sempre a contém."""
    x = matriz(df, pacote["categorias"])
    baixo = pacote["baixo"].predict(x) - pacote["folga_min"]
    alto = pacote["alto"].predict(x) + pacote["folga_min"]
    baixo, alto = np.minimum(baixo, alto), np.maximum(baixo, alto)  # quantis podem se cruzar
    if pontual is not None:
        baixo, alto = np.minimum(baixo, pontual), np.maximum(alto, pontual)
    return np.maximum(baixo, 1.0), alto


def resumo(real: np.ndarray, baixo: np.ndarray, alto: np.ndarray) -> dict:
    return {
        "cobertura_pct": round(float(((real >= baixo) & (real <= alto)).mean() * 100), 1),
        "largura_media_min": round(float((alto - baixo).mean()), 1),
        "chegou_depois_do_maximo_pct": round(float((real > alto).mean() * 100), 1),
        "chegou_antes_do_minimo_pct": round(float((real < baixo).mean() * 100), 1),
    }


def main() -> dict:
    df = carregar()
    treino, teste, _ = dividir_por_data(df)
    categorias = categorias_de(df)
    real = teste[ALVO].to_numpy()
    x = matriz(teste, categorias)

    # sem calibração: quantis crus
    ajuste_fit, _, _ = dividir_por_data(treino, frac_teste=0.2)
    cru = {"baixo": treinar_quantil(ajuste_fit, categorias, 0.1), "alto": treinar_quantil(ajuste_fit, categorias, 0.9),
           "folga_min": 0.0, "categorias": categorias}
    pacote = montar(treino, categorias)
    pontual = joblib.load(RAIZ / "modelos" / "modelo.joblib")["modelo"].predict(x)

    b0, a0 = intervalo(cru, teste)
    b1, a1 = intervalo(pacote, teste, pontual)
    resultado = {
        "cobertura_alvo_pct": COBERTURA * 100,
        "folga_conformal_min": round(pacote["folga_min"], 2),
        "quantis_crus": resumo(real, b0, a0),
        "calibrado": resumo(real, b1, a1),
        "calibrado_por_transito": {
            t: resumo(real[m], b1[m], a1[m]) for t in sorted(teste["transito"].dropna().unique())
            for m in [(teste["transito"] == t).to_numpy()]
        },
        "calibrado_por_entregas_simultaneas": {
            str(int(e)): resumo(real[m], b1[m], a1[m]) for e in sorted(teste["entregas_simultaneas"].dropna().unique())
            for m in [(teste["entregas_simultaneas"] == e).to_numpy()]
        },
    }

    SAIDA_MODELO.parent.mkdir(exist_ok=True)
    SAIDA_METRICAS.parent.mkdir(exist_ok=True)
    joblib.dump(pacote, SAIDA_MODELO)
    SAIDA_METRICAS.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Meta: {COBERTURA * 100:.0f}% dos pedidos dentro do intervalo · folga conformal: +{pacote['folga_min']:.2f} min\n")
    for nome, chave in (("Quantis crus (10%-90%)", "quantis_crus"), ("Calibrado (conformal)", "calibrado")):
        r = resultado[chave]
        print(f"{nome:24s} cobertura {r['cobertura_pct']:5.1f}% · largura {r['largura_media_min']:4.1f} min · "
              f"depois do máximo {r['chegou_depois_do_maximo_pct']:4.1f}% · antes do mínimo {r['chegou_antes_do_minimo_pct']:4.1f}%")
    print("\nCobertura calibrada por trânsito:")
    for k, v in resultado["calibrado_por_transito"].items():
        print(f"  {k:8s} {v['cobertura_pct']:5.1f}% (largura {v['largura_media_min']} min)")
    print("Cobertura calibrada por entregas simultâneas:")
    for k, v in resultado["calibrado_por_entregas_simultaneas"].items():
        print(f"  {k:8s} {v['cobertura_pct']:5.1f}% (largura {v['largura_media_min']} min)")
    return resultado


if __name__ == "__main__":
    main()

"""
Treino e avaliação do modelo de tempo de entrega.

    python -m entrega.modelo

Compara o LightGBM com duas bases simples (se o modelo não ganha delas, não vale o esforço),
valida por data (treina no passado, testa no futuro) e mede importância por permutação.

Há dois modos, porque nem sempre se conhece o entregador na hora do pedido:
  - básico:   só o que existe quando o pedido é feito (clima, trânsito, distância, hora...);
  - completo: acrescenta nota, idade e condição do veículo do entregador, e a espera até a coleta.

Salva os modelos pontuais em modelos/pontual_treino_<modo>.joblib (só para o passo seguinte, fora do Git) e as
métricas em resultados/metricas.json. Os arquivos de uso (API e tela) saem de `python -m entrega.intervalo`.
"""

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error

from entrega.dados import ALVO, RAIZ, carregar, dividir_por_data

CATEGORICAS = ["clima", "transito", "tipo_pedido", "veiculo", "cidade"]
NUMERICAS = ["idade_entregador", "nota_entregador", "condicao_veiculo", "entregas_simultaneas", "festival",
             "distancia_km", "hora", "espera_coleta_min", "dia_semana", "fim_de_semana"]
ATRIBUTOS = CATEGORICAS + NUMERICAS
DO_ENTREGADOR = ["idade_entregador", "nota_entregador", "condicao_veiculo", "espera_coleta_min"]
MODOS = {"basico": [a for a in ATRIBUTOS if a not in DO_ENTREGADOR], "completo": ATRIBUTOS}
ROTULO_MODO = {"basico": "LightGBM (básico)", "completo": "LightGBM (completo)"}
PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, random_state=27, verbose=-1)
LIMITE_ATRASO_MIN = 5  # chegar mais de 5 min depois do previsto conta como "atraso" para o cliente

SAIDA_METRICAS = RAIZ / "resultados" / "metricas.json"


def arquivo_treino(modo: str):
    return RAIZ / "modelos" / f"pontual_treino_{modo}.joblib"


def categorias_de(df: pd.DataFrame) -> dict[str, list]:
    """Lista fixa de valores de cada coluna categórica (o modelo precisa dela na hora de prever)."""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAS}


def matriz(df: pd.DataFrame, categorias: dict[str, list], atributos: list[str] = ATRIBUTOS) -> pd.DataFrame:
    """Seleciona os atributos e fixa as categorias, para treino e uso em produção ficarem idênticos."""
    x = df[atributos].copy()
    for c, valores in categorias.items():
        if c in x:
            x[c] = pd.Categorical(x[c], categories=valores)
    return x


def metricas(real: pd.Series, previsto: np.ndarray) -> dict:
    erro = np.asarray(real) - np.asarray(previsto)  # positivo = demorou mais do que o previsto
    return {
        "mae_min": round(float(np.abs(erro).mean()), 2),
        "rmse_min": round(float(root_mean_squared_error(real, previsto)), 2),
        "acerta_ate_5min_pct": round(float((np.abs(erro) <= 5).mean() * 100), 1),
        "atraso_acima_5min_pct": round(float((erro > LIMITE_ATRASO_MIN).mean() * 100), 1),
    }


def base_media_geral(treino: pd.DataFrame, teste: pd.DataFrame) -> np.ndarray:
    return np.full(len(teste), treino[ALVO].mean())


def base_regra(treino: pd.DataFrame, teste: pd.DataFrame) -> np.ndarray:
    """Regra que um analista faria à mão: média por (trânsito, entregas simultâneas)."""
    chaves = ["transito", "entregas_simultaneas"]
    tabela = treino.groupby(chaves)[ALVO].mean()
    idx = pd.MultiIndex.from_frame(teste[chaves])
    previsto = tabela.reindex(idx).to_numpy(dtype=float)
    return np.where(np.isnan(previsto), treino[ALVO].mean(), previsto)


def treinar_modelo(treino: pd.DataFrame, categorias: dict[str, list], atributos: list[str] = ATRIBUTOS) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**PARAMS).fit(matriz(treino, categorias, atributos), treino[ALVO])


def importancia(modelo, teste: pd.DataFrame, categorias: dict, atributos: list[str] = ATRIBUTOS) -> dict[str, float]:
    """Quanto o erro piora (em minutos) ao embaralhar cada atributo no teste. Mais honesto que 'gain'."""
    x, y = matriz(teste, categorias, atributos), teste[ALVO]
    r = permutation_importance(modelo, x, y, scoring="neg_mean_absolute_error", n_repeats=5, random_state=27)
    s = pd.Series(r.importances_mean, index=atributos).sort_values(ascending=False)
    return {k: round(float(v), 2) for k, v in s.items()}


def main() -> dict:
    df = carregar()
    treino, teste, corte = dividir_por_data(df)
    categorias = categorias_de(df)

    modelos, previstos = {}, {}
    for modo, atributos in MODOS.items():
        modelos[modo] = treinar_modelo(treino, categorias, atributos)
        previstos[modo] = modelos[modo].predict(matriz(teste, categorias, atributos))

    resultado = {
        "corte_treino_teste": str(corte.date()),
        "linhas_treino": len(treino), "linhas_teste": len(teste),
        "modelos": {
            "média geral": metricas(teste[ALVO], base_media_geral(treino, teste)),
            "regra por trânsito e entregas": metricas(teste[ALVO], base_regra(treino, teste)),
            **{ROTULO_MODO[m]: metricas(teste[ALVO], previstos[m]) for m in MODOS},
        },
        "importancia_por_permutacao_min": {m: importancia(modelos[m], teste, categorias, MODOS[m]) for m in MODOS},
    }

    SAIDA_METRICAS.parent.mkdir(exist_ok=True)
    for modo, modelo in modelos.items():
        arquivo_treino(modo).parent.mkdir(exist_ok=True)
        joblib.dump(modelo, arquivo_treino(modo))
    SAIDA_METRICAS.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Treino até {corte.date()} ({len(treino)} pedidos) · teste depois ({len(teste)} pedidos)\n")
    for nome, m in resultado["modelos"].items():
        print(f"{nome:32s} MAE {m['mae_min']:5.2f} min · RMSE {m['rmse_min']:5.2f} · "
              f"±5 min {m['acerta_ate_5min_pct']:4.1f}% · atraso>5 {m['atraso_acima_5min_pct']:4.1f}%")
    for modo, imp in resultado["importancia_por_permutacao_min"].items():
        print(f"\nImportância por permutação, modo {modo} (aumento do erro, em min):")
        for k, v in list(imp.items())[:8]:
            print(f"  {k:22s} {v:5.2f}")
    return resultado


if __name__ == "__main__":
    main()

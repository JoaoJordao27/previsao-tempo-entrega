"""
Camada de uso do modelo: carrega os arquivos treinados e transforma pedidos em previsões.
É compartilhada pela API (entrega/api.py) e pela tela em Streamlit, para os dois darem o mesmo resultado.
"""

import math

import joblib
import numpy as np
import pandas as pd

from entrega.dados import RAIZ
from entrega.intervalo import COBERTURA, intervalo
from entrega.modelo import CATEGORICAS, NUMERICAS, matriz

ARQ_PONTUAL = RAIZ / "modelos" / "modelo.joblib"
ARQ_INTERVALO = RAIZ / "modelos" / "intervalo.joblib"


def carregar_pacotes() -> dict:
    """Lê os modelos treinados. Se não existirem, explica o que rodar."""
    if not (ARQ_PONTUAL.exists() and ARQ_INTERVALO.exists()):
        raise FileNotFoundError("Modelos não encontrados. Rode: python -m entrega.modelo && python -m entrega.intervalo")
    return {"pontual": joblib.load(ARQ_PONTUAL)["modelo"], "intervalo": joblib.load(ARQ_INTERVALO)}


def para_dataframe(pedidos: list[dict]) -> pd.DataFrame:
    """Monta a tabela que o modelo espera. Campos ausentes (None) viram NaN, que o LightGBM sabe tratar."""
    df = pd.DataFrame(pedidos)
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce") if c in df else np.nan
    for c in CATEGORICAS:
        df[c] = df[c] if c in df else None
    df["fim_de_semana"] = np.where(df["dia_semana"].isna(), np.nan, (df["dia_semana"] >= 5).astype(float))
    df["festival"] = df["festival"].astype(float)
    return df


def prever(pacotes: dict, pedidos: list[dict]) -> list[dict]:
    """Devolve, para cada pedido, a previsão e o intervalo (em minutos inteiros)."""
    df = para_dataframe(pedidos)
    cats = pacotes["intervalo"]["categorias"]
    central = pacotes["pontual"].predict(matriz(df, cats))
    baixo, alto = intervalo(pacotes["intervalo"], df, central)
    saida = []
    for c, b, a in zip(central, baixo, alto):
        previsao = max(1, round(float(c)))
        minimo, maximo = min(math.floor(b), previsao), max(math.ceil(a), previsao)
        saida.append({
            "previsao_min": previsao, "minimo_min": minimo, "maximo_min": maximo,
            "texto": f"entre {minimo} e {maximo} min", "cobertura_pct": round(COBERTURA * 100),
        })
    return saida

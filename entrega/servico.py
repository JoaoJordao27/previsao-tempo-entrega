"""
Camada de uso do modelo: salva/carrega os arquivos treinados e transforma pedidos em previsões.
É compartilhada pela API (entrega/api.py) e pela tela em Streamlit, para os dois darem o mesmo resultado.

Os modelos ficam em arquivos de texto nativos do LightGBM (não em pickle): funcionam em qualquer versão de
Python e de scikit-learn, o que importa para o deploy rodar em uma máquina diferente da que treinou.
"""

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from entrega.dados import RAIZ
from entrega.intervalo import COBERTURA, intervalo
from entrega.modelo import CATEGORICAS, NUMERICAS, matriz

PASTA_MODELOS = RAIZ / "modelos"
ARQUIVOS = {"pontual": "pontual.txt", "baixo": "baixo.txt", "alto": "alto.txt"}
META = "meta.json"


def _booster(modelo) -> lgb.Booster:
    return modelo.booster_ if hasattr(modelo, "booster_") else modelo


def _ler(arquivo: Path) -> lgb.Booster:
    return lgb.Booster(model_str=arquivo.read_text(encoding="utf-8"))


def salvar_pacotes(pacotes: dict, pasta: Path = PASTA_MODELOS) -> None:
    """pacotes = {"pontual": modelo, "intervalo": {"baixo", "alto", "folga_min", "categorias"}}."""
    pasta.mkdir(parents=True, exist_ok=True)
    # via texto + Python (e não save_model): a biblioteca em C do LightGBM falha em caminhos com acento no Windows
    for chave, modelo in (("pontual", pacotes["pontual"]), ("baixo", pacotes["intervalo"]["baixo"]),
                          ("alto", pacotes["intervalo"]["alto"])):
        (pasta / ARQUIVOS[chave]).write_text(_booster(modelo).model_to_string(), encoding="utf-8")
    meta = {"folga_min": pacotes["intervalo"]["folga_min"], "categorias": pacotes["intervalo"]["categorias"],
            "lightgbm": lgb.__version__}
    (pasta / META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_pacotes(pasta: Path = PASTA_MODELOS) -> dict:
    """Lê os modelos treinados. Se não existirem, explica o que rodar."""
    arquivos = [pasta / n for n in (*ARQUIVOS.values(), META)]
    if not all(a.exists() for a in arquivos):
        raise FileNotFoundError("Modelos não encontrados. Rode: python -m entrega.modelo && python -m entrega.intervalo")
    meta = json.loads((pasta / META).read_text(encoding="utf-8"))
    return {
        "pontual": _ler(pasta / ARQUIVOS["pontual"]),
        "intervalo": {
            "baixo": _ler(pasta / ARQUIVOS["baixo"]),
            "alto": _ler(pasta / ARQUIVOS["alto"]),
            "folga_min": meta["folga_min"], "categorias": meta["categorias"],
        },
    }


def para_dataframe(pedidos: list[dict]) -> pd.DataFrame:
    """Monta a tabela que o modelo espera. Campos ausentes (None) viram NaN, que o LightGBM sabe tratar."""
    df = pd.DataFrame(pedidos)
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
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

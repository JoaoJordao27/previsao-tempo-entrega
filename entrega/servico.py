"""
Camada de uso do modelo: salva/carrega os arquivos treinados e transforma pedidos em previsões.
É compartilhada pela API (entrega/api.py) e pela tela em Streamlit, para os dois darem o mesmo resultado.

Os modelos ficam em arquivos de texto nativos do LightGBM (não em pickle): funcionam em qualquer versão de
Python e de scikit-learn, o que importa para o deploy rodar em uma máquina diferente da que treinou.

Estrutura dos pacotes:
    {"categorias": {...},
     "modos": {"basico":   {"atributos", "pontual", "baixo", "alto", "folga_min"},
               "completo": {...}}}
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from entrega.dados import RAIZ
from entrega.intervalo import COBERTURA, intervalo
from entrega.modelo import CATEGORICAS, MODOS, NUMERICAS, matriz

PASTA_MODELOS = RAIZ / "modelos"
PARTES = ("pontual", "baixo", "alto")
META = "meta.json"
EXIGIDOS_PARA_COMPLETO = ["nota_entregador", "idade_entregador", "condicao_veiculo"]


def _booster(modelo) -> lgb.Booster:
    return modelo.booster_ if hasattr(modelo, "booster_") else modelo


def _ler(arquivo: Path) -> lgb.Booster:
    return lgb.Booster(model_str=arquivo.read_text(encoding="utf-8"))


def _arquivo(pasta: Path, modo: str, parte: str) -> Path:
    return pasta / f"{modo}_{parte}.txt"


def salvar_pacotes(pacotes: dict, pasta: Path = PASTA_MODELOS) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    meta = {"categorias": pacotes["categorias"], "lightgbm": lgb.__version__, "modos": {}}
    for modo, p in pacotes["modos"].items():
        # via texto + Python (e não save_model): a biblioteca em C do LightGBM falha em caminhos com acento no Windows
        for parte in PARTES:
            _arquivo(pasta, modo, parte).write_text(_booster(p[parte]).model_to_string(), encoding="utf-8")
        meta["modos"][modo] = {"atributos": p["atributos"], "folga_min": p["folga_min"]}
    (pasta / META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_pacotes(pasta: Path = PASTA_MODELOS) -> dict:
    """Lê os modelos treinados. Se não existirem, explica o que rodar."""
    if not (pasta / META).exists():
        raise FileNotFoundError("Modelos não encontrados. Rode: python -m entrega.modelo && python -m entrega.intervalo")
    meta = json.loads((pasta / META).read_text(encoding="utf-8"))
    modos = {
        modo: {**m, **{parte: _ler(_arquivo(pasta, modo, parte)) for parte in PARTES}}
        for modo, m in meta["modos"].items()
    }
    return {"categorias": meta["categorias"], "modos": modos}


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


def escolher_modo(df: pd.DataFrame) -> pd.Series:
    """'completo' só quando o pedido traz os três dados do entregador; senão 'basico'."""
    conhecido = df[EXIGIDOS_PARA_COMPLETO].notna().all(axis=1)
    return pd.Series(np.where(conhecido, "completo", "basico"), index=df.index)


def prever(pacotes: dict, pedidos: list[dict]) -> list[dict]:
    """Devolve, para cada pedido, a previsão, o intervalo (minutos inteiros) e o modo usado."""
    df = para_dataframe(pedidos).reset_index(drop=True)
    modos = escolher_modo(df)
    cats = pacotes["categorias"]
    saida: list[dict | None] = [None] * len(df)
    for nome, idx in modos.groupby(modos).groups.items():
        parte = df.loc[idx]
        modo = pacotes["modos"][nome]
        central = modo["pontual"].predict(matriz(parte, cats, modo["atributos"]))
        baixo, alto = intervalo(modo, cats, parte, central)
        for i, c, b, a in zip(idx, central, baixo, alto):
            previsao = max(1, round(float(c)))
            minimo, maximo = min(int(b), previsao), max(int(a), previsao)
            saida[i] = {"previsao_min": previsao, "minimo_min": minimo, "maximo_min": maximo,
                        "texto": f"entre {minimo} e {maximo} min", "cobertura_pct": round(COBERTURA * 100), "modo": nome}
    return saida


def treinar_pacotes(treino: pd.DataFrame, categorias: dict) -> dict:
    """Treina e calibra os dois modos (usado pelos testes e por quem quiser montar tudo em memória)."""
    from entrega.intervalo import montar
    from entrega.modelo import treinar_modelo

    modos = {}
    for nome, atributos in MODOS.items():
        modos[nome] = {**montar(treino, categorias, atributos), "pontual": treinar_modelo(treino, categorias, atributos)}
    return {"categorias": categorias, "modos": modos}

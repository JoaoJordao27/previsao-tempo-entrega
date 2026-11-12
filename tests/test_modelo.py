"""Testes do modelo com dados pequenos e controlados (sem baixar o dataset)."""

import numpy as np
import pandas as pd
import pytest

from entrega.dados import ALVO
from entrega.modelo import ATRIBUTOS, base_media_geral, base_regra, categorias_de, matriz, metricas


def juntar(n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "clima": rng.choice(["Sunny", "Fog"], n), "transito": rng.choice(["Low", "Jam"], n),
        "tipo_pedido": "Snack", "veiculo": "motorcycle", "cidade": "Urban",
        "idade_entregador": 30.0, "nota_entregador": 4.5, "condicao_veiculo": 1,
        "entregas_simultaneas": rng.choice([0.0, 1.0], n), "festival": 0.0, "distancia_km": 5.0,
        "hora": 12.0, "espera_coleta_min": 10.0, "dia_semana": 2, "fim_de_semana": 0,
    })
    df[ALVO] = np.where(df.transito == "Jam", 30, 20)
    return df


def test_metricas_separam_atraso_de_adiantamento():
    real = pd.Series([30, 30, 30, 30])
    previsto = np.array([30, 20, 40, 29])  # chegou 10 min DEPOIS, 10 min ANTES, 1 min antes
    m = metricas(real, previsto)
    assert m["mae_min"] == pytest.approx(5.25)
    assert m["atraso_acima_5min_pct"] == 25.0  # só o 2º caso: real 30 vs previsto 20
    assert m["acerta_ate_5min_pct"] == 50.0


def test_base_regra_aprende_a_diferenca_de_transito():
    df = juntar()
    previsto = base_regra(df, df)
    assert previsto[df.transito == "Jam"].mean() == pytest.approx(30)
    assert previsto[df.transito == "Low"].mean() == pytest.approx(20)


def test_base_regra_usa_media_geral_para_combinacao_nova():
    treino = juntar()
    novo = treino.head(1).copy()
    novo["transito"] = "High"  # nunca visto no treino
    assert base_regra(treino, novo)[0] == pytest.approx(treino[ALVO].mean())


def test_base_media_geral_e_constante():
    df = juntar()
    assert set(base_media_geral(df, df)) == {df[ALVO].mean()}


def test_matriz_fixa_categorias_mesmo_com_valor_ausente_no_lote():
    df = juntar()
    cats = categorias_de(df)
    um = df[df.clima == "Sunny"].head(1)  # lote sem nenhuma linha "Fog"
    x = matriz(um, cats)
    assert list(x.columns) == ATRIBUTOS
    assert list(x["clima"].cat.categories) == cats["clima"]

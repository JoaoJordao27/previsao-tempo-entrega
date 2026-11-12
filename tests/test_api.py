"""Testes da API com modelos pequenos treinados na hora, sem depender dos arquivos reais."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from entrega.api import criar_app
from entrega.dados import ALVO
from entrega.intervalo import montar
from entrega.modelo import categorias_de, treinar_modelo

VALIDO = {"distancia_km": 6.5, "transito": "High", "clima": "Stormy", "veiculo": "motorcycle",
          "tipo_pedido": "Meal", "hora": 20}


def base_sintetica(n=900) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "data": pd.date_range("2022-01-01", periods=n, freq="2h"),
        "clima": rng.choice(["Sunny", "Stormy", "Fog"], n), "transito": rng.choice(["Low", "Medium", "High", "Jam"], n),
        "tipo_pedido": rng.choice(["Snack", "Meal"], n), "veiculo": rng.choice(["motorcycle", "scooter"], n),
        "cidade": rng.choice(["Urban", "Metropolitian"], n), "idade_entregador": rng.integers(20, 45, n).astype(float),
        "nota_entregador": rng.uniform(3.5, 5, n), "condicao_veiculo": rng.integers(0, 4, n),
        "entregas_simultaneas": rng.integers(0, 3, n).astype(float), "festival": 0.0,
        "distancia_km": rng.uniform(1, 20, n), "hora": rng.integers(8, 23, n).astype(float),
        "espera_coleta_min": rng.choice([5.0, 10.0, 15.0], n), "dia_semana": rng.integers(0, 7, n),
    })
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df[ALVO] = (15 + df.distancia_km * 1.2 + df.transito.map({"Low": 0, "Medium": 3, "High": 6, "Jam": 10})
                + rng.normal(0, 2, n))
    return df


@pytest.fixture(scope="module")
def cliente():
    df = base_sintetica()
    cats = categorias_de(df)
    pacotes = {"pontual": treinar_modelo(df, cats), "intervalo": montar(df, cats)}
    with TestClient(criar_app(lambda: pacotes)) as c:
        yield c


def test_saude(cliente):
    assert cliente.get("/saude").json() == {"ok": True}


def test_prever_devolve_intervalo_coerente(cliente):
    r = cliente.post("/prever", json=VALIDO)
    assert r.status_code == 200
    d = r.json()
    assert d["minimo_min"] <= d["previsao_min"] <= d["maximo_min"]
    assert d["texto"] == f"entre {d['minimo_min']} e {d['maximo_min']} min" and d["cobertura_pct"] == 80


def test_campos_opcionais_ausentes_funcionam(cliente):
    assert cliente.post("/prever", json=VALIDO).status_code == 200
    completo = dict(VALIDO, cidade="Urban", idade_entregador=30, nota_entregador=4.5, condicao_veiculo=2,
                    entregas_simultaneas=1, festival=True, espera_coleta_min=10, dia_semana=5)
    assert cliente.post("/prever", json=completo).status_code == 200


@pytest.mark.parametrize("campo,valor", [
    ("distancia_km", -1), ("distancia_km", 0), ("distancia_km", 500), ("hora", 24), ("hora", -1),
    ("transito", "Congestionado"), ("clima", "Neve"), ("veiculo", "carro"),
    ("nota_entregador", 6), ("entregas_simultaneas", 9), ("dia_semana", 7),
])
def test_valores_invalidos_sao_recusados(cliente, campo, valor):
    assert cliente.post("/prever", json=dict(VALIDO, **{campo: valor})).status_code == 422


def test_campo_obrigatorio_ausente_e_campo_desconhecido(cliente):
    sem_distancia = {k: v for k, v in VALIDO.items() if k != "distancia_km"}
    assert cliente.post("/prever", json=sem_distancia).status_code == 422
    assert cliente.post("/prever", json=dict(VALIDO, cor_da_moto="azul")).status_code == 422


def test_mais_distancia_nao_reduz_o_tempo_no_cenario_simulado(cliente):
    perto = cliente.post("/prever", json=dict(VALIDO, distancia_km=2)).json()["previsao_min"]
    longe = cliente.post("/prever", json=dict(VALIDO, distancia_km=18)).json()["previsao_min"]
    assert longe > perto


def test_sem_modelo_a_api_sobe_e_responde_503():
    def falha():
        raise FileNotFoundError("sem modelo")

    with TestClient(criar_app(falha)) as c:
        assert c.get("/saude").json() == {"ok": False}
        r = c.post("/prever", json=VALIDO)
        assert r.status_code == 503 and "sem modelo" in r.json()["detail"]

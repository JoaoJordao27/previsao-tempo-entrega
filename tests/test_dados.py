"""Testes da limpeza (rodar com: python -m pytest -q). Usam linhas sintéticas, sem baixar o dataset."""

import numpy as np
import pandas as pd
import pytest

from entrega.dados import ALVO, dividir_por_data, haversine_km, limpar


def linha(**muda):
    base = {
        "ID": "0x1", "Delivery_person_ID": "INDORES13DEL02", "Delivery_person_Age": "37",
        "Delivery_person_Ratings": "4.9", "Restaurant_latitude": 22.745049, "Restaurant_longitude": 75.892471,
        "Delivery_location_latitude": 22.765049, "Delivery_location_longitude": 75.912471,
        "Order_Date": "19-03-2022", "Time_Orderd": "11:30:00", "Time_Order_picked": "11:45:00",
        "Weatherconditions": "conditions Sunny", "Road_traffic_density": "High ", "Vehicle_condition": 2,
        "Type_of_order": "Snack ", "Type_of_vehicle": "motorcycle ", "multiple_deliveries": "0",
        "Festival": "No ", "City": "Urban ", "Time_taken(min)": "(min) 24",
    }
    base.update(muda)
    return base


def limpa(*linhas):
    return limpar(pd.DataFrame(list(linhas)))


def test_converte_textos_e_alvo():
    df, _ = limpa(linha())
    r = df.iloc[0]
    assert r[ALVO] == 24 and r["clima"] == "Sunny" and r["transito"] == "High"
    assert r["idade_entregador"] == 37 and r["festival"] == 0 and r["cidade_codigo"] == "INDO"
    assert r["hora"] == 11 and r["espera_coleta_min"] == 15


def test_textos_nan_viram_faltantes():
    df, _ = limpa(linha(Delivery_person_Age="NaN ", Weatherconditions="conditions NaN", City="NaN "))
    assert df[["idade_entregador", "clima", "cidade"]].isna().all(axis=None)


def test_nota_acima_de_5_vira_faltante():
    df, rel = limpa(linha(Delivery_person_Ratings="6"))
    assert np.isnan(df.iloc[0]["nota_entregador"]) and rel["nota_acima_de_5"] == 1


def test_coordenadas_invalidas():
    df, rel = limpa(linha(Restaurant_latitude=-22.745049), linha(Restaurant_latitude=0.0, Restaurant_longitude=0.0))
    assert df.iloc[0]["distancia_km"] < 5  # sinal corrigido -> distância curta e válida
    assert rel["rest_lat_negativa_corrigida"] == 1
    assert np.isnan(df.iloc[1]["distancia_km"]) and rel["coordenada_restaurante_invalida"] == 1


def test_espera_que_passa_da_meia_noite():
    df, _ = limpa(linha(Time_Orderd="23:55:00", Time_Order_picked="00:05:00"))
    assert df.iloc[0]["espera_coleta_min"] == 10


def test_haversine_conhecido():
    # São Paulo (Sé) -> Rio (Centro): ~360 km em linha reta
    assert haversine_km(-23.5505, -46.6333, -22.9068, -43.1729) == pytest.approx(361, abs=5)


def test_divisao_por_data_nao_mistura_passado_e_futuro():
    df = pd.DataFrame({"data": pd.date_range("2022-01-01", periods=100, freq="D")})
    treino, teste, corte = dividir_por_data(df)
    assert treino["data"].max() < corte <= teste["data"].min()
    assert len(teste) == pytest.approx(20, abs=2)

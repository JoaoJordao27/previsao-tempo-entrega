"""
Carrega e limpa o Food Delivery Dataset (Kaggle: gauravmalik26/food-delivery-dataset).

O arquivo bruto não vai para o repositório (licença "other"); baixe com:
    python -m entrega.dados --baixar
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
BRUTO = RAIZ / "dados" / "brutos" / "train.csv"
DATASET_KAGGLE = "gauravmalik26/food-delivery-dataset"

ALVO = "tempo_min"
RAIO_TERRA_KM = 6371.0

NOMES = {
    "ID": "id",
    "Delivery_person_ID": "entregador_id",
    "Delivery_person_Age": "idade_entregador",
    "Delivery_person_Ratings": "nota_entregador",
    "Restaurant_latitude": "rest_lat",
    "Restaurant_longitude": "rest_lon",
    "Delivery_location_latitude": "cli_lat",
    "Delivery_location_longitude": "cli_lon",
    "Order_Date": "data",
    "Time_Orderd": "hora_pedido",
    "Time_Order_picked": "hora_coleta",
    "Weatherconditions": "clima",
    "Road_traffic_density": "transito",
    "Vehicle_condition": "condicao_veiculo",
    "Type_of_order": "tipo_pedido",
    "Type_of_vehicle": "veiculo",
    "multiple_deliveries": "entregas_simultaneas",
    "Festival": "festival",
    "City": "cidade",
    "Time_taken(min)": ALVO,
}


def baixar() -> None:
    """Baixa o dataset com a CLI do Kaggle (precisa de ~/.kaggle/kaggle.json)."""
    BRUTO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", DATASET_KAGGLE, "--unzip", "-p", str(BRUTO.parent)],
                   check=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em linha reta entre dois pontos da Terra, em km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def _minutos(hora: pd.Series) -> pd.Series:
    """'11:30:00' -> 690.0 (minutos desde meia-noite). Valores inválidos viram NaN."""
    t = pd.to_datetime(hora, format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute


def limpar(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Padroniza tipos, corrige valores inválidos e cria atributos básicos.

    Devolve (dados limpos, relatório com quantas linhas cada regra afetou).
    """
    df = bruto.rename(columns=NOMES).copy()
    rel = {"linhas_brutas": len(df)}

    texto = df.select_dtypes(include=["object", "string"]).columns
    df[texto] = df[texto].apply(lambda s: s.str.strip())
    df = df.replace({"NaN": np.nan, "conditions NaN": np.nan})

    df[ALVO] = df[ALVO].str.replace("(min)", "", regex=False).astype(int)
    df["clima"] = df["clima"].str.replace("conditions ", "", regex=False)
    df["idade_entregador"] = pd.to_numeric(df["idade_entregador"], errors="coerce")
    df["nota_entregador"] = pd.to_numeric(df["nota_entregador"], errors="coerce")
    df["entregas_simultaneas"] = pd.to_numeric(df["entregas_simultaneas"], errors="coerce")
    df["festival"] = df["festival"].map({"Yes": 1, "No": 0})
    df["data"] = pd.to_datetime(df["data"], format="%d-%m-%Y")

    # notas fora da escala 1-5 são erro de cadastro
    fora = df["nota_entregador"] > 5
    rel["nota_acima_de_5"] = int(fora.sum())
    df.loc[fora, "nota_entregador"] = np.nan

    # coordenadas: o dataset é da Índia (latitudes positivas). Latitude negativa é sinal trocado;
    # zero (ou muito perto) é ausência de dado.
    for lat, lon in (("rest_lat", "rest_lon"), ("cli_lat", "cli_lon")):
        neg = df[lat] < 0
        rel[f"{lat}_negativa_corrigida"] = int(neg.sum())
        df.loc[neg, lat] = df.loc[neg, lat].abs()
        df.loc[df[lon] < 0, lon] = df.loc[df[lon] < 0, lon].abs()
    invalida = (df["rest_lat"] < 1) | (df["rest_lon"] < 1)
    rel["coordenada_restaurante_invalida"] = int(invalida.sum())
    df.loc[invalida, ["rest_lat", "rest_lon"]] = np.nan

    df["distancia_km"] = haversine_km(df["rest_lat"], df["rest_lon"], df["cli_lat"], df["cli_lon"])
    longe = df["distancia_km"] > 50  # entrega de comida acima de 50 km é erro de coordenada
    rel["distancia_acima_50km"] = int(longe.sum())
    df.loc[longe, "distancia_km"] = np.nan

    # horário: só o pedido tem faltantes; a espera até a coleta passa da meia-noite em alguns casos
    pedido, coleta = _minutos(df["hora_pedido"]), _minutos(df["hora_coleta"])
    df["hora"] = pedido // 60
    espera = (coleta - pedido) % (24 * 60)
    df["espera_coleta_min"] = espera.where(espera <= 60)
    df["dia_semana"] = df["data"].dt.day_of_week
    df["fim_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["cidade_codigo"] = df["entregador_id"].str.extract(r"^([A-Z]+)RES", expand=False)

    rel["linhas_finais"] = len(df)
    return df.drop(columns=["hora_pedido", "hora_coleta"]), rel


def carregar() -> pd.DataFrame:
    if not BRUTO.exists():
        raise FileNotFoundError(f"{BRUTO} não existe. Rode: python -m entrega.dados --baixar")
    return limpar(pd.read_csv(BRUTO))[0]


def dividir_por_data(df: pd.DataFrame, frac_teste: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Treina no passado e testa no futuro: os últimos ~20% dos pedidos (por data) viram teste."""
    corte = df["data"].quantile(1 - frac_teste)
    return df[df["data"] < corte], df[df["data"] >= corte], corte


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baixar", action="store_true")
    if ap.parse_args().baixar:
        baixar()
    df, rel = limpar(pd.read_csv(BRUTO))
    for k, v in rel.items():
        print(f"{k:35s} {v}")

"""
API de previsão do tempo de entrega.

    uvicorn entrega.api:app --reload        # documentação interativa em http://127.0.0.1:8000/docs

Exemplo:
    curl -X POST http://127.0.0.1:8000/prever -H "Content-Type: application/json" \
         -d '{"distancia_km": 6.5, "transito": "High", "clima": "Stormy", "veiculo": "motorcycle",
              "tipo_pedido": "Meal", "hora": 20}'
"""

from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from entrega.servico import carregar_pacotes, prever

Clima = Literal["Sunny", "Cloudy", "Fog", "Windy", "Stormy", "Sandstorms"]
Transito = Literal["Low", "Medium", "High", "Jam"]
TipoPedido = Literal["Snack", "Meal", "Drinks", "Buffet"]
Veiculo = Literal["motorcycle", "scooter", "electric_scooter", "bicycle"]
Cidade = Literal["Metropolitian", "Urban", "Semi-Urban"]  # grafia do dataset original


class Pedido(BaseModel):
    """Dados do pedido no momento em que ele é feito. Campos opcionais podem ficar de fora."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "distancia_km": 6.5, "transito": "High", "clima": "Stormy", "veiculo": "motorcycle",
        "tipo_pedido": "Meal", "hora": 20, "entregas_simultaneas": 1, "nota_entregador": 4.6,
    }]})

    distancia_km: float = Field(gt=0, le=50, description="Distância em linha reta restaurante → cliente")
    transito: Transito
    clima: Clima
    veiculo: Veiculo
    tipo_pedido: TipoPedido
    hora: int = Field(ge=0, le=23, description="Hora do pedido (0 a 23)")
    cidade: Cidade | None = None
    idade_entregador: float | None = Field(default=None, ge=18, le=70)
    nota_entregador: float | None = Field(default=None, ge=1, le=5)
    condicao_veiculo: int | None = Field(default=None, ge=0, le=3, description="0 = pior, 3 = melhor")
    entregas_simultaneas: int | None = Field(default=None, ge=0, le=3, description="Pedidos que o entregador leva junto")
    festival: bool = False
    espera_coleta_min: float | None = Field(default=None, ge=0, le=60, description="Minutos até o entregador coletar")
    dia_semana: int | None = Field(default=None, ge=0, le=6, description="0 = segunda ... 6 = domingo")


class Previsao(BaseModel):
    previsao_min: int = Field(description="Melhor estimativa, em minutos")
    minimo_min: int
    maximo_min: int = Field(description="Valor a prometer ao cliente: só ~10% dos pedidos passam dele")
    texto: str
    cobertura_pct: int = Field(description="Em quantos % dos pedidos o tempo real cai dentro do intervalo")
    modo: Literal["basico", "completo"] = Field(
        description="completo = com nota, idade e condição do veículo do entregador; básico = só o que existe no pedido")


def criar_app(carregador: Callable[[], dict] = carregar_pacotes) -> FastAPI:
    """Fábrica: os testes passam um carregador próprio, sem depender dos arquivos treinados."""

    @asynccontextmanager
    async def vida(app: FastAPI):
        try:
            app.state.pacotes = carregador()
        except Exception as e:  # a API sobe mesmo sem modelo; /prever explica o problema
            app.state.pacotes, app.state.erro = None, str(e)
        yield

    app = FastAPI(
        title="Previsão do tempo de entrega",
        description="Estimativa do tempo de entrega com intervalo calibrado (80%), em dois modos: básico e completo.",
        version="1.0.0", lifespan=vida,
    )

    @app.get("/saude")
    def saude(request: Request):
        return {"ok": request.app.state.pacotes is not None}

    @app.post("/prever", response_model=Previsao)
    def prever_entrega(pedido: Pedido, request: Request):
        pacotes = request.app.state.pacotes
        if pacotes is None:
            raise HTTPException(503, f"Modelo indisponível: {getattr(request.app.state, 'erro', 'não carregado')}")
        return prever(pacotes, [pedido.model_dump()])[0]

    return app


app = criar_app()

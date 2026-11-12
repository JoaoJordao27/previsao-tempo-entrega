"""Tela de demonstração: escolha os dados do pedido e veja o tempo de entrega previsto, com intervalo."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from entrega.servico import carregar_pacotes, prever

RAIZ = Path(__file__).resolve().parent
LARANJA, AZUL, VERDE, CINZA = "#E4572E", "#1F3A5F", "#2E8B57", "#6B7785"

CLIMA = {"Sunny": "Ensolarado", "Cloudy": "Nublado", "Fog": "Neblina", "Windy": "Ventando",
         "Stormy": "Tempestade", "Sandstorms": "Tempestade de areia"}
TRANSITO = {"Low": "Baixo", "Medium": "Médio", "High": "Alto", "Jam": "Congestionado"}
TIPO = {"Snack": "Lanche", "Meal": "Refeição", "Drinks": "Bebidas", "Buffet": "Buffet"}
VEICULO = {"motorcycle": "Moto", "scooter": "Scooter", "electric_scooter": "Scooter elétrica", "bicycle": "Bicicleta"}
DIAS = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
FATORES = {  # nome na importância -> rótulo para o usuário
    "clima": "Clima", "transito": "Trânsito", "distancia_km": "Distância", "idade_entregador": "Idade do entregador",
    "nota_entregador": "Nota do entregador", "condicao_veiculo": "Condição do veículo",
    "entregas_simultaneas": "Entregas simultâneas", "festival": "Festival", "hora": "Hora do pedido",
    "cidade": "Tipo de cidade", "fim_de_semana": "Fim de semana", "tipo_pedido": "Tipo de pedido",
    "veiculo": "Veículo", "espera_coleta_min": "Espera até a coleta", "dia_semana": "Dia da semana",
}

st.set_page_config(page_title="Previsão do tempo de entrega", page_icon="🛵", layout="wide")
st.markdown(f"""
<style>
.block-container {{max-width: 1100px; padding-top: 2rem}}
h1 {{color: {AZUL}; letter-spacing: -0.02em}}
.resultado {{background: {AZUL}; color: #fff; border-radius: 16px; padding: 22px 26px}}
.resultado .rotulo {{font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; opacity: .75}}
.resultado .faixa {{font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin: 4px 0 2px}}
.resultado .sub {{opacity: .85}}
.escala {{position: relative; height: 54px; margin-top: 18px}}
.escala .trilho {{position: absolute; top: 20px; left: 0; right: 0; height: 8px; background: rgba(255,255,255,.18); border-radius: 4px}}
.escala .faixa-barra {{position: absolute; top: 20px; height: 8px; background: {LARANJA}; border-radius: 4px}}
.escala .ponto {{position: absolute; top: 14px; width: 4px; height: 20px; background: #fff; border-radius: 2px}}
.escala .marca {{position: absolute; top: 34px; font-size: .75rem; opacity: .8; transform: translateX(-50%); white-space: nowrap}}
.aviso {{color: {CINZA}; font-size: .85rem}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def modelos():
    return carregar_pacotes()


def ler_json(nome: str) -> dict | None:
    arq = RAIZ / "resultados" / nome
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None


def escala_html(minimo: int, previsao: int, maximo: int, topo: int = 70) -> str:
    pos = lambda v: min(max(v / topo * 100, 0), 100)
    marcas = "".join(f'<div class="marca" style="left:{pos(v)}%">{v}</div>' for v in (minimo, previsao, maximo))
    return (f'<div class="escala"><div class="trilho"></div>'
            f'<div class="faixa-barra" style="left:{pos(minimo)}%;width:{pos(maximo) - pos(minimo)}%"></div>'
            f'<div class="ponto" style="left:calc({pos(previsao)}% - 2px)"></div>{marcas}</div>')


def efeito_de_cada_fator(pedido: dict) -> pd.DataFrame:
    """Quanto o tempo cairia se cada fator estivesse na melhor condição (mantendo o resto igual)."""
    melhores = {"Trânsito baixo": {"transito": "Low"}, "Clima ensolarado": {"clima": "Sunny"},
                "Sem entregas simultâneas": {"entregas_simultaneas": 0}, "Distância de 2 km": {"distancia_km": 2.0}}
    base = prever(modelos(), [pedido])[0]["previsao_min"]
    cenarios = [(nome, {**pedido, **troca}) for nome, troca in melhores.items()
                if any(pedido.get(k) != v for k, v in troca.items())]
    if not cenarios:
        return pd.DataFrame(columns=["cenario", "ganho_min"])
    prevs = prever(modelos(), [c for _, c in cenarios])
    linhas = [(nome, base - p["previsao_min"]) for (nome, _), p in zip(cenarios, prevs)]
    return pd.DataFrame(linhas, columns=["cenario", "ganho_min"]).query("ganho_min > 0").sort_values("ganho_min")


st.title("Previsão do tempo de entrega")
st.markdown("Estime quanto um pedido vai demorar **antes de ele ser feito**, com um intervalo em que o tempo real "
            "cai em 80% dos casos.")
aba_prever, aba_modelo = st.tabs(["Prever um pedido", "Sobre o modelo"])

with aba_prever:
    col_form, col_res = st.columns([5, 6], gap="large")
    with col_form:
        st.subheader("Dados do pedido")
        distancia = st.slider("Distância até o cliente (km)", 0.5, 25.0, 6.5, 0.5)
        a, b = st.columns(2)
        transito = a.selectbox("Trânsito", list(TRANSITO), index=2, format_func=TRANSITO.get)
        clima = b.selectbox("Clima", list(CLIMA), index=4, format_func=CLIMA.get)
        c, d = st.columns(2)
        veiculo = c.selectbox("Veículo", list(VEICULO), format_func=VEICULO.get)
        tipo = d.selectbox("Tipo de pedido", list(TIPO), index=1, format_func=TIPO.get)
        e, f = st.columns(2)
        hora = e.slider("Hora do pedido", 0, 23, 20)
        simultaneas = f.selectbox("Entregas simultâneas", [0, 1, 2, 3], index=1,
                                  help="Quantos pedidos o entregador leva junto.")
        festival = st.toggle("Dia de festival", value=False)
        with st.expander("Mais detalhes (opcional)"):
            st.caption("Quanto mais informação, mais estreito o intervalo.")
            informa_entregador = st.checkbox("Informar dados do entregador")
            nota = idade = condicao = None
            if informa_entregador:
                nota = st.slider("Nota do entregador", 1.0, 5.0, 4.6, 0.1)
                idade = st.slider("Idade do entregador", 18, 60, 30)
                condicao = st.select_slider("Condição do veículo (0 = pior, 3 = melhor)", [0, 1, 2, 3], value=2)
            dia = st.selectbox("Dia da semana", [None, *range(7)], format_func=lambda i: "Não informar" if i is None else DIAS[i])

    pedido = {"distancia_km": distancia, "transito": transito, "clima": clima, "veiculo": veiculo,
              "tipo_pedido": tipo, "hora": hora, "entregas_simultaneas": simultaneas, "festival": festival,
              "nota_entregador": nota, "idade_entregador": idade, "condicao_veiculo": condicao, "dia_semana": dia}

    with col_res:
        try:
            r = prever(modelos(), [pedido])[0]
        except FileNotFoundError as erro:
            st.error(str(erro))
            st.stop()
        st.markdown(
            f'<div class="resultado"><div class="rotulo">Tempo previsto</div>'
            f'<div class="faixa">{r["minimo_min"]} a {r["maximo_min"]} min</div>'
            f'<div class="sub">Melhor estimativa: <b>{r["previsao_min"]} min</b> · '
            f'o tempo real cai nesta faixa em {r["cobertura_pct"]}% dos pedidos</div>'
            f'{escala_html(r["minimo_min"], r["previsao_min"], r["maximo_min"])}</div>',
            unsafe_allow_html=True)
        st.markdown(f'<p class="aviso">Para o cliente, prometa o máximo ({r["maximo_min"]} min): só cerca de 10% '
                    f'dos pedidos demoram mais que isso.</p>', unsafe_allow_html=True)

        st.subheader("O que mais ajudaria a reduzir o tempo")
        efeito = efeito_de_cada_fator(pedido)
        if efeito.empty:
            st.success("Este pedido já está nas melhores condições para os fatores avaliados.")
        else:
            fig = go.Figure(go.Bar(x=efeito.ganho_min, y=efeito.cenario, orientation="h", marker_color=VERDE,
                                   text=[f"−{g} min" for g in efeito.ganho_min], textposition="outside", cliponaxis=False))
            fig.update_layout(height=60 + 48 * len(efeito), margin=dict(l=0, r=80, t=10, b=0), showlegend=False,
                              xaxis=dict(title="Minutos economizados", showgrid=False), yaxis_title=None,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Cada barra muda só um fator, mantendo o resto do pedido igual.")

with aba_modelo:
    metricas, interv = ler_json("metricas.json"), ler_json("intervalo.json")
    st.subheader("O modelo ganha das alternativas simples?")
    if metricas:
        linhas = [{"Modelo": nome[0].upper() + nome[1:], "Erro médio (min)": f'{m["mae_min"]:.2f}', "Acerta em até 5 min": f'{m["acerta_ate_5min_pct"]}%',
                   "Chega >5 min depois": f'{m["atraso_acima_5min_pct"]}%'} for nome, m in metricas["modelos"].items()]
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")
        n_teste = f"{metricas['linhas_teste']:,}".replace(",", ".")
        data_corte = pd.Timestamp(metricas["corte_treino_teste"]).strftime("%d/%m/%Y")
        st.caption(f"Avaliado em {n_teste} pedidos de {data_corte} em diante, que o modelo não viu no treino.")
        imp = pd.Series(metricas["importancia_por_permutacao_min"]).sort_values().tail(8)
        fig = go.Figure(go.Bar(x=imp.values, y=[FATORES.get(k, k) for k in imp.index], orientation="h", marker_color=AZUL))
        fig.update_layout(height=340, title="O que mais pesa na previsão (aumento do erro ao embaralhar o fator)",
                          margin=dict(l=0, r=10, t=40, b=0), xaxis_title="minutos", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if interv:
        st.subheader("O intervalo é confiável?")
        c1, c2 = st.columns(2)
        cru, cal = interv["quantis_crus"], interv["calibrado"]
        c1.metric("Sem calibração", f'{cru["cobertura_pct"]}%', help="Meta: 80%")
        c2.metric("Com calibração conformal", f'{cal["cobertura_pct"]}%', delta=f'{cal["cobertura_pct"] - cru["cobertura_pct"]:+.1f} p.p.',
                  help="Meta: 80%")
        st.caption("Percentual de pedidos em que o tempo real ficou dentro da faixa prevista. "
                   "O ajuste custa menos de 1 minuto de largura média.")
    st.subheader("Limitações")
    st.markdown("""
- Os dados são um conjunto público do Kaggle sobre entregas na **Índia, de fev a abr/2022** (44 dias).
  **Isto não representa o iFood nem entregas no Brasil**; é um estudo de método.
- Hora do pedido e dia da semana pesam quase nada, o que é incomum em entregas reais. É um indício de que
  parte dos dados é artificial, então os números absolutos não devem ser levados a sério.
- A nota do entregador pesa bastante. Em produção seria preciso checar se ela já reflete entregas lentas
  anteriores (vazamento de informação).
- O teste cobre apenas os dias seguintes ao treino; não mede mudanças de sazonalidade ao longo de meses.
- O modelo não usa rota nem GPS: a distância é em linha reta.
""")

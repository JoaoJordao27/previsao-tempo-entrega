"""Tela de demonstração: escolha os dados do pedido e veja o tempo de entrega previsto, com intervalo."""

import json
from pathlib import Path

import streamlit as st

from entrega.servico import carregar_pacotes, prever

RAIZ = Path(__file__).resolve().parent

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
PAGINAS = ["Prever um pedido", "Sobre o modelo"]
PADRAO = {"pagina": PAGINAS[0], "dist": 6.5, "transito": "High", "clima": "Stormy", "veiculo": "motorcycle",
          "tipo": "Meal", "hora": 20, "simult": 1, "festival_on": False, "info_ent": False, "nota": 4.6, "idade": 30,
          "condicao": 2, "dia": None}
LOGO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#E8572A" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="13" r="8"/>'
        '<path d="M12 9v4l2.5 2.5"/><path d="M9 2h6"/></svg>')
ESCUDO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FF8A5C" stroke-width="2" '
          'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
          '<path d="M9 12l2 2 4-4"/></svg>')

st.set_page_config(page_title="Previsão do tempo de entrega", page_icon=str(RAIZ / "assets" / "icone.png"),
                   layout="wide")
st.markdown(f"<style>{(RAIZ / 'estilo.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
for _k, _v in PADRAO.items():
    st.session_state.setdefault(_k, _v)


def br(valor: float, casas: int = 1) -> str:
    """Número no padrão brasileiro (vírgula decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


@st.cache_resource
def modelos():
    return carregar_pacotes()


def ler_json(nome: str) -> dict | None:
    arq = RAIZ / "resultados" / nome
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else None


def limpar():
    for chave, valor in PADRAO.items():
        if chave != "pagina":
            st.session_state[chave] = valor


def manter(chave: str):
    """Controles segmentados devolvem None ao clicar no item já marcado; aqui voltamos ao valor anterior."""
    if st.session_state.get(chave) is None:
        st.session_state[chave] = st.session_state.get("_ult_" + chave, PADRAO[chave])


def segmentado(chave: str, opcoes: dict, rotulo: str, pills: bool = False):
    st.markdown(f'<div class="rot-simples">{rotulo}</div>', unsafe_allow_html=True)
    with st.container(key=f"seg_{chave}"):
        controle = st.pills if pills else st.segmented_control
        controle(rotulo, list(opcoes), format_func=opcoes.get, key=chave, label_visibility="collapsed",
                 selection_mode="single", on_change=manter, args=(chave,))
    valor = st.session_state.get(chave) or PADRAO[chave]
    st.session_state["_ult_" + chave] = valor
    return valor


def slider_com_rotulo(rotulo: str, chave: str, minimo, maximo, passo, formatar):
    """Rótulo à esquerda e valor grande à direita, como no desenho (o valor interno do slider fica oculto)."""
    ph = st.empty()
    valor = st.slider(rotulo, minimo, maximo, step=passo, key=chave, label_visibility="collapsed")
    ph.markdown(f'<div class="rot-linha"><span class="rot">{rotulo}</span><span class="val">{formatar(valor)}</span></div>',
                unsafe_allow_html=True)
    return valor


def escala_html(minimo: int, previsao: int, maximo: int) -> str:
    topo = max(60, ((maximo + 9) // 10) * 10)
    pos = lambda v: min(max(v / topo * 100, 0), 100)
    return (f'<div><div class="escala"><div class="trilho"></div>'
            f'<div class="faixa" style="left:{pos(minimo):.2f}%;width:{pos(maximo) - pos(minimo):.2f}%"></div>'
            f'<div class="ponto" style="left:{pos(previsao):.2f}%"></div>'
            f'<div class="topo-v" style="left:{pos(previsao):.2f}%">{previsao}</div>'
            f'<div class="ext" style="left:{pos(minimo):.2f}%">{minimo}</div>'
            f'<div class="ext" style="left:{pos(maximo):.2f}%">{maximo}</div></div>'
            f'<div class="escala-pe"><span>0 min</span><span>Faixa com 80% dos pedidos</span><span>{topo} min</span></div></div>')


def resultado_html(r: dict, p: dict) -> str:
    n = p["entregas_simultaneas"]
    chips = [f'{br(p["distancia_km"])} km', f'Trânsito {TRANSITO[p["transito"]].lower()}', CLIMA[p["clima"]],
             f'{VEICULO[p["veiculo"]]} · {TIPO[p["tipo_pedido"]]}',
             f'{p["hora"]}h · ' + ("sem simultâneas" if n == 0 else f"{n} simultânea" + ("s" if n > 1 else ""))]
    if p["festival"]:
        chips.append("Dia de festival")
    return (f'<div class="res"><div class="topo"><div><div class="rotulo">Tempo previsto</div>'
            f'<div class="grande">{r["minimo_min"]}<span class="sep"> – </span>{r["maximo_min"]}<span class="un"> min</span></div></div>'
            f'<div class="est"><div class="t">Melhor estimativa</div><div class="v">{r["previsao_min"]} <small>min</small></div></div></div>'
            + escala_html(r["minimo_min"], r["previsao_min"], r["maximo_min"]) +
            f'<div class="dica">{ESCUDO}<p><strong>Prometa {r["maximo_min"]} min ao cliente.</strong> '
            f'Só cerca de 10% dos pedidos demoram mais que isso.</p></div>'
            f'<div class="chips">{"".join(f"<span>{c}</span>" for c in chips)}</div></div>')


def efeito_de_cada_fator(p: dict) -> list[tuple[str, str, int]]:
    """Quanto o tempo cairia se cada fator estivesse na melhor condição, mantendo o resto igual."""
    n = p["entregas_simultaneas"]
    cenarios = []
    if p["transito"] != "Low":
        cenarios.append(("Trânsito baixo", f'em vez de {TRANSITO[p["transito"]].lower()}', {"transito": "Low"}))
    if p["clima"] != "Sunny":
        cenarios.append(("Clima ensolarado", f'em vez de {CLIMA[p["clima"]].lower()}', {"clima": "Sunny"}))
    if n != 0:
        cenarios.append(("Sem entregas simultâneas", f"em vez de {n}", {"entregas_simultaneas": 0}))
    if p["distancia_km"] > 2:
        cenarios.append(("Distância de 2 km", f'em vez de {br(p["distancia_km"])} km', {"distancia_km": 2.0}))
    if not cenarios:
        return []
    base = prever(modelos(), [p])[0]["previsao_min"]
    prevs = prever(modelos(), [{**p, **troca} for _, _, troca in cenarios])
    ganhos = [(nome, sub, base - x["previsao_min"]) for (nome, sub, _), x in zip(cenarios, prevs)]
    return sorted([g for g in ganhos if g[2] > 0], key=lambda g: -g[2])


def reducao_html(ganhos: list) -> str:
    cab = ('<div class="card"><h2>O que mais ajudaria a reduzir o tempo</h2>'
           '<p class="sub">Cada linha muda só um fator, mantendo o resto do pedido igual.</p>')
    if not ganhos:
        return cab + '<p class="vazio">Este pedido já está nas melhores condições para os fatores avaliados.</p></div>'
    maior = max(g[2] for g in ganhos)
    linhas = "".join(
        f'<div class="barra"><div class="nome">{nome}<small>{sub}</small></div>'
        f'<div class="fundo"><i style="width:{g / maior * 100:.1f}%"></i></div><div class="num">−{g}</div></div>'
        for nome, sub, g in ganhos)
    return cab + f'<div class="barras">{linhas}</div><div class="rodape-barras">minutos economizados</div></div>'


def pagina_prever():
    st.markdown('<div class="hero"><div class="eyebrow"><i></i>Antes de o pedido ser feito</div>'
                '<h1>Quanto tempo esse pedido vai levar?</h1>'
                '<p>Preencha as condições da entrega e veja a faixa em que o tempo real cai em <strong>80% dos casos</strong>'
                ' — e o que mais pesa no resultado.</p></div>', unsafe_allow_html=True)
    col_form, col_res = st.columns([5, 7], gap="large")
    with col_form:
        with st.container(key="form"):
            a, b = st.columns([3, 1], vertical_alignment="center")
            a.markdown('<div class="form-titulo"><h2>Dados do pedido</h2></div>', unsafe_allow_html=True)
            with b.container(key="limpar"):
                st.button("Limpar", type="tertiary", on_click=limpar)
            dist = slider_com_rotulo("Distância até o cliente", "dist", 0.5, 25.0, 0.5,
                                     lambda v: f"{br(v)} <small>km</small>")
            transito = segmentado("transito", TRANSITO, "Trânsito")
            clima = segmentado("clima", CLIMA, "Clima", pills=True)
            c, d = st.columns(2)
            veiculo = c.selectbox("Veículo", list(VEICULO), format_func=VEICULO.get, key="veiculo")
            tipo = d.selectbox("Tipo de pedido", list(TIPO), format_func=TIPO.get, key="tipo")
            e, f = st.columns(2)
            with e:
                hora = slider_com_rotulo("Hora do pedido", "hora", 0, 23, 1, lambda v: f"{v}h")
            simult = f.selectbox("Entregas simultâneas", [0, 1, 2, 3], key="simult",
                                 help="Quantos pedidos o entregador leva junto.")
            with st.container(key="festival"):
                festival = st.toggle("Dia de festival", key="festival_on")
                st.caption("Datas com pico de demanda")
            with st.expander("Mais detalhes (opcional)"):
                st.caption("Quanto mais informação, mais estreito o intervalo.")
                nota = idade = condicao = None
                if st.checkbox("Informar dados do entregador", key="info_ent"):
                    nota = st.slider("Nota do entregador", 1.0, 5.0, step=0.1, key="nota")
                    idade = st.slider("Idade do entregador", 18, 60, key="idade")
                    condicao = st.select_slider("Condição do veículo (0 = pior, 3 = melhor)", [0, 1, 2, 3], key="condicao")
                dia = st.selectbox("Dia da semana", [None, *range(7)], key="dia",
                                   format_func=lambda i: "Não informar" if i is None else DIAS[i])

    pedido = {"distancia_km": dist, "transito": transito, "clima": clima, "veiculo": veiculo, "tipo_pedido": tipo,
              "hora": hora, "entregas_simultaneas": simult, "festival": festival, "nota_entregador": nota,
              "idade_entregador": idade, "condicao_veiculo": condicao, "dia_semana": dia}
    with col_res:
        try:
            r = prever(modelos(), [pedido])[0]
        except FileNotFoundError as erro:
            st.error(str(erro))
            st.stop()
        st.markdown(resultado_html(r, pedido), unsafe_allow_html=True)
        st.markdown(reducao_html(efeito_de_cada_fator(pedido)), unsafe_allow_html=True)


def pagina_modelo():
    metricas, interv = ler_json("metricas.json"), ler_json("intervalo.json")
    st.markdown('<div class="hero"><div class="eyebrow"><i></i>Como foi avaliado</div>'
                '<h1>O que está por trás da previsão</h1>'
                '<p>Comparação com alternativas simples, os fatores que mais pesam, a confiabilidade do intervalo '
                'e o que este estudo <strong>não</strong> garante.</p></div>', unsafe_allow_html=True)
    if metricas:
        linhas = "".join(
            f'<tr class="{"dest" if nome == "LightGBM" else ""}"><td>{nome[0].upper() + nome[1:]}</td>'
            f'<td class="n">{br(m["mae_min"], 2)}</td><td class="n">{br(m["acerta_ate_5min_pct"])}%</td>'
            f'<td class="n">{br(m["atraso_acima_5min_pct"])}%</td></tr>' for nome, m in metricas["modelos"].items())
        n_teste = f"{metricas['linhas_teste']:,}".replace(",", ".")
        corte = "/".join(reversed(metricas["corte_treino_teste"].split("-")))
        st.markdown('<div class="card"><h2>O modelo ganha das alternativas simples?</h2>'
                    '<p class="sub">Se não ganhasse, não valeria a complexidade.</p><div class="tab-wrap"><table class="tab"><thead><tr>'
                    '<th>Modelo</th><th class="n">Erro médio (min)</th><th class="n">Acerta em até 5 min</th>'
                    f'<th class="n">Chega &gt;5 min depois</th></tr></thead><tbody>{linhas}</tbody></table></div>'
                    f'<p class="nota">Avaliado em {n_teste} pedidos de {corte} em diante, que o modelo não viu no treino.</p></div>',
                    unsafe_allow_html=True)
        imp = sorted(metricas["importancia_por_permutacao_min"].items(), key=lambda kv: -kv[1])[:8]
        maior = max(v for _, v in imp)
        barras = "".join(
            f'<div class="barra"><div class="nome">{FATORES.get(k, k)}</div>'
            f'<div class="fundo ink"><i style="width:{max(v, 0) / maior * 100:.1f}%"></i></div>'
            f'<div class="num ink">{br(v)} min</div></div>' for k, v in imp)
        st.markdown('<div class="card"><h2>O que mais pesa na previsão</h2>'
                    '<p class="sub">Quanto o erro piora ao embaralhar cada fator (importância por permutação).</p>'
                    f'<div class="barras">{barras}</div></div>', unsafe_allow_html=True)
    if interv:
        cru, cal = interv["quantis_crus"], interv["calibrado"]
        st.markdown('<div class="card"><h2>O intervalo é confiável?</h2>'
                    '<p class="sub">Percentual de pedidos em que o tempo real ficou dentro da faixa prevista. Meta: 80%.</p>'
                    f'<div class="duo"><div class="stat"><div class="t">Sem calibração</div><div class="v">{br(cru["cobertura_pct"])}%</div></div>'
                    f'<div class="stat dest"><div class="t">Com calibração conformal</div><div class="v">{br(cal["cobertura_pct"])}%</div>'
                    f'<span class="d">+{br(cal["cobertura_pct"] - cru["cobertura_pct"])} p.p.</span></div></div>'
                    '<p class="nota">O ajuste custa menos de 1 minuto de largura média.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><h2>Limitações</h2><p class="sub">O que este estudo não garante.</p><ul class="lim">'
                '<li>Os dados são um conjunto público do Kaggle sobre entregas na <strong>Índia, de fev a abr/2022</strong> (44 dias). '
                '<strong>Isto não representa o iFood nem entregas no Brasil</strong>; é um estudo de método.</li>'
                '<li>Hora do pedido e dia da semana pesam quase nada, o que é incomum em entregas reais. É um indício de que '
                'parte dos dados é artificial, então os números absolutos não devem ser levados a sério.</li>'
                '<li>A nota do entregador pesa bastante. Em produção seria preciso checar se ela já reflete entregas lentas '
                'anteriores (vazamento de informação).</li>'
                '<li>O teste cobre apenas os dias seguintes ao treino; não mede mudanças de sazonalidade ao longo de meses.</li>'
                '<li>O modelo não usa rota nem GPS: a distância é em linha reta.</li></ul></div>', unsafe_allow_html=True)


with st.container(key="topo"):
    esq, dir_ = st.columns([1, 1], vertical_alignment="center")
    esq.markdown(f'<div class="marca"><div class="logo">{LOGO}</div><div><div class="nome">ETA Delivery</div>'
                 '<div class="sub">Previsão do tempo de entrega</div></div></div>', unsafe_allow_html=True)
    with dir_.container(key="nav"):
        st.segmented_control("Seção", PAGINAS, key="pagina", label_visibility="collapsed", selection_mode="single",
                             on_change=manter, args=("pagina",))
pagina = st.session_state.get("pagina") or PAGINAS[0]
st.session_state["_ult_pagina"] = pagina

if pagina == PAGINAS[0]:
    pagina_prever()
else:
    pagina_modelo()

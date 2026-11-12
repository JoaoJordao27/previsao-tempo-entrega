# Previsão do tempo de entrega

Modelo de machine learning que estima **quanto um pedido de delivery vai demorar, antes de ele ser feito**, e devolve não um número solto, mas um **intervalo calibrado**: "entre 18 e 37 min", em que o tempo real cai em 80% dos pedidos. Vem com API (FastAPI) e uma tela de demonstração (Streamlit).

🚀 **[Experimente online](https://previsao-tempo-entrega.streamlit.app/)**

![Tela de previsão](docs/prever.png)

## Por que um intervalo, e não um número

Um app que diz "chega em 26 min" erra para menos em metade dos pedidos. Para o cliente, **chegar depois do prometido é pior do que chegar antes**. Por isso o projeto entrega:

- a **melhor estimativa** (26 min);
- um **intervalo** (18 a 37 min) calibrado para cobrir 80% dos casos;
- a recomendação de **prometer o máximo** (37 min): só cerca de 10% dos pedidos passam dele.

## Resultados

Avaliação em 9.447 pedidos de 29/03/2022 em diante, que o modelo **não viu no treino** (a divisão é por data, não aleatória: treina-se no passado e testa-se no futuro).

Há **dois modos**, porque na hora do pedido o entregador ainda não foi designado: o **básico** usa só o que existe no pedido (clima, trânsito, distância, hora, entregas simultâneas...); o **completo** também conhece nota, idade e condição do veículo do entregador.

| Modelo | Erro médio | Acerta em até 5 min | Chega >5 min depois do previsto |
|---|---|---|---|
| Média geral | 7,65 min | 38,2% | 27,9% |
| Regra por trânsito e entregas simultâneas | 6,28 min | 48,3% | 25,2% |
| **LightGBM, modo básico** | **4,75 min** | **60,6%** | **19,1%** |
| **LightGBM, modo completo** | **3,20 min** | **80,1%** | **10,1%** |

### O intervalo é confiável?

| Modo | Quantis crus | Calibrado | Largura média | Chega depois do máximo |
|---|---|---|---|---|
| Básico | 74,9% | **80,9%** | 14,6 min | 9,6% |
| Completo | 72,4% | **80,5%** | 9,2 min | 9,9% |

A regressão quantílica sozinha prometia 80% e entregava 72% a 75%. A **calibração conformal** mede esse erro num pedaço separado dos dados e ajusta o intervalo na medida certa.

**Um detalhe que quase passou despercebido:** o tempo real é medido em minutos inteiros e a faixa mostrada também. Calibrar em minutos contínuos e só arredondar depois fazia a faixa que a pessoa vê cobrir **83% a 88%** (não 80%), com largura desnecessária. Por isso a calibração é feita já sobre os limites inteiros. A cobertura se mantém perto de 80% nos grupos testados (de 77,3% a 82,7% no modo básico e de 79,0% a 82,0% no completo, por nível de trânsito e por número de entregas simultâneas).

Saber quem é o entregador estreita a faixa de 14,6 para 9,2 min.

![O que mais pesa e confiabilidade do intervalo](docs/modelo.png)

## Como funciona

```
dados brutos (Kaggle) ─► limpeza e atributos ─► LightGBM (estimativa)         ┐ um conjunto
                                              └► LightGBM quantílico 10% e 90% ─► calibração (limites inteiros) ─► intervalo   ┘ por modo (básico e completo)
                                                                                          │
                                                       API FastAPI  /prever  ◄────────────┴────────► tela Streamlit
```

1. **Limpeza** (`entrega/dados.py`): corrige latitudes com sinal trocado, trata coordenadas zeradas e notas fora da escala como dado faltante, calcula a distância em km (fórmula de Haversine) e a espera até a coleta (tratando virada de meia-noite).
2. **Bases de comparação** (`entrega/modelo.py`): média geral e uma regra feita à mão. Se o modelo não ganha delas, não vale a complexidade.
3. **Validação por data**: embaralhar misturaria passado e futuro e inflaria os números.
4. **Importância por permutação**: mede quanto o erro piora ao embaralhar cada fator. É mais confiável do que a importância interna do LightGBM.
5. **Intervalo conformal** (`entrega/intervalo.py`): quantis 10% e 90%, calibrados em um conjunto separado (mais recente que o treino, mais antigo que o teste), já sobre limites em minutos inteiros e separadamente para cada modo.
6. **Uso** (`entrega/servico.py`): módulo único usado pela API e pela tela, então os dois dão sempre o mesmo resultado. Os modelos ficam em arquivos de texto do LightGBM, que funcionam em qualquer versão de Python.

## Como rodar

```bash
pip install -r requirements-dev.txt

# 1. baixar o dataset (precisa de ~/.kaggle/kaggle.json)
python -m entrega.dados --baixar

# 2. treinar e calibrar (gera os arquivos em modelos/ e resultados/)
python -m entrega.modelo
python -m entrega.intervalo

# 3. usar
streamlit run app.py                    # tela de demonstração
uvicorn entrega.api:app --reload        # API, documentação em http://127.0.0.1:8000/docs
python -m pytest -q                     # testes
```

Os modelos treinados já vêm no repositório (`modelos/`, cerca de 20 MB), então tela e API funcionam sem baixar o dataset nem treinar.

### API

```bash
curl -X POST http://127.0.0.1:8000/prever -H "Content-Type: application/json" \
  -d '{"distancia_km": 18, "transito": "Jam", "clima": "Stormy", "veiculo": "scooter",
       "tipo_pedido": "Meal", "hora": 20, "entregas_simultaneas": 2}'
```

```json
{"previsao_min": 39, "minimo_min": 34, "maximo_min": 44, "texto": "entre 34 e 44 min", "cobertura_pct": 80, "modo": "basico"}
```

Informando também `nota_entregador`, `idade_entregador` e `condicao_veiculo`, a API passa para o **modo completo** (`"modo": "completo"`), com faixa mais estreita.

- Só 6 campos são obrigatórios (distância, trânsito, clima, veículo, tipo de pedido, hora). Os demais são opcionais. O modo completo exige os três dados do entregador juntos; sem eles, vale o modo básico.
- Valores impossíveis (hora 24, nota 6, distância negativa, categoria inventada, campo desconhecido) são recusados com erro 422.
- Sem modelo treinado, a API sobe e responde 503 explicando o que rodar.

## Testes

39 testes (`python -m pytest -q`), sem precisar do dataset:

- **Limpeza**: conversão de textos, coordenadas inválidas, espera que passa da meia-noite, fórmula de distância contra um valor conhecido, divisão por data sem misturar passado e futuro.
- **Métricas e bases**: separação entre chegar antes e chegar depois, comportamento da regra com combinação nunca vista.
- **Calibração**: com dados simulados, um intervalo estreito demais passa a cobrir de fato 80% dos casos novos, inclusive com tempos inteiros e limites arredondados.
- **API**: respostas coerentes (mínimo ≤ estimativa ≤ máximo, limites inteiros), escolha do modo, validação de entradas, 503 sem modelo.
- **Persistência**: o modelo salvo e recarregado dá exatamente o mesmo resultado que o original.

## Limitações

- **Os dados não são do Brasil nem de uma plataforma específica.** São um conjunto público do Kaggle ([Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)) sobre entregas na Índia, de fev a abr/2022 (44 dias). O projeto é um estudo de método, e os números absolutos não se transferem para outro contexto.
- **Parte dos dados parece artificial.** A hora do pedido e o dia da semana pesam quase nada, o que é incomum em entregas reais.
- **Nota, idade e condição do veículo do entregador pesam bastante**, mas só existiriam depois que um entregador fosse designado; por isso o modo básico é o padrão. Seria preciso checar também se a nota já reflete entregas lentas anteriores (vazamento de informação).
- O teste cobre apenas os dias seguintes ao treino: não mede mudanças de sazonalidade ao longo de meses.
- A distância é em linha reta, sem rota nem GPS.
- O arquivo do dataset **não é redistribuído** aqui (licença "other"); ele é baixado pelo comando acima.

## Estrutura

```
entrega/dados.py       carga, limpeza e atributos
entrega/modelo.py      bases, LightGBM, validação por data, importância por permutação
entrega/intervalo.py   quantis + calibração conformal
entrega/servico.py     salvar/carregar modelos e prever (usado pela API e pela tela)
entrega/api.py         FastAPI
app.py                 tela Streamlit
modelos/               modelos treinados (texto) e metadados
resultados/            métricas em JSON
tests/                 testes
```

## Tecnologias

Python · pandas · scikit-learn · LightGBM · FastAPI · Streamlit · Plotly · pytest

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "orders_daily.csv"

st.set_page_config(
    page_title="AI Demand Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Demand Forecast")
st.caption("Machine Learning aplicado à previsão de demanda")

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

df = carregar_dados()

st.sidebar.header("Filtros")

data_inicio = st.sidebar.date_input(
    "Data inicial", df["date"].min().date()
)

data_fim = st.sidebar.date_input(
    "Data final", df["date"].max().date()
)

filtrado = df[
    (df["date"].dt.date >= data_inicio)
    & (df["date"].dt.date <= data_fim)
]

if filtrado.empty:
    st.warning("Nenhum dado encontrado nesse período.")
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("Total de pedidos", f"{filtrado['orders'].sum():,}")
col2.metric("Média diária", f"{filtrado['orders'].mean():.1f}")
col3.metric("Dias analisados", len(filtrado))

st.subheader("Histórico de pedidos")

fig = px.line(
    filtrado,
    x="date",
    y="orders",
    title="Evolução diária dos pedidos"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Dados utilizados")
st.dataframe(filtrado, use_container_width=True)

st.info(
    "Os dados utilizados são sintéticos e destinados "
    "exclusivamente a fins educacionais."
)


# ==========================================
# PREVISÕES DO RANDOM FOREST
# ==========================================

PREDICTIONS_PATH = ROOT / "outputs" / "previsoes.csv"

st.divider()
st.subheader("🤖 Previsões do Random Forest")

if PREDICTIONS_PATH.exists():

    previsoes = pd.read_csv(PREDICTIONS_PATH)

    previsoes["data"] = pd.to_datetime(previsoes["data"])

    # Filtrar pelo período selecionado
    previsoes = previsoes[
        (previsoes["data"].dt.date >= data_inicio)
        & (previsoes["data"].dt.date <= data_fim)
    ]

    if not previsoes.empty:

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error
        )

        mae = mean_absolute_error(
            previsoes["pedidos"],
            previsoes["pedidos_previstos"]
        )

        rmse = mean_squared_error(
            previsoes["pedidos"],
            previsoes["pedidos_previstos"]
        ) ** 0.5

        col1, col2, col3 = st.columns(3)

        col1.metric("MAE", f"{mae:.2f} pedidos")
        col2.metric("RMSE", f"{rmse:.2f} pedidos")
        col3.metric("Dias avaliados", len(previsoes))

        # Gráfico de comparação
        fig_previsoes = px.line(
            previsoes,
            x="data",
            y=["pedidos", "pedidos_previstos"],
            title="Pedidos reais vs. previstos",
            labels={
                "data": "Data",
                "value": "Quantidade de pedidos",
                "variable": "Série"
            }
        )

        fig_previsoes.update_layout(
            legend_title_text="Comparação"
        )

        st.plotly_chart(
            fig_previsoes,
            use_container_width=True
        )

        # Tabela de previsões
        st.subheader("Tabela de previsões")

        st.dataframe(
            previsoes,
            use_container_width=True
        )

        st.caption(
            "Previsões históricas em dados sintéticos "
            "do conjunto de teste. Não são previsões futuras."
        )

    else:
        st.info(
            "Não existem previsões no período selecionado."
        )

else:
    st.warning(
        "Execute python src/train.py para gerar as previsões."
    )
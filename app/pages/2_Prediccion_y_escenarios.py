import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import get_dataset, predict, require_api
from expotb import config

st.set_page_config(page_title="Predicción | ExpoTB", layout="wide")
st.title("Predicción y escenarios")
require_api()

df = get_dataset()

# opción de precargar los indicadores reales de un país y año
with st.sidebar:
    st.header("Precargar país")
    country = st.selectbox("País", ["(ninguno)"] + sorted(df["country"].unique()))
    base = None
    if country != "(ninguno)":
        years = sorted(df.loc[df["country"] == country, "year"])
        year = st.selectbox("Año", years, index=len(years) - 1)
        base = df[(df["country"] == country) & (df["year"] == year)].iloc[0]

st.markdown("Ajusta los indicadores para simular un escenario.")
indicators = {}
cols = st.columns(3)
for i, feat in enumerate(config.FEATURES):
    lo, hi, default = config.FEATURE_RANGES[feat]
    value = default if base is None or pd.isna(base[feat]) else float(base[feat])
    indicators[feat] = cols[i % 3].slider(config.FEATURE_LABELS[feat], lo, hi, min(max(value, lo), hi))

st.session_state["indicators"] = indicators  # se reutiliza en la página de explicabilidad

preds = predict(indicators)
ens = preds.get("ensamble")
c1, c2 = st.columns(2)
if ens is not None:
    c1.metric("Incidencia estimada (ensamble)", f"{ens:,.1f} casos / 100 000 hab.")
if base is not None:
    c2.metric(f"Incidencia real ({country}, {year})", f"{base[config.TARGET]:,.1f}")

table = pd.DataFrame({
    "Modelo": [config.MODEL_LABELS[k] for k in preds],
    "Tipo": [config.MODEL_TYPE[k] for k in preds],
    "Incidencia estimada": list(preds.values()),
}).sort_values("Incidencia estimada")
fig = px.bar(table, x="Incidencia estimada", y="Modelo", color="Tipo", orientation="h",
             labels={"Incidencia estimada": "Casos por 100 000 hab."})
st.plotly_chart(fig, width="stretch")
st.dataframe(table.round(2), hide_index=True, width="stretch")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import get_dataset, predict, require_api
from expotb import config
from ui import DIM_COLORS, note, page_header, style_fig

page_header("Predicción y escenarios",
            "Ajusta los indicadores del exposoma y observa la incidencia estimada por cada modelo.",
            "query_stats")
require_api()

df = get_dataset()

# opción de precargar los indicadores reales de un país y año
with st.sidebar:
    st.markdown("#### Precargar un país")
    country = st.selectbox("País", ["(ninguno)"] + sorted(df["country"].unique()))
    base = None
    if country != "(ninguno)":
        years = sorted(df.loc[df["country"] == country, "year"])
        year = st.selectbox("Año", years, index=len(years) - 1)
        base = df[(df["country"] == country) & (df["year"] == year)].iloc[0]
        st.caption("Los controles parten de los valores reales de ese país y año.")

state = st.session_state
key = (country, None if base is None else int(year))
if state.get("_preload") != key or any(f"in_{feat}" not in state for feat in config.FEATURES):
    for feat in config.FEATURES:
        lo, hi, default = config.FEATURE_RANGES[feat]
        value = default if base is None or pd.isna(base[feat]) else float(base[feat])
        state[f"in_{feat}"] = min(max(value, lo), hi)
    state["_preload"] = key
    state.pop("reference", None)
    if base is not None:
        st.toast(f"Indicadores de {country} ({year}) cargados", icon=":material/check_circle:")

left, right = st.columns([3, 2], gap="medium")

with left:
    st.markdown("## Indicadores del escenario")
    indicators = {}
    for dim, feats in config.DIMENSIONS.items():
        st.markdown(f'<span style="color:{DIM_COLORS[dim]};font-weight:700">{dim}</span>',
                    unsafe_allow_html=True)
        cols = st.columns(len(feats))
        for col, feat in zip(cols, feats):
            lo, hi, _ = config.FEATURE_RANGES[feat]
            indicators[feat] = col.slider(config.FEATURE_LABELS[feat], lo, hi, key=f"in_{feat}")

state["indicators"] = indicators  # se reutiliza en la página de explicabilidad

with st.spinner("Calculando..."):
    preds = predict(indicators)
ens = preds.get("ensamble")

with right:
    st.markdown("## Incidencia estimada")
    real = None if base is None else float(base[config.TARGET])
    top = max(300.0, (ens or 0) * 1.4, (real or 0) * 1.2)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=ens or 0, number=dict(suffix="", valueformat=",.1f"),
        title=dict(text="casos por 100 000 hab. (ensamble)", font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, top]), bar=dict(color="#0F766E", thickness=0.35),
                   bgcolor="#E6F2F1", borderwidth=0,
                   threshold=dict(line=dict(color="#F97316", width=4), thickness=0.8, value=real)
                   if real is not None else None)))
    st.plotly_chart(style_fig(gauge, 260), width="stretch")
    if real is not None:
        diff = (ens or 0) - real
        st.metric(f"Incidencia real ({country}, {year})", f"{real:,.1f}",
                  delta=f"{diff:+,.1f} del modelo", delta_color="off")
        st.caption("La línea naranja del medidor marca la incidencia real.")

    ref = state.get("reference")
    b1, b2 = st.columns(2)
    if b1.button("Fijar como referencia", icon=":material/push_pin:", width="stretch"):
        state["reference"] = ens
        st.rerun()
    if ref is not None:
        if b2.button("Quitar", icon=":material/close:", width="stretch"):
            state.pop("reference")
            st.rerun()
        st.metric("Cambio frente a la referencia", f"{ens:,.1f}", delta=f"{ens - ref:+,.1f}",
                  delta_color="inverse")

st.markdown("## Estimación por modelo")
table = pd.DataFrame({
    "Modelo": [config.MODEL_LABELS[k] for k in preds],
    "Tipo": [config.MODEL_TYPE[k] for k in preds],
    "Incidencia estimada": list(preds.values()),
}).sort_values("Incidencia estimada")
fig = px.bar(table, x="Incidencia estimada", y="Modelo", color="Tipo", orientation="h",
             labels={"Incidencia estimada": "Casos por 100 000 hab."},
             color_discrete_map={"Caja blanca": "#0EA5E9", "Caja negra": "#0F766E", "Ensamble": "#F97316"})
if real is not None:
    fig.add_vline(x=real, line_dash="dash", line_color="#F97316", annotation_text="real")
st.plotly_chart(style_fig(fig, 340), width="stretch")
note("Los modelos pueden discrepar bastante entre sí: un árbol de decisión es muy sensible a pequeños cambios "
     "en los indicadores, mientras que el ensamble promedia a varios modelos y es más estable.")
with st.expander("Ver tabla"):
    st.dataframe(table.round(2), hide_index=True, width="stretch")

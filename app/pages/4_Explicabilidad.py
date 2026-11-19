import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import explain, get_global_shap, require_api
from expotb import config

st.set_page_config(page_title="Explicabilidad | ExpoTB", layout="wide")
st.title("Explicabilidad (SHAP)")
require_api()
st.caption(f"Valores SHAP calculados sobre {config.MODEL_LABELS[config.EXPLAINER_MODEL]}, "
           "en escala log(1 + incidencia).")


def to_frame(values: dict, labels: dict) -> pd.DataFrame:
    return pd.DataFrame({"Factor": [labels.get(k, k) for k in values], "SHAP": list(values.values())})


st.subheader("Importancia global")
g = get_global_shap()
c1, c2 = st.columns(2)
feat = to_frame(g["features"], config.FEATURE_LABELS).sort_values("SHAP")
c1.plotly_chart(px.bar(feat, x="SHAP", y="Factor", orientation="h", title="Por indicador (|SHAP| medio)"),
                width="stretch")
dim = to_frame(g["dimensions"], {}).sort_values("SHAP")
c2.plotly_chart(px.bar(dim, x="SHAP", y="Factor", orientation="h", title="Por dimensión del exposoma"),
                width="stretch")

st.subheader("Explicación del escenario actual")
indicators = st.session_state.get("indicators")
if indicators is None:
    st.info("Primero define un escenario en la página **Predicción y escenarios**.")
    st.stop()

local = explain(indicators)
loc = to_frame(local["features"], config.FEATURE_LABELS)
loc["Efecto"] = loc["SHAP"].map(lambda v: "Aumenta la incidencia" if v > 0 else "Reduce la incidencia")
fig = px.bar(loc.sort_values("SHAP"), x="SHAP", y="Factor", color="Efecto", orientation="h",
             color_discrete_map={"Aumenta la incidencia": "#d62728", "Reduce la incidencia": "#1f77b4"})
st.plotly_chart(fig, width="stretch")

dims = local["dimensions"]
cols = st.columns(len(dims))
for col, (name, value) in zip(cols, dims.items()):
    col.metric(name, f"{value:+.3f}")

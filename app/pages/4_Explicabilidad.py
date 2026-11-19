import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import explain, get_global_shap, require_api
from expotb import config
from ui import DIM_COLORS, FEATURE_DIM, kpi_row, note, page_header, style_fig

page_header("Explicabilidad (SHAP)",
            "Qué indicadores y qué dimensiones del exposoma empujan la estimación hacia arriba o hacia abajo.",
            "psychology")
require_api()
st.caption(f"Valores SHAP calculados sobre {config.MODEL_LABELS[config.EXPLAINER_MODEL]}, "
           "en escala log(1 + incidencia).")


def to_frame(values: dict, labels: dict) -> pd.DataFrame:
    return pd.DataFrame({"key": list(values), "Factor": [labels.get(k, k) for k in values],
                         "SHAP": list(values.values())})


tab_g, tab_l = st.tabs([":material/public: Importancia global", ":material/person_search: Escenario actual"])

with tab_g:
    note("Cuanto más larga la barra, más influye el factor en las estimaciones del modelo en promedio "
         "(valor absoluto medio de SHAP). No indica si el efecto es positivo o negativo.")
    g = get_global_shap()
    c1, c2 = st.columns(2)
    feat = to_frame(g["features"], config.FEATURE_LABELS)
    feat["Dimensión"] = feat["key"].map(FEATURE_DIM)
    fig = px.bar(feat.sort_values("SHAP"), x="SHAP", y="Factor", orientation="h", color="Dimensión",
                 color_discrete_map=DIM_COLORS, title="Por indicador")
    c1.plotly_chart(style_fig(fig, 380), width="stretch")
    dim = to_frame(g["dimensions"], {})
    fig = px.bar(dim.sort_values("SHAP"), x="SHAP", y="Factor", orientation="h", color="Factor",
                 color_discrete_map=DIM_COLORS, title="Por dimensión del exposoma")
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style_fig(fig, 380), width="stretch")

with tab_l:
    indicators = st.session_state.get("indicators")
    if indicators is None:
        note("Primero define un escenario en la página <b>Predicción y escenarios</b>.", warn=True)
        st.page_link("pages/2_Prediccion_y_escenarios.py", label="Ir a Predicción y escenarios",
                     icon=":material/query_stats:")
        st.stop()

    local = explain(indicators)
    loc = to_frame(local["features"], config.FEATURE_LABELS)
    loc["Efecto"] = loc["SHAP"].map(lambda v: "Aumenta la incidencia" if v > 0 else "Reduce la incidencia")
    fig = px.bar(loc.sort_values("SHAP"), x="SHAP", y="Factor", color="Efecto", orientation="h",
                 color_discrete_map={"Aumenta la incidencia": "#DC2626", "Reduce la incidencia": "#0EA5E9"})
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    st.markdown("## Contribución por dimensión")
    dims = local["dimensions"]
    kpi_row([
        (name, f"{value:+.3f}", "aumenta la incidencia" if value > 0 else "reduce la incidencia",
         DIM_COLORS[name])
        for name, value in dims.items()
    ])

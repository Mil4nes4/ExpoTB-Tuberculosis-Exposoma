import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import explain, get_global_shap, predict, predict_interval, require_api
from expotb import config
from ui import DIM_COLORS, FEATURE_DIM, kpi_row, note, page_header, style_fig
from ui import csv_download, module_route, quick_guide

page_header("Explicabilidad (SHAP)",
            "Qué indicadores y qué dimensiones del exposoma empujan la estimación hacia arriba o hacia abajo.",
            "psychology")
require_api()
module_route(4)
quick_guide(
    ("Consultar la importancia global para comparar indicadores y dimensiones.",
     "Abrir Escenario actual después de ajustar los indicadores en Predicción y escenarios."),
    (("SHAP", "Método que atribuye a cada indicador una parte de la diferencia entre la estimación y el valor base."),
     ("Valor absoluto medio", "Magnitud promedio de la contribución, sin distinguir su dirección.")),
)
EXPLAINED = config.MODEL_LABELS[config.EXPLAINER_MODEL]
st.caption(f"Valores SHAP calculados sobre {EXPLAINED}, en escala log(1 + incidencia).")


def to_frame(values: dict, labels: dict) -> pd.DataFrame:
    return pd.DataFrame({"key": list(values), "Factor": [labels.get(k, k) for k in values],
                         "SHAP": list(values.values())})


def pct_effect(value: float) -> str:
    """Traduce una contribución SHAP (escala log) a un cambio porcentual aproximado de la estimación."""
    change = float(np.expm1(value))
    return f"la sube cerca de un {change:.0%}" if value > 0 else f"la baja cerca de un {-change:.0%}"


def short(label: str) -> str:
    """Nombre del indicador sin unidad, para usarlo dentro de una frase."""
    text = label.split(" (")[0]
    return text[0].lower() + text[1:] if not text[:3].isupper() else text


tab_g, tab_l = st.tabs([":material/public: Importancia global", ":material/person_search: Escenario actual"])

with tab_g:
    g = get_global_shap()
    feat = to_frame(g["features"], config.FEATURE_LABELS)
    feat["Dimensión"] = feat["key"].map(FEATURE_DIM)
    ranked = feat.sort_values("SHAP", ascending=False).reset_index(drop=True)
    dim = to_frame(g["dimensions"], {}).sort_values("SHAP", ascending=False).reset_index(drop=True)
    first, second = ranked.iloc[0], ranked.iloc[1]
    note(f"<b>En pocas palabras:</b> el indicador que más pesa en las estimaciones es "
         f"<b>{short(first['Factor'])}</b> (dimensión {first['Dimensión']}), seguido de "
         f"{short(second['Factor'])} ({second['Dimensión']}). Por dimensión, la más influyente es "
         f"<b>{dim.iloc[0]['Factor']}</b> y la menos influyente, {dim.iloc[-1]['Factor']}. "
         "Cuanto más larga la barra, más influye el factor en promedio; la barra no indica si sube o baja la "
         "incidencia.")
    c1, c2 = st.columns(2)
    fig = px.bar(feat, x="SHAP", y="Factor", orientation="h", color="Dimensión",
                 color_discrete_map=DIM_COLORS, title="Por indicador",
                 labels={"SHAP": "Influencia media (|SHAP|, escala log)"})
    fig.update_yaxes(categoryorder="total ascending", title=None)
    c1.plotly_chart(style_fig(fig, 380), width="stretch")
    fig = px.bar(dim, x="SHAP", y="Factor", orientation="h", color="Factor",
                 color_discrete_map=DIM_COLORS, title="Por dimensión del exposoma",
                 labels={"SHAP": "Influencia media (|SHAP|, escala log)"})
    fig.update_yaxes(categoryorder="total ascending", title=None)
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style_fig(fig, 380), width="stretch")
    note("<b>Asociación, no causalidad.</b> El desarrollo económico, el acceso a combustibles limpios y la "
         "urbanización se mueven juntos, así que el peso de un indicador también recoge el de los que lo "
         "acompañan. SHAP explica lo que el modelo aprendió, no el efecto de una intervención.", warn=True)
    csv_download(ranked[["Factor", "Dimensión", "SHAP"]], "expotb_shap_global.csv",
                 "Descargar importancia global (CSV)")

with tab_l:
    indicators = st.session_state.get("indicators")
    if indicators is None:
        note("Primero define un escenario en la página <b>Predicción y escenarios</b>.", warn=True)
        st.page_link("pages/2_Prediccion_y_escenarios.py", label="Ir a Predicción y escenarios",
                     icon=":material/query_stats:")
        st.stop()

    with st.spinner("Calculando las contribuciones del escenario…"):
        local = explain(indicators)
        preds = predict(indicators)
        iv = predict_interval(indicators)
    label = st.session_state.get("scenario_label", "escenario actual")
    loc = to_frame(local["features"], config.FEATURE_LABELS)
    loc["Dimensión"] = loc["key"].map(FEATURE_DIM)
    loc["Efecto"] = loc["SHAP"].map(lambda v: "Aumenta la incidencia" if v > 0 else "Reduce la incidencia")
    loc["Cambio aproximado"] = loc["SHAP"].map(lambda v: f"{np.expm1(v):+.0%}")

    # resumen en lenguaje sencillo
    up = loc[loc["SHAP"] > 0].sort_values("SHAP", ascending=False)
    down = loc[loc["SHAP"] < 0].sort_values("SHAP")
    base_cases = float(np.expm1(local["base_value_log"]))
    model_cases = preds.get(config.EXPLAINER_MODEL)
    parts = [f"<b>Escenario:</b> {label}. "]
    if model_cases is not None:
        parts.append(f"{EXPLAINED} parte de un valor promedio de <b>{base_cases:,.0f}</b> casos por 100 000 hab. "
                     f"y, con estos indicadores, llega a <b>{model_cases:,.0f}</b>. ")
    if len(up):
        parts.append(f"El factor que más <b>aumenta</b> la estimación es <b>{short(up.iloc[0]['Factor'])}</b> "
                     f"({pct_effect(up.iloc[0]['SHAP'])}). ")
    if len(down):
        parts.append(f"El que más la <b>reduce</b> es <b>{short(down.iloc[0]['Factor'])}</b> "
                     f"({pct_effect(down.iloc[0]['SHAP'])}).")
    note("".join(parts))
    if iv:
        st.caption(f"Estimación del ensamble para este escenario: {iv['ensamble']:,.1f} casos por 100 000 hab. "
                   f"(rango plausible al {iv['nivel']:.0%}: {iv['inferior']:,.0f} a {iv['superior']:,.0f}). "
                   f"Las contribuciones se calculan sobre {EXPLAINED}, por eso su cifra puede diferir de la del "
                   "ensamble.")

    fig = px.bar(loc, x="SHAP", y="Factor", color="Efecto", orientation="h",
                 color_discrete_map={"Aumenta la incidencia": "#DC2626", "Reduce la incidencia": "#0EA5E9"},
                 hover_data={"Cambio aproximado": True, "Dimensión": True},
                 labels={"SHAP": "Contribución SHAP (escala log; hacia la derecha sube la estimación)"})
    fig.update_yaxes(categoryorder="total ascending", title=None)
    fig.add_vline(x=0, line_color="#4B6A69", line_width=1)
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    csv_download(loc[["Factor", "Dimensión", "SHAP", "Cambio aproximado", "Efecto"]],
                 "expotb_shap_escenario.csv", "Descargar contribuciones (CSV)")

    st.markdown("## Contribución por dimensión")
    dims = local["dimensions"]
    kpi_row([
        (name, f"{value:+.3f}", f"{'aumenta' if value > 0 else 'reduce'} la incidencia (≈ {np.expm1(value):+.0%})",
         DIM_COLORS[name])
        for name, value in dims.items()
    ])
    st.caption("El porcentaje es aproximado: los valores SHAP están en escala log(1 + incidencia), así que cada "
               "contribución multiplica la estimación en lugar de sumarle casos. Es una asociación, no un efecto "
               "causal.")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import get_dataset, predict, predict_interval, require_api
from expotb import config
from ui import DIM_COLORS, note, page_header, style_fig
from ui import csv_download, module_route, quick_guide

page_header("Predicción y escenarios",
            "Ajusta los indicadores del exposoma y observa la incidencia estimada por cada modelo.",
            "query_stats")
require_api()
module_route(2)
quick_guide(
    ("Precargar un país y un año o ajustar los seis indicadores.",
     "Fijar una referencia antes de cambiar los valores para comparar escenarios.",
     "Consultar la estimación por modelo y descargar la tabla."),
    (("Incidencia estimada", "Casos nuevos estimados durante un año por cada 100 000 habitantes."),
     ("Ensamble", "Modelo que combina las predicciones de varios modelos.")),
)

df = get_dataset()

# opción de precargar los indicadores reales de un país y año
with st.sidebar:
    st.markdown("#### Precargar un país")
    explored = st.session_state.get("filtros_explorador", {})
    if explored.get("country") in df["country"].values:
        def use_explored_country():
            st.session_state["prediccion_pais"] = explored["country"]
            available = df.loc[df["country"].eq(explored["country"]), "year"]
            st.session_state["prediccion_anio"] = (
                explored["year"] if explored["year"] in available.values else int(available.max())
            )

        st.button("Usar país del Explorador", icon=":material/travel_explore:", on_click=use_explored_country,
                  help="Carga el país y, si está disponible, el año consultado en el mapa.")
    country = st.selectbox("País", ["(ninguno)"] + sorted(df["country"].unique()), key="prediccion_pais",
                           help="Precarga los indicadores del país.",
                           on_change=st.session_state.pop, args=("prediccion_anio", None))
    base = None
    if country != "(ninguno)":
        years = sorted(df.loc[df["country"] == country, "year"])
        year = st.selectbox("Año", years, index=None if "prediccion_anio" in st.session_state else len(years) - 1,
                            key="prediccion_anio", help="Año de los indicadores que se cargarán.")
        base = df[(df["country"] == country) & (df["year"] == year)].iloc[0]
        st.caption("Los controles parten de los valores reales de ese país y año.")

state = st.session_state
state.setdefault("_base_values", {})
key = (country, None if base is None else int(year))
if state.get("_preload") != key or any(f"in_{feat}" not in state for feat in config.FEATURES):
    for feat in config.FEATURES:
        lo, hi, default = config.FEATURE_RANGES[feat]
        value = default if base is None or pd.isna(base[feat]) else float(base[feat])
        state[f"in_{feat}"] = min(max(value, lo), hi)
    state["_base_values"] = {feat: state[f"in_{feat}"] for feat in config.FEATURES}
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
            indicators[feat] = col.slider(config.FEATURE_LABELS[feat], lo, hi, key=f"in_{feat}",
                                          help=f"Indicador de la dimensión {dim}. Rango del control: {lo:g} a {hi:g}.")

state["indicators"] = indicators  # se reutiliza en la página de explicabilidad
if base is None:
    state["scenario_label"] = "escenario personalizado"
else:
    adjusted = any(abs(indicators[f] - state["_base_values"].get(f, indicators[f])) > 1e-9 for f in config.FEATURES)
    state["scenario_label"] = f"{country} ({year})" + (" con indicadores ajustados" if adjusted else "")

with st.spinner("Calculando..."):
    preds = predict(indicators)
ens = preds.get("ensamble")
iv = predict_interval(indicators)   # None si la API aún no tiene el intervalo calibrado

with right:
    st.markdown("## Incidencia estimada")
    real = None if base is None else float(base[config.TARGET])
    top = max(300.0, (ens or 0) * 1.4, (real or 0) * 1.2, (iv["superior"] * 1.1) if iv else 0)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=ens or 0, number=dict(suffix="", valueformat=",.1f"),
        title=dict(text="casos por 100 000 hab. (ensamble)", font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, top]), bar=dict(color="#0F766E", thickness=0.35),
                   bgcolor="#E6F2F1", borderwidth=0,
                   steps=[dict(range=[iv["inferior"], iv["superior"]], color="#99D5CF")] if iv else [],
                   threshold=dict(line=dict(color="#2563EB", width=4), thickness=0.8, value=real)
                   if real is not None else None)))
    st.plotly_chart(style_fig(gauge, 260), width="stretch")
    if iv:
        st.markdown(f"**Rango plausible ({iv['nivel']:.0%}):** {iv['inferior']:,.0f} a {iv['superior']:,.0f} "
                    "casos por 100 000 hab.")
        cover = iv.get("cobertura_por_pais")
        st.caption("La franja clara del medidor marca ese rango. Se calibró con los errores del modelo en países "
                   "que no vio durante el entrenamiento"
                   + (f" (en la prueba, el {cover:.0%} de los países reales cayó dentro)." if cover else ".")
                   + " Es una guía de la incertidumbre, no una garantía.")
    if real is not None:
        diff = (ens or 0) - real
        st.metric(f"Incidencia real ({country}, {year})", f"{real:,.1f}",
                  delta=f"{diff:+,.1f} (estimación menos real)", delta_color="off")
        st.caption("La línea azul del medidor marca la incidencia real. Este país-año pudo formar parte del "
                   "entrenamiento del modelo, así que la cercanía al valor real es optimista: para un país nuevo "
                   "guíate por el rango plausible.")

    ref = state.get("reference")
    if ref is not None and not isinstance(ref, dict):   # formato anterior (solo la cifra)
        ref = None
    b1, b2 = st.columns(2)
    if b1.button("Fijar como referencia", icon=":material/push_pin:", width="stretch",
                 help="Guarda este escenario para compararlo, lado a lado, con los ajustes que hagas después."):
        state["reference"] = {"ens": ens, "indicators": dict(indicators), "iv": iv,
                              "label": f"{country} ({year})" if base is not None else "escenario fijado"}
        st.toast("Referencia fijada. Ahora mueve un indicador.", icon=":material/push_pin:")
        st.rerun()
    if ref is not None:
        if b2.button("Quitar referencia", icon=":material/close:", width="stretch",
                     help="Quita la referencia guardada."):
            state.pop("reference")
            st.rerun()
    elif base is not None:
        moved = any(abs(indicators[f] - state["_base_values"].get(f, indicators[f])) > 1e-9
                    for f in config.FEATURES)
        if moved:
            def reset_country():
                for f, v in state["_base_values"].items():
                    state[f"in_{f}"] = v

            b2.button("Restablecer", icon=":material/restart_alt:", width="stretch", on_click=reset_country,
                      help=f"Vuelve a los valores reales de {country} ({year}).")
    st.page_link("pages/4_Explicabilidad.py", label="Ver por qué el modelo estima esto (SHAP)",
                 icon=":material/psychology:")

if ref is not None:
    st.markdown("## Referencia frente a escenario actual")
    cmp = pd.DataFrame([{"Indicador": config.FEATURE_LABELS[f], "Referencia": ref["indicators"][f],
                         "Escenario actual": indicators[f], "Cambio": indicators[f] - ref["indicators"][f]}
                        for f in config.FEATURES])
    changed = cmp.loc[cmp["Cambio"].abs() > 1e-9, "Indicador"].tolist()

    def fmt_range(r):
        return f"{r['inferior']:,.0f} a {r['superior']:,.0f}" if r else "no disponible"

    c1, c2 = st.columns(2)
    c1.metric(f"Referencia: {ref['label']}", f"{ref['ens']:,.1f}",
              help="Incidencia estimada por el ensamble, en casos por 100 000 hab.")
    c1.caption(f"Rango plausible: {fmt_range(ref.get('iv'))} casos por 100 000 hab.")
    c2.metric("Escenario actual", f"{ens:,.1f}", delta=f"{ens - ref['ens']:+,.1f} frente a la referencia",
              delta_color="inverse", help="Una flecha hacia abajo (verde) indica menos incidencia estimada.")
    c2.caption(f"Rango plausible: {fmt_range(iv)} casos por 100 000 hab.")
    st.dataframe(cmp, hide_index=True, width="stretch", column_config={
        "Referencia": st.column_config.NumberColumn(format="%.2f"),
        "Escenario actual": st.column_config.NumberColumn(format="%.2f"),
        "Cambio": st.column_config.NumberColumn(format="%+.2f")})
    if changed:
        st.caption("Indicadores modificados: " + ", ".join(changed) + ". El cambio en la estimación es una "
                   "asociación aprendida de los datos, no el efecto causal de una intervención.")
    else:
        st.caption("Todavía no hay cambios: mueve uno o más indicadores para comparar.")

st.markdown("## Estimación por modelo")
table = pd.DataFrame({
    "Modelo": [config.MODEL_LABELS[k] for k in preds],
    "Tipo": [config.MODEL_TYPE[k] for k in preds],
    "Incidencia estimada": list(preds.values()),
}).sort_values("Incidencia estimada")
fig = px.bar(table, x="Incidencia estimada", y="Modelo", color="Tipo", orientation="h",
             labels={"Incidencia estimada": "Casos por 100 000 hab."},
             color_discrete_map={"Caja blanca": "#0EA5E9", "Caja negra": "#0F766E", "Ensamble": "#2563EB"})
fig.update_yaxes(categoryorder="total ascending", title=None)
if iv:
    fig.add_vrect(x0=iv["inferior"], x1=iv["superior"], fillcolor="#99D5CF", opacity=0.35, line_width=0,
                  layer="below")
if real is not None:
    fig.add_vline(x=real, line_dash="dash", line_color="#2563EB", annotation_text="real")
st.plotly_chart(style_fig(fig, 340), width="stretch")
if iv:
    st.caption(f"La franja clara marca el rango plausible del ensamble ({iv['nivel']:.0%}).")
note("Los modelos pueden discrepar bastante entre sí: un árbol de decisión es muy sensible a pequeños cambios "
     "en los indicadores, mientras que el ensamble promedia a varios modelos y es más estable.")
with st.expander("Ver tabla", icon=":material/table:"):
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    csv_download(table, "expotb_estimaciones.csv")

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import require_api, require_metrics
from expotb import config
from ui import kpi_row, note, page_header, style_fig
from ui import csv_download, module_route, quick_guide

page_header("Comparación de modelos",
            "Métricas con tres protocolos de validación, cada uno responde a una pregunta distinta.",
            "leaderboard")
require_api()
module_route(3)
quick_guide(
    ("Elegir R², MAE o RMSE para comparar los modelos.",
     "Consultar el protocolo que se ajuste al uso previsto y descargar su tabla."),
    (("R²", "Mide cuánto mejora el modelo frente a predecir la media. Puede ser negativo."),
     ("MAE y RMSE", "Miden el error en las unidades de la incidencia. RMSE da más peso a los errores grandes."),
     ("Protocolo de validación", "Forma de separar los datos de entrenamiento y evaluación.")),
)

m = require_metrics()

PROTOCOLS = {
    "aleatorio": ("Aleatorio", "Referencia y comparación con otros estudios",
                  "Los registros se sortean al azar entre entrenamiento (80 %) y prueba (20 %). "
                  "Se aplica validación cruzada de cinco particiones al conjunto de entrenamiento. Un mismo país "
                  "puede aparecer en ambos lados, por eso tiende a ser optimista."),
    "por_pais": ("Por país", "¿Qué tan bien generaliza a países que no vio?",
                 "La partición separa países completos: ningún país está a la vez en entrenamiento y prueba, "
                 "y la validación cruzada usa GroupKFold por código ISO3."),
    "temporal": ("Temporal", "¿Qué tan bien anticipa años futuros?",
                 "Se entrena con 2000–2015 y se evalúa con 2016–2019. Es una sola partición de cuatro años."),
}


def protocol_table(proto: str) -> pd.DataFrame:
    rows = []
    for name, r in m["modelos"].items():
        test = r["por_pais"]["prueba"] if proto == "por_pais" else r["prueba"] if proto == "aleatorio" else r["temporal"]
        cv = r["por_pais"]["cv"] if proto == "por_pais" else r["cv"] if proto == "aleatorio" else None
        row = {"Modelo": config.MODEL_LABELS[name], "Tipo": config.MODEL_TYPE[name],
               "R²": test["r2"], "MAE": test["mae"], "RMSE": test["rmse"]}
        if "rmsle" in test:
            row["R² (log)"] = test["r2_log"]
            row["RMSLE"] = test["rmsle"]
        if cv:
            row["R² validación cruzada"] = cv["r2"]["media"]
            row["Desviación estándar de R² (VC)"] = cv["r2"]["std"]
            row["MAE validación cruzada"] = cv["mae"]["media"]
            row["RMSE validación cruzada"] = cv["rmse"]["media"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


tables = {p: protocol_table(p) for p in PROTOCOLS}
opt = m.get("optimizacion", {})
tuned = opt.get("activa", False)
st.caption(f"{m['n_registros']:,} registros país-año de {m['n_paises']} países. ".replace(",", " ") +
           (f"Hiperparámetros optimizados con búsqueda aleatoria ({opt.get('n_iteraciones')} combinaciones por "
            "modelo" + (f"; {' y '.join(config.MODEL_LABELS[k] for k in opt['reajuste']['modelos'])} se reajustaron "
                        f"con hasta {opt['reajuste']['n_iteraciones']}" if opt.get("reajuste") else "") +
            ") dentro de la validación cruzada de cada protocolo, solo con datos de entrenamiento. "
            if tuned else "Hiperparámetros por defecto. ") + "random_state = 42.")
METRIC_OPTIONS = ["R²", "MAE", "RMSE"] + (["R² (log)", "RMSLE"] if "RMSLE" in tables["aleatorio"] else [])
LOWER_IS_BETTER = {"MAE", "RMSE", "RMSLE"}

kpi_row([(f"Mejor R² · {PROTOCOLS[p][0]}", f"{t.iloc[0]['R²']:.3f}", t.iloc[0]["Modelo"])
         for p, t in tables.items()])

tab_all, *tab_p = st.tabs([":material/compare_arrows: Visión general"] +
                          [f":material/science: {v[0]}" for v in PROTOCOLS.values()] +
                          [":material/rule: Líneas base y errores"])
tab_extra = tab_p.pop()

with tab_all:
    metric = st.radio("Métrica", METRIC_OPTIONS, horizontal=True, key="metric_general",
                      help="Un R² más alto indica mejor ajuste. Para MAE, RMSE y RMSLE se prefieren valores menores. "
                           "Las métricas (log) miden el error relativo y no las domina un grupo pequeño de países "
                           "con incidencias muy altas.")
    evaluations = {label: tables[p].set_index("Modelo")[metric]
                   for p, (label, _, _) in PROTOCOLS.items()}
    if f"{metric} validación cruzada" in tables["aleatorio"]:
        evaluations["Validación cruzada"] = tables["aleatorio"].set_index("Modelo")[f"{metric} validación cruzada"]
    overview = pd.DataFrame(evaluations).rename_axis("Modelo").reset_index()
    long = overview.melt(id_vars="Modelo", var_name="Evaluación", value_name=metric)
    order = tables["aleatorio"]["Modelo"].tolist()
    fig = px.bar(long, x="Modelo", y=metric, color="Evaluación", barmode="group",
                 category_orders={"Modelo": order},
                 color_discrete_map={"Aleatorio": "#0F766E", "Por país": "#2563EB",
                                     "Temporal": "#0EA5E9", "Validación cruzada": "#64748B"})
    st.plotly_chart(style_fig(fig, 430), width="stretch")
    st.dataframe(overview.round(3), hide_index=True, width="stretch")
    csv_download(overview, "expotb_comparacion.csv", "Descargar comparación (CSV)")
    note("<b>El R² aleatorio es optimista</b>: el modelo reconoce países que ya vio durante el entrenamiento. "
         "Al separar países completos, el desempeño baja de forma marcada. Esa cifra es la más realista para "
         "estimar la incidencia en un país nuevo.", warn=True)

for tab, (proto, (label, question, desc)) in zip(tab_p, PROTOCOLS.items()):
    with tab:
        st.markdown(f"**{question}**")
        note(desc)
        t = tables[proto]
        metric = st.radio("Métrica", METRIC_OPTIONS, horizontal=True, key=f"metric_{proto}",
                          help="Cambia la métrica representada en el gráfico de este protocolo.")
        asc = metric in LOWER_IS_BETTER
        data = t.sort_values(metric, ascending=asc)
        fig = px.bar(data, x=metric, y="Modelo", color="Tipo", orientation="h",
                     color_discrete_map={"Caja blanca": "#0EA5E9", "Caja negra": "#0F766E",
                                         "Ensamble": "#2563EB"})
        fig.update_yaxes(categoryorder="array", categoryarray=data["Modelo"].tolist()[::-1])
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.dataframe(t.round(3), hide_index=True, width="stretch", column_config={
            "R²": st.column_config.ProgressColumn("R²", min_value=0.0, max_value=1.0, format="%.3f"),
        })
        csv_download(t, f"expotb_validacion_{proto}.csv")


with tab_extra:
    st.markdown("### ¿Cuánto aporta el exposoma frente a no usarlo?")
    note("Las líneas base predicen sin mirar los seis indicadores: la <b>mediana global</b> responde siempre lo mismo "
         "y la <b>mediana por grupo de ingreso</b> solo conoce si el país es de ingreso bajo, medio o alto. "
         "Un modelo útil debe superarlas con claridad, sobre todo en el protocolo por país.")
    base = m.get("linea_base")
    if not base or "RMSLE" not in tables["aleatorio"]:
        st.info("Vuelve a entrenar (`python -m expotb.train`) para calcular las líneas base.", icon=":material/info:")
    else:
        rows = []
        for proto, (label, _, _) in PROTOCOLS.items():
            for key, name in (("mediana_global", "Mediana global"),
                              ("mediana_por_ingreso", "Mediana por grupo de ingreso")):
                r = base[key][proto]
                rows.append({"Protocolo": label, "Referencia": name, "Modelo": "—", "R²": r["r2"], "R² (log)": r["r2_log"],
                             "MAE": r["mae"], "RMSLE": r["rmsle"]})
            best = tables[proto].sort_values("R² (log)", ascending=False).iloc[0]
            rows.append({"Protocolo": label, "Referencia": "Mejor modelo ExpoTB",
                         "Modelo": best["Modelo"], "R²": best["R²"], "R² (log)": best["R² (log)"],
                         "MAE": best["MAE"], "RMSLE": best["RMSLE"]})
        cmp = pd.DataFrame(rows)
        fig = px.bar(cmp, x="Protocolo", y="R² (log)", color="Referencia", barmode="group",
                     color_discrete_sequence=["#94A3B8", "#64748B", "#0F766E"])
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        st.dataframe(cmp.round(3), hide_index=True, width="stretch")
        csv_download(cmp, "expotb_lineas_base.csv", "Descargar líneas base (CSV)")

    st.markdown("### ¿Dónde se equivoca más el modelo?")
    note("Se evalúa en países que el modelo no vio. El <b>sesgo</b> es la estimación menos el valor real: "
         "un valor negativo indica que el modelo subestima.")
    model_ids = [k for k in config.MODEL_LABELS if k in m["modelos"]]
    chosen = st.selectbox("Modelo", model_ids, index=model_ids.index("ensamble") if "ensamble" in model_ids else 0,
                          format_func=lambda k: config.MODEL_LABELS[k], key="modelo_errores")
    strat = m["modelos"][chosen]["por_pais"].get("estratificado")
    if not strat:
        st.info("Vuelve a entrenar para obtener los errores por grupo.", icon=":material/info:")
    else:
        c1, c2 = st.columns(2)
        inc = pd.DataFrame(strat["por_incidencia"]).T.rename_axis("Incidencia real").reset_index()
        inc = inc.rename(columns={"n": "Registros", "mae": "MAE", "sesgo": "Sesgo"})
        fig = px.bar(inc, x="Incidencia real", y="MAE", text=inc["Sesgo"].map(lambda v: f"sesgo {v:+.0f}"),
                     title="Error absoluto medio por nivel de incidencia (por 100 000 hab.)")
        fig.update_traces(marker_color="#0F766E", textposition="outside")
        c1.plotly_chart(style_fig(fig, 340), width="stretch")
        ing = pd.DataFrame(strat["por_ingreso"]).T.rename_axis("Grupo de ingreso").reset_index()
        ing = ing.rename(columns={"n": "Registros", "mae": "MAE", "rmsle": "RMSLE"})
        fig = px.bar(ing, x="Grupo de ingreso", y="RMSLE", title="Error relativo (RMSLE) por grupo de ingreso",
                     category_orders={"Grupo de ingreso": ["LIC", "LMC", "UMC", "HIC"]})
        fig.update_traces(marker_color="#2563EB")
        c2.plotly_chart(style_fig(fig, 340), width="stretch")
        st.dataframe(inc.round(2), hide_index=True, width="stretch")
        st.dataframe(ing.round(3), hide_index=True, width="stretch")
        note("LIC: ingreso bajo · LMC: ingreso medio bajo · UMC: ingreso medio alto · HIC: ingreso alto.")

    if tuned:
        with st.expander("Hiperparámetros elegidos por protocolo", icon=":material/tune:"):
            table_hp = pd.DataFrame([
                {"Protocolo": PROTOCOLS[proto][0], "Modelo": config.MODEL_LABELS[name],
                 "Hiperparámetros": ", ".join(f"{k}={v}" for k, v in values.items())}
                for proto, models in opt["hiperparametros"].items() for name, values in models.items()])
            st.dataframe(table_hp, hide_index=True, width="stretch")
            st.caption("El ensamble reutiliza los hiperparámetros de Random Forest, XGBoost y SVR. Los modelos que "
                       "sirve la API se entrenan con el 80 % aleatorio y los hiperparámetros del protocolo por país.")

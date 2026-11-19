import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import get_metrics, require_api
from expotb import config
from ui import kpi_row, note, page_header, style_fig

page_header("Comparación de modelos",
            "Métricas con tres protocolos de validación, cada uno responde a una pregunta distinta.",
            "leaderboard")
require_api()

m = get_metrics()

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
        if cv:
            row["R² validación cruzada"] = cv["r2"]["media"]
            row["Desviación estándar de R² (VC)"] = cv["r2"]["std"]
            row["MAE validación cruzada"] = cv["mae"]["media"]
            row["RMSE validación cruzada"] = cv["rmse"]["media"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


tables = {p: protocol_table(p) for p in PROTOCOLS}
st.caption(f"{m['n_registros']:,} registros país-año de {m['n_paises']} países. Hiperparámetros por defecto y "
           "random_state = 42.".replace(",", " "))

kpi_row([(f"Mejor R² · {PROTOCOLS[p][0]}", f"{t.iloc[0]['R²']:.3f}", t.iloc[0]["Modelo"])
         for p, t in tables.items()])

tab_all, *tab_p = st.tabs([":material/compare_arrows: Visión general"] +
                          [f":material/science: {v[0]}" for v in PROTOCOLS.values()])

with tab_all:
    metric = st.radio("Métrica", ["R²", "MAE", "RMSE"], horizontal=True, key="metric_general")
    evaluations = {label: tables[p].set_index("Modelo")[metric]
                   for p, (label, _, _) in PROTOCOLS.items()}
    evaluations["Validación cruzada"] = tables["aleatorio"].set_index("Modelo")[f"{metric} validación cruzada"]
    overview = pd.DataFrame(evaluations).rename_axis("Modelo").reset_index()
    long = overview.melt(id_vars="Modelo", var_name="Evaluación", value_name=metric)
    order = tables["aleatorio"]["Modelo"].tolist()
    fig = px.bar(long, x="Modelo", y=metric, color="Evaluación", barmode="group",
                 category_orders={"Modelo": order},
                 color_discrete_map={"Aleatorio": "#0F766E", "Por país": "#F97316",
                                     "Temporal": "#0EA5E9", "Validación cruzada": "#64748B"})
    st.plotly_chart(style_fig(fig, 430), width="stretch")
    st.dataframe(overview.round(3), hide_index=True, width="stretch")
    note("<b>El R² aleatorio es optimista</b>: el modelo reconoce países que ya vio durante el entrenamiento. "
         "Al separar países completos, el desempeño baja de forma marcada. Esa cifra es la más realista para "
         "estimar la incidencia en un país nuevo.", warn=True)

for tab, (proto, (label, question, desc)) in zip(tab_p, PROTOCOLS.items()):
    with tab:
        st.markdown(f"**{question}**")
        note(desc)
        t = tables[proto]
        metric = st.radio("Métrica", ["R²", "MAE", "RMSE"], horizontal=True, key=f"metric_{proto}")
        asc = metric != "R²"
        data = t.sort_values(metric, ascending=asc)
        fig = px.bar(data, x=metric, y="Modelo", color="Tipo", orientation="h",
                     color_discrete_map={"Caja blanca": "#0EA5E9", "Caja negra": "#0F766E",
                                         "Ensamble": "#F97316"})
        fig.update_yaxes(categoryorder="array", categoryarray=data["Modelo"].tolist()[::-1])
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.dataframe(t.round(3), hide_index=True, width="stretch", column_config={
            "R²": st.column_config.ProgressColumn("R²", min_value=0.0, max_value=1.0, format="%.3f"),
        })

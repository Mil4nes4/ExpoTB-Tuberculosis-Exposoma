import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import get_metrics, require_api
from expotb import config

st.set_page_config(page_title="Comparación | ExpoTB", layout="wide")
st.title("Comparación de modelos")
require_api()

m = get_metrics()
st.caption(f"{m['n_registros']} registros país-año de {m['n_paises']} países. "
           "Validación cruzada K=5 sobre el 80 %, prueba sobre el 20 % y prueba temporal (2000-2015 → 2016-2019).")

rows = []
for name, r in m["modelos"].items():
    rows.append({
        "Modelo": config.MODEL_LABELS[name], "Tipo": config.MODEL_TYPE[name],
        "MAE CV": r["cv"]["mae"]["media"], "RMSE CV": r["cv"]["rmse"]["media"],
        "R² CV": r["cv"]["r2"]["media"], "R² CV (std)": r["cv"]["r2"]["std"],
        "MAE prueba": r["prueba"]["mae"], "RMSE prueba": r["prueba"]["rmse"], "R² prueba": r["prueba"]["r2"],
        "MAE temporal": r["temporal"]["mae"], "RMSE temporal": r["temporal"]["rmse"],
        "R² temporal": r["temporal"]["r2"],
    })
table = pd.DataFrame(rows).sort_values("R² prueba", ascending=False)

best = table.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Mejor modelo (R² prueba)", best["Modelo"])
c2.metric("R² prueba", f"{best['R² prueba']:.3f}")
c3.metric("RMSE prueba", f"{best['RMSE prueba']:.1f}")

metric = st.radio("Métrica", ["R²", "MAE", "RMSE"], horizontal=True)
long = table.melt(id_vars=["Modelo"], value_vars=[f"{metric} CV", f"{metric} prueba", f"{metric} temporal"],
                  var_name="Evaluación", value_name=metric)
fig = px.bar(long, x="Modelo", y=metric, color="Evaluación", barmode="group")
st.plotly_chart(fig, width="stretch")

st.dataframe(table.round(3), hide_index=True, width="stretch")

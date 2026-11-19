import plotly.express as px
import streamlit as st

from api_client import get_dataset, require_api
from expotb import config

st.set_page_config(page_title="Explorador | ExpoTB", layout="wide")
st.title("Explorador del exposoma")
require_api()

df = get_dataset()
variables = {config.TARGET: "Incidencia de TB (por 100 000 hab.)", **config.FEATURE_LABELS}

col1, col2 = st.columns([2, 1])
var = col1.selectbox("Variable", list(variables), format_func=variables.get)
year = col2.slider("Año", int(df["year"].min()), int(df["year"].max()), int(df["year"].max()))

# mapa coroplético por país para el año elegido
fig = px.choropleth(df[df["year"] == year], locations="iso3", color=var, hover_name="country",
                    color_continuous_scale="Reds", labels={var: variables[var]})
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450)
st.plotly_chart(fig, width="stretch")

st.subheader("Evolución por país")
countries = sorted(df["country"].unique())
default = [c for c in ("Peru", "India", "South Africa") if c in countries]
selected = st.multiselect("Países", countries, default=default)
if selected:
    line = px.line(df[df["country"].isin(selected)], x="year", y=var, color="country",
                   markers=True, labels={var: variables[var], "year": "Año"})
    st.plotly_chart(line, width="stretch")

with st.expander("Ver datos"):
    st.dataframe(df, width="stretch")

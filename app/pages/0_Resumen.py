import streamlit as st

from api_client import api_ok, get_metrics
from expotb import config
from ui import card_grid, dim_card, hero_banner, kpi_row, note

hero_banner(
    eyebrow="Exposoma y tuberculosis",
    title="Expo<em>TB</em>",
    headline="De los factores del entorno a la incidencia de tuberculosis",
    lead="Se estima la incidencia anual de tuberculosis por país (casos nuevos por 100 000 habitantes) "
         "a partir de seis indicadores del exposoma y se explica su contribución a la estimación.",
    steps=[("travel_explore", "Explorar"), ("query_stats", "Estimar"), ("psychology", "Interpretar")],
)

if api_ok():
    m = get_metrics()
    kpi_row([
        ("Registros país-año", f"{m['n_registros']:,}".replace(",", " "), "2000–2019"),
        ("Países", f"{m['n_paises']}", "con datos completos suficientes"),
        ("Indicadores", f"{len(config.FEATURES)}", "en cuatro dimensiones"),
        ("Modelos", f"{len(m['modelos'])}", "caja blanca, negra y ensamble"),
    ])
else:
    st.info("Los indicadores del resumen se muestran cuando la API está disponible.",
            icon=":material/info:")

st.markdown("## Las cuatro dimensiones del exposoma")
dims = {
    "Externa": "PM2.5",
    "Hogar": "Combustibles limpios para cocinar",
    "Social": "Población urbana y PIB per cápita",
    "Biológica": "Prevalencia de VIH y de desnutrición",
}
card_grid(dim_card(name, indicators) for name, indicators in dims.items())

st.markdown("## Cómo usar la herramienta")
left, right = st.columns(2)
with left:
    st.page_link("pages/1_Explorador_del_exposoma.py", label="**Explorador del exposoma**: mapa mundial, "
                 "ranking y evolución por país", icon=":material/public:")
    st.page_link("pages/2_Prediccion_y_escenarios.py", label="**Predicción y escenarios**: simula "
                 "indicadores y estima la incidencia", icon=":material/query_stats:")
with right:
    st.page_link("pages/3_Comparacion_de_modelos.py", label="**Comparación de modelos**: métricas con tres "
                 "protocolos de validación", icon=":material/leaderboard:")
    st.page_link("pages/4_Explicabilidad.py", label="**Explicabilidad**: contribución de cada indicador "
                 "y dimensión (SHAP)", icon=":material/psychology:")

note("Los resultados orientan la priorización de intervenciones, pero no sustituyen la vigilancia "
     "epidemiológica ni constituyen un diagnóstico. La incidencia estimada depende de la calidad de los datos "
     "de la OMS y del Banco Mundial.", warn=True)

import streamlit as st

from api_client import api_ok, ensemble_r2, get_metrics
from expotb import config
from ui import card_grid, dim_card, hero_banner, kpi_row, note
from ui import quick_guide

hero_banner(
    eyebrow="Exposoma y tuberculosis",
    title="Expo<em>TB</em>",
    headline="Aprendizaje automático para estimar la incidencia de tuberculosis",
    lead="Se estima la incidencia anual de tuberculosis por país (casos nuevos por 100 000 habitantes) "
         "a partir de seis indicadores del exposoma y se explica su contribución a la estimación.",
    steps=[("travel_explore", "Explorar"), ("query_stats", "Estimar"), ("psychology", "Interpretar")],
)

m = get_metrics() if api_ok() else None
if m:
    kpi_row([
        ("Registros país-año", f"{m['n_registros']:,}".replace(",", " "), "2000–2019"),
        ("Países", f"{m['n_paises']}", "con datos completos suficientes"),
        ("Indicadores", f"{len(config.FEATURES)}", "en cuatro dimensiones"),
        ("Modelos", f"{len(m['modelos'])}", "caja blanca, negra y ensamble"),
    ])
else:
    st.info("Los indicadores del resumen se muestran cuando la API y las métricas están disponibles.",
            icon=":material/info:")

# desempeño realista: el R² aleatorio se muestra siempre junto al de países no vistos
r2_seen, r2_new = ensemble_r2(m)
if r2_seen is not None:
    st.markdown("## ¿Qué tan confiable es la estimación?")
    ens_new = m["modelos"]["ensamble"]["por_pais"]["prueba"]
    cards = [
        ("R² con países ya vistos", f"{r2_seen:.2f}",
         "partición aleatoria: optimista", "#94A3B8"),
        ("R² en países nuevos", f"{r2_new:.2f}",
         "países que el modelo no vio: cifra realista", "#2563EB"),
        ("Error medio en países nuevos", f"{ens_new['mae']:,.0f}",
         "casos por 100 000 hab. (MAE)", "#2563EB"),
    ]
    iv = m.get("intervalo")
    if iv and iv.get("cobertura_por_pais") is not None:
        cards.append((f"Rango plausible al {iv['nivel']:.0%}", f"{iv['cobertura_por_pais']:.0%}",
                      "de los países no vistos cayó dentro", "#0F766E"))
    kpi_row(cards)
    note("Valores del ensamble. Si se mezclan países al azar, el modelo reconoce países que ya vio y el R² parece "
         "muy alto; con países que nunca vio, el R² baja de forma marcada. Por eso ExpoTB sirve para "
         "<b>orientar y comparar escenarios</b>, siempre con su rango de incertidumbre, y no para dar una cifra "
         "exacta de un país.", warn=True)

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

quick_guide(
    ("Consultar el mapa y comparar países en el Explorador.",
     "Ajustar los indicadores en Predicción y escenarios.",
     "Revisar el desempeño de los modelos y consultar la contribución de cada factor."),
    (("Exposoma", "Conjunto de exposiciones a lo largo de la vida. Aquí se aproxima con seis indicadores."),
     ("Incidencia", "Casos nuevos estimados durante un año por cada 100 000 habitantes.")),
    title="Guía rápida de ExpoTB",
)

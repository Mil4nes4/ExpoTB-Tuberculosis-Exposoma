"""Interfaz de ExpoTB: punto de entrada y navegación.  Ejecutar:  streamlit run app/Inicio.py"""
import streamlit as st

from api_client import API_URL, api_ok, get_metrics
from chatbot import render_chat
from ui import inject_css, sidebar_summary, site_footer

st.set_page_config(page_title="ExpoTB", page_icon=":material/pulmonology:", layout="wide",
                   initial_sidebar_state="auto")
inject_css()

pages = {
    "Visión general": [
        st.Page("pages/0_Resumen.py", title="Inicio", icon=":material/home:", default=True),
    ],
    "Análisis": [
        st.Page("pages/1_Explorador_del_exposoma.py", title="Explorador del exposoma",
                icon=":material/public:"),
        st.Page("pages/2_Prediccion_y_escenarios.py", title="Predicción y escenarios",
                icon=":material/query_stats:"),
    ],
    "Modelos": [
        st.Page("pages/3_Comparacion_de_modelos.py", title="Comparación de modelos",
                icon=":material/leaderboard:"),
        st.Page("pages/4_Explicabilidad.py", title="Explicabilidad (SHAP)", icon=":material/psychology:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown('<div class="xbrand"><div><b>ExpoTB</b><span>Exposoma y tuberculosis</span></div></div>',
                unsafe_allow_html=True)
    if api_ok():
        st.success(f"API conectada en {API_URL}", icon=":material/check_circle:")
    else:
        st.error(f"API no disponible en {API_URL}", icon=":material/error:")
    m = get_metrics() if api_ok() else None
    if m:
        sidebar_summary([(f"{m['n_registros']:,}".replace(",", " "), "registros"),
                         (str(m["n_paises"]), "países"), (str(len(m["modelos"])), "modelos")])
    tuned = bool(m and m.get("optimizacion", {}).get("activa"))
    with st.expander("Fuentes de datos", icon=":material/database:"):
        st.markdown("Incidencia de tuberculosis: Organización Mundial de la Salud, vía Our World in Data. "
                    "Indicadores del exposoma y grupo de ingreso: Banco Mundial. Periodo 2000–2019.")
    with st.expander("Cómo se estima", icon=":material/account_tree:"):
        st.markdown(
            '<ol class="xsteplist"><li>Se integran las dos fuentes por país y año.</li>'
            "<li>Se imputan los faltantes con la mediana de su grupo de ingreso.</li>"
            "<li>Se entrenan seis modelos y un ensamble"
            + (", con hiperparámetros elegidos por búsqueda aleatoria" if tuned else "") + ".</li>"
            "<li>Se evalúan con tres protocolos: aleatorio, por país y temporal.</li>"
            "<li>Se explica cada estimación con SHAP.</li></ol>", unsafe_allow_html=True)
    with st.expander("Para desarrolladores", icon=":material/api:"):
        st.markdown(f"Documentación interactiva de la API: [{API_URL}/docs]({API_URL}/docs)")
    st.caption("Prototipo académico del curso de Inteligencia Artificial, FISI-UNMSM. "
               "No es una herramienta clínica.")

# asistente disponible en todas las páginas (botón flotante en una esquina)
render_chat()

nav.run()

site_footer()

"""Interfaz de ExpoTB: punto de entrada y navegación.  Ejecutar:  streamlit run app/Inicio.py"""
import streamlit as st

from api_client import API_URL, api_ok
from chatbot import render_chat
from ui import inject_css

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
    st.caption("Prototipo académico del curso de Inteligencia Artificial, FISI-UNMSM. "
               "No es una herramienta clínica.")

# asistente disponible en todas las páginas (botón flotante en una esquina)
render_chat()

nav.run()

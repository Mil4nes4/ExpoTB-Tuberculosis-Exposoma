"""Página de inicio de la interfaz ExpoTB.  Ejecutar:  streamlit run app/Inicio.py"""
import streamlit as st

from api_client import API_URL, api_ok

st.set_page_config(page_title="ExpoTB", page_icon="🫁", layout="wide")

st.title("ExpoTB")
st.subheader("Incidencia de tuberculosis estimada a partir de factores del exposoma")

st.markdown("""
ExpoTB estima la **tasa de incidencia anual de tuberculosis** (casos nuevos por 100 000 habitantes)
de un país a partir de seis indicadores del exposoma organizados en cuatro dimensiones:

| Dimensión | Indicadores |
|---|---|
| Externa | PM2.5 |
| Hogar | Acceso a combustibles limpios para cocinar |
| Social | Población urbana, PIB per cápita |
| Biológica | Prevalencia de VIH, prevalencia de desnutrición |

**Módulos** (menú lateral):
1. **Explorador del exposoma**: mapa mundial y evolución de los indicadores por país.
2. **Predicción y escenarios**: estima la incidencia con cada modelo a partir de los indicadores.
3. **Comparación de modelos**: métricas de validación cruzada, prueba y prueba temporal.
4. **Explicabilidad**: contribución SHAP de cada indicador y dimensión.
""")

if api_ok():
    st.success(f"API conectada en {API_URL}")
else:
    st.warning(f"API no disponible en {API_URL}. Revisa el README para iniciarla.")

st.caption("Prototipo académico — Curso de Inteligencia Artificial, FISI-UNMSM. No es una herramienta clínica.")

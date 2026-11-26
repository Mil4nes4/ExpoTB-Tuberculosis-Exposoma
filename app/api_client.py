"""Cliente HTTP que conecta la interfaz Streamlit con la API FastAPI."""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("EXPOTB_API_URL", "http://127.0.0.1:8000")


def _get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=5, show_spinner=False)
def api_ok() -> bool:
    try:
        return bool(_get("/health")["modelos_cargados"])
    except requests.RequestException:
        return False


def require_api():
    if not api_ok():
        st.error(f"No se pudo conectar con la API en {API_URL} o no hay modelos entrenados. "
                 "Inicia la API con `uvicorn api.main:app` y entrena con `python -m expotb.train`.")
        st.stop()


@st.cache_data(ttl=600)
def get_dataset() -> pd.DataFrame:
    return pd.DataFrame(_get("/dataset"))


@st.cache_data(ttl=600, show_spinner=False)
def _metrics() -> dict:
    return _get("/metrics")


def get_metrics() -> dict | None:
    """Métricas del entrenamiento; None si aún no existen (por ejemplo, durante un entrenamiento).

    Los errores no se guardan en caché, así que la página se recupera sola cuando el archivo vuelve a estar listo.
    """
    try:
        return _metrics()
    except requests.RequestException:
        return None


def require_metrics() -> dict:
    m = get_metrics()
    if m is None:
        st.info("Las métricas todavía no están disponibles (el entrenamiento puede estar en curso). "
                "Vuelve a cargar la página en unos minutos.", icon=":material/hourglass_top:")
        st.stop()
    return m


@st.cache_data(ttl=600, show_spinner=False)
def get_global_shap() -> dict:
    return _get("/explain/global")


# Las consultas se guardan en caché por escenario: al volver a una página o mover otro control no se repite la
# llamada a la API con los mismos indicadores.
@st.cache_data(ttl=300, show_spinner=False, max_entries=500)
def predict(indicators: dict) -> dict:
    return _post("/predict", indicators)


@st.cache_data(ttl=300, show_spinner=False, max_entries=500)
def explain(indicators: dict) -> dict:
    return _post("/explain", indicators)


@st.cache_data(ttl=300, show_spinner=False, max_entries=500)
def _interval(indicators: dict) -> dict:
    return _post("/predict/intervalo", indicators)


def predict_interval(indicators: dict) -> dict | None:
    """Estimación del ensamble con intervalo; devuelve None si la API aún no tiene el intervalo calibrado."""
    try:
        return _interval(indicators)
    except requests.RequestException:
        return None


def ensemble_r2(metrics: dict | None) -> tuple:
    """(R² aleatorio, R² por país) de prueba del ensamble, para comunicar el desempeño realista."""
    ens = (metrics or {}).get("modelos", {}).get("ensamble")
    if not ens:
        return None, None
    return ens["prueba"]["r2"], ens["por_pais"]["prueba"]["r2"]

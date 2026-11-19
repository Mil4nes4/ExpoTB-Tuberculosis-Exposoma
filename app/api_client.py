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


@st.cache_data(ttl=600)
def get_metrics() -> dict:
    return _get("/metrics")


@st.cache_data(ttl=600)
def get_global_shap() -> dict:
    return _get("/explain/global")


def predict(indicators: dict) -> dict:
    return _post("/predict", indicators)


def explain(indicators: dict) -> dict:
    return _post("/explain", indicators)

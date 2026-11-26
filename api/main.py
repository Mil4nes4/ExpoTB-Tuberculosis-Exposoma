"""API REST de ExpoTB (FastAPI). Documentación interactiva en http://localhost:8000/docs

Ejecutar:  uvicorn api.main:app --reload
"""
import json
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from expotb import config
from expotb.explain import Explainer

STATE: dict = {}


def _read_json(path) -> dict | None:
    """Lee un JSON de models/; devuelve None si no existe o está incompleto (p. ej., durante un entrenamiento)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # carga modelos, métricas y datos una sola vez al iniciar
    STATE["models"] = {p.stem: joblib.load(p) for p in config.MODELS_DIR.glob("*.joblib")}
    if config.EXPLAINER_MODEL in STATE["models"]:
        STATE["explainer"] = Explainer(STATE["models"][config.EXPLAINER_MODEL])
    STATE["interval"] = (_read_json(config.METRICS_PATH) or {}).get("intervalo")
    yield
    STATE.clear()


app = FastAPI(title="ExpoTB API", version="0.2.0", lifespan=lifespan,
              description="Estimación de la incidencia de tuberculosis a partir de factores del exposoma.")


class Indicators(BaseModel):
    """Seis indicadores del exposoma de un país. Un valor nulo se imputa con la mediana."""
    pm25: float | None = Field(None, description="PM2.5 (µg/m³)", examples=[25.0])
    clean_fuel: float | None = Field(None, description="Acceso a combustibles limpios (%)", examples=[70.0])
    urban_pop: float | None = Field(None, description="Población urbana (%)", examples=[55.0])
    gdp_pc: float | None = Field(None, description="PIB per cápita (US$)", examples=[6000.0])
    hiv: float | None = Field(None, description="Prevalencia de VIH (%)", examples=[0.5])
    undernourishment: float | None = Field(None, description="Desnutrición (%)", examples=[8.0])
    income: Literal["LIC", "LMC", "UMC", "HIC"] | None = Field(
        None, description="Grupo de ingreso del Banco Mundial (opcional). Se usa para imputar indicadores "
                          "faltantes; si no se indica, se imputa con la mediana global.", examples=["UMC"])

    def to_frame(self) -> pd.DataFrame:
        values = {k: (np.nan if v is None else v) for k, v in self.model_dump().items()}
        return pd.DataFrame([values])[config.FEATURES + [config.INCOME_COL]]


def _require_models():
    if not STATE.get("models"):
        raise HTTPException(503, "No hay modelos entrenados. Ejecuta: python -m expotb.train")


@app.get("/health")
def health():
    return {"status": "ok", "modelos_cargados": sorted(STATE.get("models", {}))}


@app.get("/models")
def models():
    return [{"id": k, "nombre": config.MODEL_LABELS[k], "tipo": config.MODEL_TYPE[k]}
            for k in config.MODEL_LABELS if k in STATE.get("models", {})]


@app.get("/metrics")
def metrics():
    data = _read_json(config.METRICS_PATH)
    if data is None:
        raise HTTPException(404, "metrics.json no existe o está incompleto. Ejecuta: python -m expotb.train")
    return data


@app.get("/dataset")
def dataset():
    if not config.DATASET_PATH.exists():
        raise HTTPException(404, "Dataset no encontrado. Ejecuta: python -m expotb.data")
    df = pd.read_csv(config.DATASET_PATH)
    return json.loads(df.to_json(orient="records"))


@app.post("/predict")
def predict(ind: Indicators):
    """Incidencia estimada (casos por 100 000 hab.) por cada modelo."""
    _require_models()
    X = ind.to_frame()
    return {name: float(m.predict(X)[0]) for name, m in STATE["models"].items()}


@app.post("/predict/intervalo")
def predict_interval(ind: Indicators):
    """Estimación del ensamble con un intervalo de incertidumbre.

    El intervalo se calibra con los residuales fuera de muestra por país (ver `expotb.train`), por lo que
    refleja el error esperado al estimar un país que el modelo no vio.
    """
    _require_models()
    interval = STATE.get("interval")
    if not interval or "ensamble" not in STATE["models"]:
        raise HTTPException(404, "Intervalo no disponible. Ejecuta: python -m expotb.train")
    estimate = float(STATE["models"]["ensamble"].predict(ind.to_frame())[0])
    log_est = float(np.log1p(max(estimate, 0.0)))
    return {"ensamble": estimate,
            "inferior": float(max(np.expm1(log_est + interval["q_inf"]), 0.0)),
            "superior": float(np.expm1(log_est + interval["q_sup"])),
            "nivel": interval["nivel"],
            "cobertura_por_pais": interval.get("cobertura_por_pais")}


@app.post("/explain")
def explain(ind: Indicators):
    """Contribución SHAP de cada indicador y de cada dimensión del exposoma."""
    if "explainer" not in STATE:
        raise HTTPException(503, "Explicador no disponible. Ejecuta: python -m expotb.train")
    return STATE["explainer"].explain_one(ind.to_frame())


@app.get("/explain/global")
def explain_global():
    data = _read_json(config.SHAP_GLOBAL_PATH)
    if data is None:
        raise HTTPException(404, "shap_global.json no existe o está incompleto. Ejecuta: python -m expotb.train")
    return data

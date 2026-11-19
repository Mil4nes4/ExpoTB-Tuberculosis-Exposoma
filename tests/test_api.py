"""Pruebas básicas de la API. Requieren haber ejecutado antes `python -m expotb.train`."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from expotb import config

pytestmark = pytest.mark.skipif(not config.METRICS_PATH.exists(), reason="modelos no entrenados")

SAMPLE = {"pm25": 30, "clean_fuel": 60, "urban_pop": 50, "gdp_pc": 4000, "hiv": 1.0, "undernourishment": 10}


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_predict_devuelve_todos_los_modelos():
    with TestClient(app) as client:
        preds = client.post("/predict", json=SAMPLE).json()
        assert set(preds) == set(config.MODEL_LABELS)
        assert all(v >= 0 for v in preds.values())


def test_predict_acepta_faltantes():
    with TestClient(app) as client:
        assert client.post("/predict", json={**SAMPLE, "hiv": None}).status_code == 200


def test_explain_por_dimension():
    with TestClient(app) as client:
        out = client.post("/explain", json=SAMPLE).json()
        assert set(out["dimensions"]) == set(config.DIMENSIONS)

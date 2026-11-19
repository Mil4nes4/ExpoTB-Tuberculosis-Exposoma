"""Preprocesamiento y definición de los modelos de ExpoTB."""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from expotb import config

RS = config.RANDOM_STATE


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quita registros sin incidencia o con más de 2 indicadores faltantes."""
    df = df.dropna(subset=[config.TARGET])
    return df[df[config.FEATURES].isna().sum(axis=1) <= 2].reset_index(drop=True)


def _pipe(regressor) -> Pipeline:
    # imputación y escalado dentro del pipeline: se ajustan solo con datos de entrenamiento (sin fuga)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", regressor),
    ])


def _log_target(estimator) -> TransformedTargetRegressor:
    # entrena sobre log1p(incidencia) y devuelve predicciones en la escala original con expm1
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)


def build_models() -> dict:
    base = {
        "ridge": Ridge(),
        "arbol_decision": DecisionTreeRegressor(random_state=RS),
        "random_forest": RandomForestRegressor(random_state=RS, n_jobs=-1),
        "xgboost": XGBRegressor(random_state=RS, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=RS),
        "svr": SVR(),
    }
    models = {name: _log_target(_pipe(reg)) for name, reg in base.items()}
    # el ensamble promedia en escala logarítmica las predicciones de RF, XGBoost y SVR
    voting = VotingRegressor([
        ("rf", _pipe(RandomForestRegressor(random_state=RS, n_jobs=-1))),
        ("xgb", _pipe(XGBRegressor(random_state=RS, n_jobs=-1))),
        ("svr", _pipe(SVR())),
    ])
    models["ensamble"] = _log_target(voting)
    return models

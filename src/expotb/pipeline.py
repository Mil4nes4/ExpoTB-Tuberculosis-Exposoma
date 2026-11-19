"""Preprocesamiento y definición de los modelos de ExpoTB."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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


class IncomeGroupImputer(BaseEstimator, TransformerMixin):
    """Imputa cada indicador con la mediana de su grupo de ingreso del Banco Mundial.

    Los faltantes de VIH y desnutrición se concentran en países de ingreso alto, por lo que la
    mediana global los sobrestima. Las medianas se calculan solo con los datos de entrenamiento.
    Si el grupo de ingreso no está disponible (columna ausente o valor nulo) se usa la mediana global.
    """

    def __init__(self, features=None, group_col=config.INCOME_COL):
        self.features = features
        self.group_col = group_col

    def fit(self, X, y=None):
        self.features_ = list(self.features or config.FEATURES)
        self.global_median_ = X[self.features_].median()
        self.group_medians_ = {}
        if self.group_col in X.columns:
            for group, part in X.groupby(self.group_col)[self.features_]:
                self.group_medians_[group] = part.median().fillna(self.global_median_)
        return self

    def transform(self, X):
        out = X[self.features_].copy()
        if self.group_col in X.columns:
            for group, medians in self.group_medians_.items():
                rows = (X[self.group_col] == group).to_numpy()
                out.loc[rows] = out.loc[rows].fillna(medians)
        return out.fillna(self.global_median_).to_numpy(dtype=float)


def _pipe(regressor, features, imputation: str, add_indicator: bool) -> Pipeline:
    # imputación y escalado dentro del pipeline: se ajustan solo con datos de entrenamiento (sin fuga)
    if imputation == "ingreso":
        imputer = IncomeGroupImputer(features=features)
    else:
        imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)
    return Pipeline([
        ("imputer", imputer),
        ("scaler", StandardScaler()),
        ("model", regressor),
    ])


def _log_target(estimator) -> TransformedTargetRegressor:
    # entrena sobre log1p(incidencia) y devuelve predicciones en la escala original con expm1
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)


def build_models(features=None, imputation: str = "ingreso", add_indicator: bool = False) -> dict:
    """imputation: "ingreso" (mediana por grupo de ingreso) o "mediana" (mediana global)."""
    features = list(features or config.FEATURES)

    def pipe(reg):
        return _pipe(reg, features, imputation, add_indicator)

    base = {
        "ridge": Ridge(),
        "arbol_decision": DecisionTreeRegressor(random_state=RS),
        "random_forest": RandomForestRegressor(random_state=RS, n_jobs=-1),
        "xgboost": XGBRegressor(random_state=RS, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=RS),
        "svr": SVR(),
    }
    models = {name: _log_target(pipe(reg)) for name, reg in base.items()}
    # el ensamble promedia en escala logarítmica las predicciones de RF, XGBoost y SVR
    voting = VotingRegressor([
        ("rf", pipe(RandomForestRegressor(random_state=RS, n_jobs=-1))),
        ("xgb", pipe(XGBRegressor(random_state=RS, n_jobs=-1))),
        ("svr", pipe(SVR())),
    ])
    models["ensamble"] = _log_target(voting)
    return models

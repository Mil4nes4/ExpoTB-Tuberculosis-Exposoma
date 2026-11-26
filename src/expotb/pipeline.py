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


class LogSkew(BaseEstimator, TransformerMixin):
    """Aplica log1p a los indicadores con asimetría positiva fuerte (PIB, VIH, PM2.5, desnutrición).

    No tiene parámetros que aprender, así que no hay fuga de información. Los modelos de árboles son
    invariantes a esta transformación; Ridge y SVR sí se benefician, porque dejan de estar dominados
    por unos pocos países con valores extremos.
    """

    def __init__(self, features=None, skewed=config.SKEWED_FEATURES):
        self.features = features
        self.skewed = skewed

    def fit(self, X, y=None):
        names = list(self.features or config.FEATURES)
        self.idx_ = [names.index(f) for f in self.skewed if f in names]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if self.idx_:
            X[:, self.idx_] = np.log1p(np.clip(X[:, self.idx_], 0, None))
        return X


def _pipe(regressor, features, imputation: str, add_indicator: bool, log_skew: bool = True) -> Pipeline:
    # imputación, transformación y escalado dentro del pipeline: se ajustan solo con datos de entrenamiento
    if imputation == "ingreso":
        imputer = IncomeGroupImputer(features=features)
    else:
        imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)
    steps = [("imputer", imputer)]
    if log_skew:
        steps.append(("logskew", LogSkew(features=features)))
    steps += [("scaler", StandardScaler()), ("model", regressor)]
    return Pipeline(steps)


def _log_target(estimator) -> TransformedTargetRegressor:
    # entrena sobre log1p(incidencia) y devuelve predicciones en la escala original con expm1
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)


def make_regressors(params: dict | None = None) -> dict:
    """Regresores base. `params` permite sobreescribir los hiperparámetros por modelo (salida de expotb.tune)."""
    params = params or {}
    return {
        "ridge": Ridge(**params.get("ridge", {})),
        "arbol_decision": DecisionTreeRegressor(random_state=RS, **params.get("arbol_decision", {})),
        "random_forest": RandomForestRegressor(random_state=RS, n_jobs=-1, **params.get("random_forest", {})),
        "xgboost": XGBRegressor(random_state=RS, n_jobs=-1, **params.get("xgboost", {})),
        "gradient_boosting": GradientBoostingRegressor(random_state=RS, **params.get("gradient_boosting", {})),
        "svr": SVR(**params.get("svr", {})),
    }


def build_models(features=None, imputation: str = "ingreso", add_indicator: bool = False,
                 params: dict | None = None, log_skew: bool = True) -> dict:
    """imputation: "ingreso" (mediana por grupo de ingreso) o "mediana" (mediana global).

    params: hiperparámetros por modelo (si es None se usan los valores por defecto).
    log_skew: aplica log1p a los indicadores asimétricos antes de escalar.
    """
    features = list(features or config.FEATURES)
    params = params or {}

    def pipe(reg):
        return _pipe(reg, features, imputation, add_indicator, log_skew)

    models = {name: _log_target(pipe(reg)) for name, reg in make_regressors(params).items()}
    # el ensamble promedia en escala logarítmica las predicciones de RF, XGBoost y SVR
    base = make_regressors(params)
    voting = VotingRegressor([
        ("rf", pipe(base["random_forest"])),
        ("xgb", pipe(base["xgboost"])),
        ("svr", pipe(base["svr"])),
    ])
    models["ensamble"] = _log_target(voting)
    return models


def inner_pipeline(name: str, features=None, imputation: str = "ingreso", params: dict | None = None) -> Pipeline:
    """Pipeline sin la transformación del objetivo, para ajustar hiperparámetros sobre log1p(incidencia)."""
    features = list(features or config.FEATURES)
    return _pipe(make_regressors(params)[name], features, imputation, False, True)

"""Explicabilidad con SHAP sobre el modelo de árboles (XGBoost por defecto).

Los valores SHAP están en escala log1p(incidencia): indican cuánto sube o baja
cada indicador la predicción respecto al valor base.
"""
import numpy as np
import pandas as pd
import shap

from expotb import config


class Explainer:
    def __init__(self, fitted_model):
        pipe = fitted_model.regressor_          # Pipeline ajustado dentro del TransformedTargetRegressor
        self.preprocess = pipe[:-1]
        self.tree_explainer = shap.TreeExplainer(pipe[-1])

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        cols = config.FEATURES + ([config.INCOME_COL] if config.INCOME_COL in X.columns else [])
        return self.tree_explainer.shap_values(self.preprocess.transform(X[cols]))

    @property
    def base_value(self) -> float:
        return float(np.ravel(self.tree_explainer.expected_value)[0])

    @staticmethod
    def by_dimension(values: dict) -> dict:
        return {dim: float(sum(values[f] for f in feats)) for dim, feats in config.DIMENSIONS.items()}

    def explain_one(self, row: pd.DataFrame) -> dict:
        sv = self.shap_values(row)[0]
        per_feature = {f: float(v) for f, v in zip(config.FEATURES, sv)}
        return {"base_value_log": self.base_value,
                "features": per_feature,
                "dimensions": self.by_dimension(per_feature)}

    def global_importance(self, X: pd.DataFrame) -> dict:
        mean_abs = np.abs(self.shap_values(X)).mean(axis=0)
        per_feature = {f: float(v) for f, v in zip(config.FEATURES, mean_abs)}
        return {"features": per_feature, "dimensions": self.by_dimension(per_feature)}

"""Optimización de hiperparámetros (búsqueda aleatoria con validación cruzada).

Cada protocolo de evaluación se optimiza con su propia validación cruzada, solo sobre sus datos de
entrenamiento, de modo que el conjunto de prueba nunca interviene:
- aleatorio: KFold K=5 mezclando registros
- por país: GroupKFold K=5 por código ISO3 (premia la generalización a países no vistos)
- temporal: ventanas crecientes por año (entrena hasta un año y valida en los dos siguientes)

El criterio es el RMSE sobre log1p(incidencia) (equivale al RMSLE), que no está dominado por los pocos
países con incidencias muy altas.
"""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from expotb import config
from expotb.pipeline import inner_pipeline

SPACES = {
    "ridge": {"alpha": [0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]},
    "arbol_decision": {"max_depth": [3, 4, 5, 7, 9, 12, None], "min_samples_leaf": [1, 3, 5, 10, 20],
                       "min_samples_split": [2, 5, 10]},
    "random_forest": {"n_estimators": [200, 400], "max_depth": [None, 8, 12, 16],
                      "min_samples_leaf": [1, 2, 3, 5, 10], "max_features": [0.5, 0.75, 1.0]},
    "xgboost": {"n_estimators": [200, 400, 600], "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 6, 8], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5],
                "reg_lambda": [1, 5, 10]},
    "gradient_boosting": {"n_estimators": [200, 400], "learning_rate": [0.03, 0.05, 0.1],
                          "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0],
                          "min_samples_leaf": [1, 5, 10]},
    "svr": {"C": [1, 3, 10, 30, 100], "gamma": ["scale", 0.05, 0.1, 0.3], "epsilon": [0.05, 0.1, 0.2]},
}
PARALLEL_INSIDE = {"random_forest", "xgboost"}   # estos modelos ya usan todos los núcleos por dentro


def temporal_splits(years, first_train_end: int = 2009, step: int = 2, horizon: int = 2):
    """Ventanas crecientes: entrena con años <= t y valida con los `horizon` años siguientes."""
    years = np.asarray(years)
    splits = []
    end = first_train_end
    last = config.TEMPORAL_SPLIT_YEAR
    while end + horizon <= last:
        tr = np.where(years <= end)[0]
        va = np.where((years > end) & (years <= end + horizon))[0]
        splits.append((tr, va))
        end += step
    return splits


def tune_model(name: str, X, y, splits, n_iter: int = 20) -> dict:
    """Busca los mejores hiperparámetros de `name`. `splits` es una lista de (índices_entrena, índices_valida)."""
    space = {f"model__{k}": v for k, v in SPACES[name].items()}
    size = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(
        inner_pipeline(name), space, n_iter=min(n_iter, size), cv=splits,
        scoring="neg_root_mean_squared_error", random_state=config.RANDOM_STATE,
        n_jobs=1 if name in PARALLEL_INSIDE else -1, refit=False)
    search.fit(X, np.log1p(y))
    best = {k.removeprefix("model__"): (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()}
    return {"params": best, "rmsle_cv": float(-search.best_score_)}

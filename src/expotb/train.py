"""Entrenamiento y evaluación de ExpoTB.

- Validación cruzada K=5 sobre el 80 % de los datos y prueba sobre el 20 % restante.
- Prueba temporal: entrena con 2000-2015 y evalúa con 2016-2019.
- Guarda los modelos (.joblib), las métricas (metrics.json) y la importancia SHAP global.

Uso:
    python -m expotb.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from expotb import config
from expotb.explain import Explainer
from expotb.pipeline import build_models, clean

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}


def metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def main():
    df = clean(pd.read_csv(config.DATASET_PATH))
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE)

    train_mask = df["year"] <= config.TEMPORAL_SPLIT_YEAR
    kfold = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"n_registros": len(df), "n_paises": int(df["iso3"].nunique()), "modelos": {}}
    fitted = {}
    for name, model in build_models().items():
        print(f"Entrenando {config.MODEL_LABELS[name]}...")
        cv = cross_validate(model, X_train, y_train, cv=kfold, scoring=SCORING)
        cv_summary = {}
        for m in SCORING:
            scores = cv[f"test_{m}"] * (1 if m == "r2" else -1)  # sklearn devuelve errores negativos
            cv_summary[m] = {"media": float(scores.mean()), "std": float(scores.std())}

        model.fit(X_train, y_train)
        test_summary = metrics(y_test, model.predict(X_test))

        temporal = clone(model).fit(X[train_mask], y[train_mask])
        temporal_summary = metrics(y[~train_mask], temporal.predict(X[~train_mask]))

        results["modelos"][name] = {"cv": cv_summary, "prueba": test_summary, "temporal": temporal_summary}
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        fitted[name] = model

    config.METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    explainer = Explainer(fitted[config.EXPLAINER_MODEL])
    config.SHAP_GLOBAL_PATH.write_text(json.dumps(explainer.global_importance(X_test), indent=2))

    table = pd.DataFrame({config.MODEL_LABELS[n]: {
        "R² CV": r["cv"]["r2"]["media"], "R² prueba": r["prueba"]["r2"],
        "MAE prueba": r["prueba"]["mae"], "RMSE prueba": r["prueba"]["rmse"]}
        for n, r in results["modelos"].items()}).T.round(3)
    print(table.to_string())


if __name__ == "__main__":
    main()

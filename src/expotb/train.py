"""Entrenamiento y evaluación de ExpoTB.

Protocolos de evaluación:
- Aleatorio: validación cruzada K=5 sobre el 80 % de los registros y prueba sobre el 20 % restante.
- Por país: igual, pero la partición separa países completos (ningún país aparece en entrenamiento
  y prueba a la vez) y la validación cruzada usa GroupKFold por código ISO3.
- Temporal: entrena con 2000-2015 y evalúa con 2016-2019.
Guarda los modelos (.joblib), las métricas (metrics.json) y la importancia SHAP global.

Uso:
    python -m expotb.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate, train_test_split

from expotb import config
from expotb.explain import Explainer
from expotb.pipeline import build_models, clean

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}


def metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def cv_summary(model, X, y, cv, groups=None) -> dict:
    res = cross_validate(model, X, y, cv=cv, groups=groups, scoring=SCORING)
    out = {}
    for m in SCORING:
        scores = res[f"test_{m}"] * (1 if m == "r2" else -1)  # sklearn devuelve errores negativos
        out[m] = {"media": float(scores.mean()), "std": float(scores.std())}
    return out


def input_columns(df: pd.DataFrame) -> list:
    """Indicadores más el grupo de ingreso (si existe), que usa el imputador."""
    return config.FEATURES + ([config.INCOME_COL] if config.INCOME_COL in df.columns else [])


def main():
    df = clean(pd.read_csv(config.DATASET_PATH))
    X, y, groups = df[input_columns(df)], df[config.TARGET], df["iso3"]
    rs = config.RANDOM_STATE

    idx = np.arange(len(df))
    tr_a, te_a = train_test_split(idx, test_size=0.2, random_state=rs)
    tr_p, te_p = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs).split(idx, groups=groups))
    train_mask = (df["year"] <= config.TEMPORAL_SPLIT_YEAR).to_numpy()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"n_registros": len(df), "n_paises": int(groups.nunique()), "modelos": {}}
    fitted = {}
    for name, model in build_models().items():
        print(f"Entrenando {config.MODEL_LABELS[name]}...", flush=True)
        kfold = KFold(n_splits=5, shuffle=True, random_state=rs)
        cv = cv_summary(clone(model), X.iloc[tr_a], y.iloc[tr_a], kfold)
        fit = clone(model).fit(X.iloc[tr_a], y.iloc[tr_a])
        prueba = metrics(y.iloc[te_a], fit.predict(X.iloc[te_a]))

        cv_p = cv_summary(clone(model), X.iloc[tr_p], y.iloc[tr_p], GroupKFold(n_splits=5), groups.iloc[tr_p])
        fit_p = clone(model).fit(X.iloc[tr_p], y.iloc[tr_p])
        prueba_p = metrics(y.iloc[te_p], fit_p.predict(X.iloc[te_p]))

        temporal = clone(model).fit(X[train_mask], y[train_mask])
        temporal_summary = metrics(y[~train_mask], temporal.predict(X[~train_mask]))

        results["modelos"][name] = {"cv": cv, "prueba": prueba, "temporal": temporal_summary,
                                    "por_pais": {"cv": cv_p, "prueba": prueba_p}}
        joblib.dump(fit, config.MODELS_DIR / f"{name}.joblib")
        fitted[name] = fit

    config.METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    explainer = Explainer(fitted[config.EXPLAINER_MODEL])
    config.SHAP_GLOBAL_PATH.write_text(json.dumps(explainer.global_importance(X.iloc[te_a]), indent=2))

    table = pd.DataFrame({config.MODEL_LABELS[n]: {
        "R² aleatorio": r["prueba"]["r2"], "R² por país": r["por_pais"]["prueba"]["r2"],
        "R² temporal": r["temporal"]["r2"], "RMSE aleatorio": r["prueba"]["rmse"]}
        for n, r in results["modelos"].items()}).T.round(3)
    print(table.to_string())


if __name__ == "__main__":
    main()

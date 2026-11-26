"""Entrenamiento y evaluación de ExpoTB.

Protocolos de evaluación:
- Aleatorio: validación cruzada K=5 sobre el 80 % de los registros y prueba sobre el 20 % restante.
- Por país: igual, pero la partición separa países completos (ningún país aparece en entrenamiento
  y prueba a la vez) y la validación cruzada usa GroupKFold por código ISO3.
- Temporal: entrena con 2000-2015 y evalúa con 2016-2019.

Los hiperparámetros se optimizan dentro de cada protocolo, solo con sus datos de entrenamiento
(ver expotb.tune). Además se reportan dos líneas base (mediana global y mediana por grupo de ingreso),
los errores por grupo de ingreso y por nivel de incidencia, y un intervalo de incertidumbre calibrado
con los residuales por país.

Guarda los modelos servidos por la API (.joblib), las métricas (metrics.json) y la importancia SHAP global.
Los modelos servidos se entrenan con el 80 % aleatorio y los hiperparámetros del protocolo por país,
porque el uso previsto es estimar países nuevos.

Uso:
    python -m expotb.train                 # con optimización de hiperparámetros
    python -m expotb.train --sin-ajuste    # hiperparámetros por defecto (rápido)
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, cross_val_predict, cross_validate,
                                     train_test_split)

from expotb import config
from expotb.explain import Explainer
from expotb.pipeline import build_models, clean
from expotb.tune import temporal_splits, tune_model

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
INTERVAL_LEVEL = 0.8
INCIDENCE_BINS = [(0, 50, "< 50"), (50, 200, "50 a 200"), (200, np.inf, "≥ 200")]
REPEATED_ORIGINS = (2007, 2011, 2015)   # holdout temporal repetido: entrena <= t, prueba t+1..t+4
REPEATED_HORIZON = 4


def metrics(y_true, y_pred) -> dict:
    """MAE, RMSE y R² en la escala original, más RMSLE y R² en escala logarítmica.

    Las métricas en escala original las domina un grupo pequeño de países con incidencia muy alta;
    las logarítmicas miden el error relativo y son más representativas del conjunto.
    """
    y_true, y_pred = np.asarray(y_true, float), np.clip(np.asarray(y_pred, float), 0, None)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "rmsle": float(np.sqrt(mean_squared_error(lt, lp))),
            "r2_log": float(r2_score(lt, lp))}


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


class IncomeMedianBaseline(BaseEstimator, RegressorMixin):
    """Línea base: predice la mediana de incidencia del grupo de ingreso (sin usar el exposoma)."""

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.medians_ = y.groupby(X[config.INCOME_COL]).median()
        self.global_ = float(y.median())
        return self

    def predict(self, X):
        return X[config.INCOME_COL].map(self.medians_).fillna(self.global_).to_numpy(dtype=float)


class GlobalMedianBaseline(BaseEstimator, RegressorMixin):
    """Línea base: predice siempre la mediana global de incidencia del entrenamiento."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


def stratified_errors(y_true, y_pred, income) -> dict:
    """Errores por nivel de incidencia real y por grupo de ingreso (sesgo = predicho - real)."""
    y_true, y_pred = np.asarray(y_true, float), np.clip(np.asarray(y_pred, float), 0, None)
    out = {"por_incidencia": {}, "por_ingreso": {}}
    for lo, hi, label in INCIDENCE_BINS:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.any():
            out["por_incidencia"][label] = {
                "n": int(mask.sum()), "mae": float(np.abs(y_pred[mask] - y_true[mask]).mean()),
                "sesgo": float((y_pred[mask] - y_true[mask]).mean())}
    income = np.asarray(income)
    for group in ("LIC", "LMC", "UMC", "HIC"):
        mask = income == group
        if mask.any():
            out["por_ingreso"][group] = {
                "n": int(mask.sum()), "mae": float(np.abs(y_pred[mask] - y_true[mask]).mean()),
                "rmsle": float(np.sqrt(np.mean((np.log1p(y_pred[mask]) - np.log1p(y_true[mask])) ** 2)))}
    return out


def interval_quantiles(model, X, y, groups) -> dict:
    """Cuantiles de los residuales logarítmicos fuera de muestra por país (GroupKFold) para el intervalo."""
    oof = cross_val_predict(clone(model), X, y, cv=GroupKFold(n_splits=5), groups=groups)
    resid = np.log1p(np.asarray(y, float)) - np.log1p(np.clip(oof, 0, None))
    lo, hi = (1 - INTERVAL_LEVEL) / 2, 1 - (1 - INTERVAL_LEVEL) / 2
    return {"nivel": INTERVAL_LEVEL, "q_inf": float(np.quantile(resid, lo)), "q_sup": float(np.quantile(resid, hi))}


def interval_coverage(q: dict, y_true, y_pred) -> float:
    resid = np.log1p(np.asarray(y_true, float)) - np.log1p(np.clip(np.asarray(y_pred, float), 0, None))
    return float(np.mean((resid >= q["q_inf"]) & (resid <= q["q_sup"])))


def rolling_origins(years, origins=REPEATED_ORIGINS, horizon=REPEATED_HORIZON):
    """Holdout temporal repetido: para cada origen t entrena con años <= t y prueba con t+1..t+horizon."""
    years = np.asarray(years)
    return {int(t): (np.where(years <= t)[0], np.where((years > t) & (years <= t + horizon))[0]) for t in origins}


def repeated_temporal(model, X, y, origins: dict) -> dict:
    """Métricas en cada origen y su media (Cerqueira et al., 2020 recomiendan varios periodos de prueba)."""
    per = {str(t): metrics(y.iloc[te], clone(model).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te]))
           for t, (tr, te) in origins.items()}
    keys = next(iter(per.values())).keys()
    return {"origenes": per,
            "media": {k: float(np.mean([v[k] for v in per.values()])) for k in keys},
            "std": {k: float(np.std([v[k] for v in per.values()])) for k in keys}}


def tune_all(X, y, splits, n_iter, label):
    """Optimiza cada regresor individual con las particiones `splits` y devuelve {modelo: hiperparámetros}."""
    params, scores = {}, {}
    for name in ("ridge", "arbol_decision", "random_forest", "xgboost", "gradient_boosting", "svr"):
        t0 = time.time()
        res = tune_model(name, X, y, splits, n_iter)
        params[name], scores[name] = res["params"], res["rmsle_cv"]
        print(f"  [{label}] {config.MODEL_LABELS[name]}: RMSLE VC {res['rmsle_cv']:.3f} "
              f"({time.time() - t0:.0f} s) {res['params']}", flush=True)
    return params, scores


def main(tune: bool = True, n_iter: int = 20, reusar: bool = False, reajustar: tuple = ()):
    df = clean(pd.read_csv(config.DATASET_PATH))
    X, y, groups = df[input_columns(df)], df[config.TARGET], df["iso3"]
    rs = config.RANDOM_STATE

    idx = np.arange(len(df))
    tr_a, te_a = train_test_split(idx, test_size=0.2, random_state=rs)
    tr_p, te_p = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs).split(idx, groups=groups))
    train_mask = (df["year"] <= config.TEMPORAL_SPLIT_YEAR).to_numpy()
    tr_t, te_t = idx[train_mask], idx[~train_mask]
    origins = rolling_origins(df["year"].to_numpy())
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # particiones de la búsqueda de hiperparámetros, expresadas sobre el subconjunto de entrenamiento
    protocols = {
        "aleatorio": (tr_a, list(KFold(5, shuffle=True, random_state=rs).split(tr_a))),
        "por_pais": (tr_p, list(GroupKFold(5).split(tr_p, groups=groups.iloc[tr_p]))),
        "temporal": (tr_t, temporal_splits(df["year"].iloc[tr_t].to_numpy())),
    }

    t_start = time.time()
    params, tuning = {}, {}
    prev = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))["optimizacion"] if reusar else None
    for proto, (tr, splits) in protocols.items():
        if reusar:
            # reutiliza los hiperparámetros ya elegidos y solo reajusta los modelos pedidos
            params[proto] = {k: dict(v) for k, v in prev["hiperparametros"][proto].items()}
            tuning[proto] = dict(prev["rmsle_validacion"][proto])
            for name in reajustar:
                res = tune_model(name, X.iloc[tr], y.iloc[tr], splits, n_iter)
                params[proto][name], tuning[proto][name] = res["params"], res["rmsle_cv"]
                print(f"  [{proto}] {config.MODEL_LABELS[name]}: RMSLE VC {res['rmsle_cv']:.3f} {res['params']}", flush=True)
        elif tune:
            print(f"Optimizando hiperparámetros ({proto})...", flush=True)
            params[proto], tuning[proto] = tune_all(X.iloc[tr], y.iloc[tr], splits, n_iter, proto)
        else:
            params[proto] = {}

    results = {"n_registros": len(df), "n_paises": int(groups.nunique()),
               "optimizacion": {"activa": tune or reusar,
                                "n_iteraciones": prev["n_iteraciones"] if reusar else (n_iter if tune else 0),
                                **({"reajuste": {"modelos": list(reajustar), "n_iteraciones": n_iter}} if reusar else {}),
                                "criterio": "RMSE sobre log1p(incidencia) en la validación cruzada del protocolo",
                                "hiperparametros": params, "rmsle_validacion": tuning},
               "modelos": {}}

    fitted, preds_pais = {}, {}
    for name in build_models():
        print(f"Entrenando {config.MODEL_LABELS[name]}...", flush=True)
        entry = {}
        # aleatorio
        mod = build_models(params=params["aleatorio"])[name]
        entry["cv"] = cv_summary(clone(mod), X.iloc[tr_a], y.iloc[tr_a], KFold(5, shuffle=True, random_state=rs))
        entry["prueba"] = metrics(y.iloc[te_a], clone(mod).fit(X.iloc[tr_a], y.iloc[tr_a]).predict(X.iloc[te_a]))
        # por país
        mod = build_models(params=params["por_pais"])[name]
        cv_p = cv_summary(clone(mod), X.iloc[tr_p], y.iloc[tr_p], GroupKFold(n_splits=5), groups.iloc[tr_p])
        pred_p = clone(mod).fit(X.iloc[tr_p], y.iloc[tr_p]).predict(X.iloc[te_p])
        preds_pais[name] = pred_p
        entry["por_pais"] = {"cv": cv_p, "prueba": metrics(y.iloc[te_p], pred_p),
                             "estratificado": stratified_errors(y.iloc[te_p], pred_p, df[config.INCOME_COL].iloc[te_p])}
        # temporal
        mod = build_models(params=params["temporal"])[name]
        entry["temporal"] = metrics(y.iloc[te_t], clone(mod).fit(X.iloc[tr_t], y.iloc[tr_t]).predict(X.iloc[te_t]))
        entry["temporal_repetido"] = repeated_temporal(mod, X, y, origins)
        results["modelos"][name] = entry
        # modelo servido por la API: 80 % aleatorio con los hiperparámetros del protocolo por país
        fitted[name] = clone(build_models(params=params["por_pais"])[name]).fit(X.iloc[tr_a], y.iloc[tr_a])
        joblib.dump(fitted[name], config.MODELS_DIR / f"{name}.joblib")

    # líneas base (no usan el exposoma)
    baselines = {"mediana_global": GlobalMedianBaseline(), "mediana_por_ingreso": IncomeMedianBaseline()}
    results["linea_base"] = {}
    for bname, base in baselines.items():
        results["linea_base"][bname] = {
            "aleatorio": metrics(y.iloc[te_a], clone(base).fit(X.iloc[tr_a], y.iloc[tr_a]).predict(X.iloc[te_a])),
            "por_pais": metrics(y.iloc[te_p], clone(base).fit(X.iloc[tr_p], y.iloc[tr_p]).predict(X.iloc[te_p])),
            "temporal": metrics(y.iloc[te_t], clone(base).fit(X.iloc[tr_t], y.iloc[tr_t]).predict(X.iloc[te_t])),
            "temporal_repetido": repeated_temporal(base, X, y, origins),
        }

    # intervalo de incertidumbre del ensamble (residuales fuera de muestra por país)
    print("Calibrando el intervalo de incertidumbre...", flush=True)
    ens = build_models(params=params["por_pais"])["ensamble"]
    q_eval = interval_quantiles(ens, X.iloc[tr_p], y.iloc[tr_p], groups.iloc[tr_p])
    q_served = interval_quantiles(ens, X.iloc[tr_a], y.iloc[tr_a], groups.iloc[tr_a])
    results["intervalo"] = {**q_served,
                            "cobertura_por_pais": interval_coverage(q_eval, y.iloc[te_p], preds_pais["ensamble"]),
                            "descripcion": "Intervalo del ensamble en escala logarítmica; la cobertura se mide en países no vistos."}

    results["duracion_s"] = round(time.time() - t_start)
    config.METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    explainer = Explainer(fitted[config.EXPLAINER_MODEL])
    config.SHAP_GLOBAL_PATH.write_text(json.dumps(explainer.global_importance(X.iloc[te_a]), indent=2),
                                       encoding="utf-8")

    table = pd.DataFrame({config.MODEL_LABELS[n]: {
        "R² aleatorio": r["prueba"]["r2"], "R² por país": r["por_pais"]["prueba"]["r2"],
        "R² temporal": r["temporal"]["r2"], "R² log por país": r["por_pais"]["prueba"]["r2_log"]}
        for n, r in results["modelos"].items()}).T.round(3)
    print(table.to_string())
    print("Cobertura del intervalo (países no vistos):", round(results["intervalo"]["cobertura_por_pais"], 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y evalúa los modelos de ExpoTB.")
    parser.add_argument("--sin-ajuste", action="store_true", help="usa hiperparámetros por defecto (rápido)")
    parser.add_argument("--n-iter", type=int, default=20, help="combinaciones por modelo en la búsqueda aleatoria")
    parser.add_argument("--reusar-params", action="store_true",
                        help="reutiliza los hiperparámetros de models/metrics.json (reentrena en minutos)")
    parser.add_argument("--reajustar", default="", help="modelos a re-optimizar con --reusar-params, separados por coma")
    args = parser.parse_args()
    main(tune=not args.sin_ajuste, n_iter=args.n_iter, reusar=args.reusar_params,
         reajustar=tuple(m for m in args.reajustar.split(",") if m))

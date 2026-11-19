"""Comparación de conjuntos de variables: base (6) vs. ampliado (10).

Todas las variantes usan exactamente los mismos registros (se limpia con los 6 indicadores base),
los mismos modelos, los mismos hiperparámetros por defecto y las mismas particiones, de modo que
la única diferencia es el conjunto de variables.

Variantes:
  base          los 6 indicadores originales
  ampliado      base + alcohol, saneamiento, gasto en salud y BCG
  ampliado_ind  ampliado + indicadores de faltante en el imputador

Protocolos de evaluación:
  aleatorio  partición aleatoria por filas (el protocolo de train.py)
  por_pais   partición por país: ningún país aparece a la vez en entrenamiento y prueba
  temporal   entrena 2000-2015, prueba 2016-2019

Uso:
    python -m expotb.compare
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate, train_test_split

from expotb import config
from expotb.pipeline import build_models, clean
from expotb.train import SCORING, metrics

VARIANTS = {
    "base": (config.FEATURE_SETS["base"], False),
    "ampliado": (config.FEATURE_SETS["ampliado"], False),
    "ampliado_ind": (config.FEATURE_SETS["ampliado"], True),
}


def _cv_summary(model, X, y, cv, groups=None) -> dict:
    res = cross_validate(model, X, y, cv=cv, groups=groups, scoring=SCORING)
    out = {}
    for m in SCORING:
        scores = res[f"test_{m}"] * (1 if m == "r2" else -1)
        out[m] = {"media": float(scores.mean()), "std": float(scores.std())}
    return out


def main():
    df = clean(pd.read_csv(config.DATASET_PATH))
    missing = [c for c in config.FEATURE_SETS["ampliado"] if c not in df.columns]
    if missing:
        sys.exit(f"Faltan columnas {missing}. Reconstruye el dataset: python -m expotb.data --tb data/raw/tuberculosis.csv")
    y, groups = df[config.TARGET], df["iso3"]
    rs = config.RANDOM_STATE

    idx = np.arange(len(df))
    tr_a, te_a = train_test_split(idx, test_size=0.2, random_state=rs)
    tr_p, te_p = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs).split(idx, groups=groups))
    tr_t = idx[(df["year"] <= config.TEMPORAL_SPLIT_YEAR).to_numpy()]
    te_t = idx[(df["year"] > config.TEMPORAL_SPLIT_YEAR).to_numpy()]

    results = {"n_registros": len(df), "n_paises": int(groups.nunique()),
               "variantes": {k: {"variables": v[0], "indicadores_faltante": v[1]} for k, v in VARIANTS.items()},
               "resultados": {}}

    for vname, (features, add_ind) in VARIANTS.items():
        X = df[features]
        results["resultados"][vname] = {}
        for mname, model in build_models(add_ind).items():
            print(f"[{vname}] {config.MODEL_LABELS[mname]}...", flush=True)
            r = {}
            # aleatorio
            kf = KFold(n_splits=5, shuffle=True, random_state=rs)
            fit = clone(model).fit(X.iloc[tr_a], y.iloc[tr_a])
            r["aleatorio"] = {"cv": _cv_summary(clone(model), X.iloc[tr_a], y.iloc[tr_a], kf),
                              "prueba": metrics(y.iloc[te_a], fit.predict(X.iloc[te_a]))}
            # por país
            gk = GroupKFold(n_splits=5)
            fit = clone(model).fit(X.iloc[tr_p], y.iloc[tr_p])
            r["por_pais"] = {"cv": _cv_summary(clone(model), X.iloc[tr_p], y.iloc[tr_p], gk, groups.iloc[tr_p]),
                             "prueba": metrics(y.iloc[te_p], fit.predict(X.iloc[te_p]))}
            # temporal
            fit = clone(model).fit(X.iloc[tr_t], y.iloc[tr_t])
            r["temporal"] = {"prueba": metrics(y.iloc[te_t], fit.predict(X.iloc[te_t]))}
            results["resultados"][vname][mname] = r
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.COMPARISON_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))  # guarda avance

    for proto in ("aleatorio", "por_pais", "temporal"):
        rows = {config.MODEL_LABELS[m]: {v: results["resultados"][v][m][proto]["prueba"]["r2"] for v in VARIANTS}
                for m in config.MODEL_LABELS}
        print(f"\nR² en prueba - protocolo {proto}")
        print(pd.DataFrame(rows).T.round(3).to_string())
    print(f"\nResultados completos en {config.COMPARISON_PATH}")


if __name__ == "__main__":
    main()

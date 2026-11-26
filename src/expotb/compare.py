"""Comparación de conjuntos de variables y de estrategias de imputación.

Todas las variantes usan exactamente los mismos registros (se limpia con los 6 indicadores base),
los mismos modelos, los mismos hiperparámetros por defecto y las mismas particiones.

Variantes:
  base          6 indicadores, imputación por grupo de ingreso (configuración del informe)
  base_mediana  6 indicadores, imputación con la mediana global
  ampliado      base + alcohol, saneamiento, gasto en salud y BCG, imputación por grupo de ingreso

Protocolos de evaluación: aleatorio, por país y temporal (ver expotb.train).

Uso:
    python -m expotb.compare
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

from expotb import config
from expotb.pipeline import build_models, clean
from expotb.train import cv_summary, metrics

VARIANTS = {
    "base": (config.FEATURE_SETS["base"], "ingreso"),
    "base_mediana": (config.FEATURE_SETS["base"], "mediana"),
    "ampliado": (config.FEATURE_SETS["ampliado"], "ingreso"),
}


def main():
    df = clean(pd.read_csv(config.DATASET_PATH))
    needed = config.FEATURE_SETS["ampliado"] + [config.INCOME_COL]
    missing = [c for c in needed if c not in df.columns]
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
               "variantes": {k: {"variables": v[0], "imputacion": v[1]} for k, v in VARIANTS.items()},
               "resultados": {}}

    for vname, (features, imputation) in VARIANTS.items():
        # la imputación por mediana global no usa el grupo de ingreso (columna de texto)
        X = df[features + ([config.INCOME_COL] if imputation == "ingreso" else [])]
        results["resultados"][vname] = {}
        for mname, model in build_models(features, imputation).items():
            print(f"[{vname}] {config.MODEL_LABELS[mname]}...", flush=True)
            r = {}
            kf = KFold(n_splits=5, shuffle=True, random_state=rs)
            fit = clone(model).fit(X.iloc[tr_a], y.iloc[tr_a])
            r["aleatorio"] = {"cv": cv_summary(clone(model), X.iloc[tr_a], y.iloc[tr_a], kf),
                              "prueba": metrics(y.iloc[te_a], fit.predict(X.iloc[te_a]))}
            gk = GroupKFold(n_splits=5)
            fit = clone(model).fit(X.iloc[tr_p], y.iloc[tr_p])
            r["por_pais"] = {"cv": cv_summary(clone(model), X.iloc[tr_p], y.iloc[tr_p], gk, groups.iloc[tr_p]),
                             "prueba": metrics(y.iloc[te_p], fit.predict(X.iloc[te_p]))}
            fit = clone(model).fit(X.iloc[tr_t], y.iloc[tr_t])
            r["temporal"] = {"prueba": metrics(y.iloc[te_t], fit.predict(X.iloc[te_t]))}
            results["resultados"][vname][mname] = r
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.COMPARISON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")  # guarda avance

    for proto in ("aleatorio", "por_pais", "temporal"):
        rows = {config.MODEL_LABELS[m]: {v: results["resultados"][v][m][proto]["prueba"]["r2"] for v in VARIANTS}
                for m in config.MODEL_LABELS}
        print(f"\nR² en prueba - protocolo {proto}")
        print(pd.DataFrame(rows).T.round(3).to_string())
    print(f"\nResultados completos en {config.COMPARISON_PATH}")


if __name__ == "__main__":
    main()

"""Adquisición e integración de datos: incidencia de TB (OMS) + indicadores del Banco Mundial.

Uso:
    python -m expotb.data                       # descarga la incidencia desde Our World in Data
    python -m expotb.data --tb data/raw/tb.csv  # usa un CSV local (p. ej. el de Kaggle)
"""
import argparse

import pandas as pd

from expotb import config

OWID_URL = "https://ourworldindata.org/grapher/incidence-of-tuberculosis-sdgs.csv"


def load_tb_incidence(path: str | None = None) -> pd.DataFrame:
    """Devuelve columnas iso3, year, incidence. Acepta el formato de OWID (Entity, Code, Year, valor)."""
    df = pd.read_csv(path or OWID_URL, storage_options=None if path else {"User-Agent": "ExpoTB"})
    value_col = [c for c in df.columns if c not in ("Entity", "Code", "Year")][-1]
    df = df.rename(columns={"Code": "iso3", "Year": "year", "Entity": "country", value_col: config.TARGET})
    # los agregados regionales de OWID no tienen código ISO3 de 3 letras
    df = df[df["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    return df[["iso3", "country", "year", config.TARGET]]


def load_world_bank() -> pd.DataFrame:
    """Descarga los 6 indicadores con wbgapi (solo países, sin agregados)."""
    import wbgapi as wb

    codes = list(config.WB_INDICATORS.values())
    df = wb.data.DataFrame(codes, economy="all", time=range(config.YEAR_START, config.YEAR_END + 1),
                           skipAggs=True, columns="series").reset_index()
    df = df.rename(columns={"economy": "iso3", "time": "year"})
    df["year"] = df["year"].astype(str).str.replace("YR", "").astype(int)
    return df.rename(columns={v: k for k, v in config.WB_INDICATORS.items()})


def build_dataset(tb_path: str | None = None) -> pd.DataFrame:
    tb = load_tb_incidence(tb_path)
    wb_df = load_world_bank()
    df = tb.merge(wb_df, on=["iso3", "year"], how="inner")
    df = df[df["year"].between(config.YEAR_START, config.YEAR_END)]
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Construye data/processed/expotb.csv")
    parser.add_argument("--tb", help="CSV local con la incidencia de TB (formato OWID)")
    args = parser.parse_args()
    df = build_dataset(args.tb)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"Dataset guardado en {config.DATASET_PATH}: {len(df)} registros, {df['iso3'].nunique()} países")


if __name__ == "__main__":
    main()

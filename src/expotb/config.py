"""Configuración central de ExpoTB: rutas, variables y dimensiones del exposoma."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATASET_PATH = DATA_PROCESSED / "expotb.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_GLOBAL_PATH = MODELS_DIR / "shap_global.json"

RANDOM_STATE = 42
YEAR_START, YEAR_END = 2000, 2019          # se excluye 2020-2022 por la distorsión de la COVID-19
TEMPORAL_SPLIT_YEAR = 2015                # prueba temporal: entrena <=2015, prueba 2016-2019
TARGET = "incidence"                      # casos nuevos por 100 000 habitantes

# Indicadores del Banco Mundial (World Development Indicators)
WB_INDICATORS = {
    "pm25": "EN.ATM.PM25.MC.M3",          # PM2.5 promedio anual (µg/m³)
    "clean_fuel": "EG.CFT.ACCS.ZS",       # acceso a combustibles limpios para cocinar (%)
    "urban_pop": "SP.URB.TOTL.IN.ZS",     # población urbana (%)
    "gdp_pc": "NY.GDP.PCAP.CD",           # PIB per cápita (US$ actuales)
    "hiv": "SH.DYN.AIDS.ZS",              # prevalencia de VIH (% de 15-49 años)
    "undernourishment": "SN.ITK.DEFC.ZS", # prevalencia de desnutrición (%)
}
FEATURES = list(WB_INDICATORS)             # conjunto "base" (los 6 indicadores originales)

# Variables adicionales para el conjunto "ampliado" (se comparan en expotb.compare)
WB_INDICATORS_EXTRA = {
    "alcohol": "SH.ALC.PCAP.LI",          # consumo de alcohol per cápita (litros de alcohol puro)
    "sanitation": "SH.STA.BASS.ZS",       # población con servicios básicos de saneamiento (%)
    "health_exp": "SH.XPD.CHEX.GD.ZS",    # gasto corriente en salud (% del PIB)
}
OWID_EXTRA = {                            # series de Our World in Data (CSV)
    "bcg": "https://ourworldindata.org/grapher/bcg-immunization-coverage-for-tb-among-1-year-olds.csv",
}
EXTRA_FEATURES = list(WB_INDICATORS_EXTRA) + list(OWID_EXTRA)
FEATURE_SETS = {"base": FEATURES, "ampliado": FEATURES + EXTRA_FEATURES}

FEATURE_LABELS = {
    "pm25": "PM2.5 (µg/m³)",
    "clean_fuel": "Acceso a combustibles limpios (%)",
    "urban_pop": "Población urbana (%)",
    "gdp_pc": "PIB per cápita (US$)",
    "hiv": "Prevalencia de VIH (%)",
    "undernourishment": "Prevalencia de desnutrición (%)",
    "alcohol": "Consumo de alcohol (L/persona)",
    "sanitation": "Saneamiento básico (%)",
    "health_exp": "Gasto en salud (% del PIB)",
    "bcg": "Cobertura BCG (%)",
}

DIMENSIONS = {
    "Externa": ["pm25"],
    "Hogar": ["clean_fuel"],
    "Social": ["urban_pop", "gdp_pc"],
    "Biológica": ["hiv", "undernourishment"],
}

# Dimensiones del conjunto ampliado (solo para el análisis comparativo)
DIMENSIONS_AMPLIADO = {
    "Externa": ["pm25"],
    "Hogar": ["clean_fuel", "sanitation"],
    "Social": ["urban_pop", "gdp_pc", "health_exp"],
    "Biológica": ["hiv", "undernourishment", "alcohol", "bcg"],
}
COMPARISON_PATH = MODELS_DIR / "comparacion.json"

# Rangos para los controles de la interfaz (mín, máx, valor por defecto)
FEATURE_RANGES = {
    "pm25": (0.0, 120.0, 25.0),
    "clean_fuel": (0.0, 100.0, 70.0),
    "urban_pop": (0.0, 100.0, 55.0),
    "gdp_pc": (100.0, 120000.0, 6000.0),
    "hiv": (0.0, 30.0, 0.5),
    "undernourishment": (0.0, 70.0, 8.0),
}

MODEL_LABELS = {
    "ridge": "Ridge",
    "arbol_decision": "Árbol de Decisión",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "svr": "SVR",
    "ensamble": "Ensamble (RF + XGB + SVR)",
}
MODEL_TYPE = {
    "ridge": "Caja blanca", "arbol_decision": "Caja blanca",
    "random_forest": "Caja negra", "xgboost": "Caja negra",
    "gradient_boosting": "Caja negra", "svr": "Caja negra",
    "ensamble": "Ensamble",
}
EXPLAINER_MODEL = "xgboost"  # modelo de árboles usado para los valores SHAP

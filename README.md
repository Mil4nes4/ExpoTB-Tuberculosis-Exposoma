# ExpoTB

**Modelo de machine learning para predecir la incidencia de tuberculosis a partir de factores del exposoma ambiental y socioeconómico.**

Trabajo grupal del curso de Inteligencia Artificial, Facultad de Ingeniería de Sistemas e Informática, UNMSM.

ExpoTB estima la tasa de incidencia anual de tuberculosis (casos nuevos por 100 000 habitantes) de un país a partir de seis indicadores del exposoma, organizados en cuatro dimensiones:

| Dimensión | Indicador | Código Banco Mundial |
|---|---|---|
| Externa | PM2.5 (µg/m³) | `EN.ATM.PM25.MC.M3` |
| Hogar | Acceso a combustibles limpios para cocinar (%) | `EG.CFT.ACCS.ZS` |
| Social | Población urbana (%) | `SP.URB.TOTL.IN.ZS` |
| Social | PIB per cápita (US$) | `NY.GDP.PCAP.CD` |
| Biológica | Prevalencia de VIH (%) | `SH.DYN.AIDS.ZS` |
| Biológica | Prevalencia de desnutrición (%) | `SN.ITK.DEFC.ZS` |

> ⚠️ Prototipo académico. No es una herramienta clínica ni de diagnóstico.

## Arquitectura

Arquitectura por capas con el frontend desacoplado del backend:

```
┌──────────────────────────────────────────────────────────────┐
│ Presentación   Streamlit (multipágina) + Plotly               │
│                Explorador · Predicción · Comparación · SHAP   │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP / JSON
┌───────────────────────────▼──────────────────────────────────┐
│ Servicio       FastAPI + Uvicorn                              │
│                /predict · /explain · /metrics · /dataset      │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────▼──────────────────────────────────┐
│ Modelos y      scikit-learn, XGBoost, SHAP                    │
│ explicabilidad Ridge, Árbol, RF, XGB, GB, SVR + ensamble      │
└───────────────────────────┬──────────────────────────────────┘
┌───────────────────────────▼──────────────────────────────────┐
│ Datos          OMS (Our World in Data / Kaggle) + Banco       │
│                Mundial (wbgapi) → pandas                      │
└──────────────────────────────────────────────────────────────┘
```

## Estructura del repositorio

```
expotb/
├── api/main.py               # API REST (FastAPI)
├── app/
│   ├── Inicio.py             # página principal de Streamlit
│   ├── api_client.py         # conexión interfaz ↔ API
│   └── pages/                # módulos de la interfaz
├── src/expotb/
│   ├── config.py             # rutas, variables y dimensiones
│   ├── data.py               # adquisición e integración de datos
│   ├── pipeline.py           # preprocesamiento y modelos
│   ├── train.py              # validación cruzada, prueba y prueba temporal
│   ├── compare.py            # comparación de conjuntos de variables (base vs. ampliado)
│   └── explain.py            # valores SHAP
├── data/                     # raw/ y processed/ (no se suben a git)
├── models/                   # modelos .joblib y métricas (no se suben a git)
├── notebooks/                # análisis exploratorio
└── tests/
```

## Instalación

```bash
git clone <url-del-repositorio>
cd expotb
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Construir el dataset (OMS + Banco Mundial, 2000-2019)
python -m expotb.data
#    o con el CSV de Kaggle/OWID descargado manualmente:
python -m expotb.data --tb data/raw/tuberculosis.csv

# 2. Entrenar y evaluar los modelos
python -m expotb.train

# 2b. (Opcional) Comparar el conjunto base de 6 indicadores con el ampliado de 10
python -m expotb.compare

# 3. Levantar la API (documentación en http://localhost:8000/docs)
uvicorn api.main:app --reload

# 4. En otra terminal, levantar la interfaz (http://localhost:8501)
streamlit run app/Inicio.py
```

Si la API corre en otra dirección, define `EXPOTB_API_URL` antes de iniciar Streamlit.

### Pruebas

```bash
pytest
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado y modelos cargados |
| GET | `/models` | Lista de modelos (caja blanca, caja negra, ensamble) |
| GET | `/metrics` | Métricas de validación cruzada, prueba y prueba temporal |
| GET | `/dataset` | Dataset país-año procesado |
| POST | `/predict` | Incidencia estimada por cada modelo |
| POST | `/explain` | Contribución SHAP por indicador y por dimensión |
| GET | `/explain/global` | Importancia SHAP global |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pm25": 30, "clean_fuel": 60, "urban_pop": 50, "gdp_pc": 4000, "hiv": 1.0, "undernourishment": 10}'
```

## Metodología

- Preprocesamiento: se eliminan los registros sin incidencia o con más de 2 indicadores faltantes. Luego se aplica imputación por mediana y estandarización dentro de un pipeline, para que no haya fuga de información. La incidencia se transforma con `log1p` y se devuelve a su escala con `expm1`.
- Modelos de caja blanca: Ridge y Árbol de Decisión. Modelos de caja negra: Random Forest, XGBoost, Gradient Boosting y SVR. Ensamble por votación de RF, XGBoost y SVR.
- Evaluación: validación cruzada K=5 sobre el 80 %, prueba sobre el 20 % y prueba temporal (entrenamiento 2000-2015, prueba 2016-2019). Métricas: MAE, RMSE y R².
- Explicabilidad: SHAP (TreeExplainer) por indicador y por dimensión del exposoma.
- Comparación de conjuntos de variables (`expotb.compare`): el conjunto ampliado añade consumo de alcohol, saneamiento básico, gasto en salud y cobertura BCG. Se evalúa con los mismos registros y modelos bajo tres particiones: aleatoria, por país y temporal.
- Reproducibilidad: `random_state = 42`.

## Futuras implementaciones

- [ ] **Migrar la interfaz de Streamlit a React** (por ejemplo, con Next.js) para tener una interfaz más personalizable y de nivel producción. Se reutiliza la misma API FastAPI.
- [ ] Contenerizar la API y la interfaz con Docker y `docker-compose`.
- [ ] Registrar los experimentos y versionar los modelos con MLflow.
- [ ] Optimizar los hiperparámetros (Grid Search / búsqueda bayesiana).
- [ ] Desplegar en la nube (Streamlit Community Cloud / Hugging Face Spaces / Render).
- [ ] Agente que consuma la API para responder consultas en lenguaje natural (ejercicio de API).

## Fuentes de datos

- World Health Organization, *Global Tuberculosis Report*, vía Our World in Data: https://ourworldindata.org/grapher/incidence-of-tuberculosis-sdgs (Kaggle: https://www.kaggle.com/datasets/imtkaggleteam/tuberculosis)
- World Bank, *World Development Indicators*: https://databank.worldbank.org/source/world-development-indicators

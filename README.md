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
│                /predict · /predict/intervalo · /explain ...   │
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
│   ├── train.py              # optimización de hiperparámetros, tres protocolos, líneas base e intervalo
│   ├── tune.py               # búsqueda aleatoria de hiperparámetros y particiones temporales
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

# 2. Entrenar y evaluar los modelos (con búsqueda aleatoria de hiperparámetros; es lento)
python -m expotb.train --n-iter 15
#    sin optimización (unos 5 minutos):
python -m expotb.train --sin-ajuste
#    reutilizando los hiperparámetros de models/metrics.json y reajustando solo algunos modelos:
python -m expotb.train --reusar-params --reajustar ridge,arbol_decision --n-iter 60

# 2b. (Opcional) Comparar el conjunto base de 6 indicadores con el ampliado de 10
python -m expotb.compare

# 3. Levantar la API (documentación en http://localhost:8000/docs)
uvicorn api.main:app --reload

# 4. En otra terminal, levantar la interfaz (http://localhost:8501)
streamlit run app/Inicio.py
```

Si la API corre en otra dirección, define `EXPOTB_API_URL` antes de iniciar Streamlit.

### Asistente en Inicio

Para habilitar **Pregúntale a ExpoTB**, configura `DEEPSEEK_API_KEY` como
variable de entorno del proceso de Streamlit o en `.streamlit/secrets.toml`
y reinícialo. Opcionalmente,
`EXPOTB_CHAT_MODEL` permite cambiar el modelo de conversación (por defecto,
`deepseek-flash`). Mantén la clave fuera del repositorio.

El asistente envía los mensajes de la conversación a DeepSeek y utiliza
la API de ExpoTB para calcular estimaciones históricas y escenarios.

### Pruebas

```bash
pytest
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado y modelos cargados |
| GET | `/models` | Lista de modelos (caja blanca, caja negra, ensamble) |
| GET | `/metrics` | Métricas de validación cruzada, prueba, prueba por país y prueba temporal, líneas base, hiperparámetros elegidos e intervalo |
| GET | `/dataset` | Dataset país-año procesado |
| POST | `/predict` | Incidencia estimada por cada modelo |
| POST | `/predict/intervalo` | Estimación del ensamble con rango plausible del 80 % y su cobertura en países no vistos |
| POST | `/explain` | Contribución SHAP por indicador y por dimensión |
| GET | `/explain/global` | Importancia SHAP global |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"pm25": 30, "clean_fuel": 60, "urban_pop": 50, "gdp_pc": 4000, "hiv": 1.0, "undernourishment": 10}'
```

El campo opcional `income` (`LIC`, `LMC`, `UMC` o `HIC`) indica el grupo de ingreso del país y se usa para imputar indicadores faltantes; si se omite, se usa la mediana global.

## Metodología

- Preprocesamiento: se eliminan los registros sin incidencia o con más de 2 indicadores faltantes. Luego se aplica, dentro de un pipeline y solo con datos de entrenamiento (sin fuga de información), la imputación por la mediana del grupo de ingreso del Banco Mundial (los faltantes de VIH y desnutrición se concentran en países de ingreso alto, donde la mediana global los sobrestima) y la estandarización. La incidencia se transforma con `log1p` y se devuelve a su escala con `expm1`. Además, PM2.5, PIB per cápita, VIH y desnutrición (asimetría mayor a 1,5) se transforman con `log1p` antes de estandarizar (`LogSkew`); no tiene parámetros que aprender, así que no hay fuga de información.
- Hiperparámetros: búsqueda aleatoria (`RandomizedSearchCV`) por modelo, con RMSE sobre `log1p(incidencia)` como criterio y solo con los datos de entrenamiento de cada protocolo (validación cruzada K=5 aleatoria, `GroupKFold` por país y ventanas temporales crecientes). Los modelos servidos usan los hiperparámetros del protocolo por país. Los valores elegidos quedan en `models/metrics.json`. Como se optimiza en escala logarítmica, Ridge y el Árbol muestran un R² bajo en escala original aunque su R² logarítmico es razonable.
- Líneas base: mediana global y mediana por grupo de ingreso, evaluadas con los tres protocolos, para medir cuánto aporta el exposoma. También se reportan errores por nivel de incidencia y por grupo de ingreso.
- Intervalo: cuantiles del 80 % de los residuales logarítmicos del ensamble fuera de muestra por país. La cobertura real en países no vistos (aprox. 72 %) queda por debajo del 80 % nominal y se reporta tal cual.
- Modelos de caja blanca: Ridge y Árbol de Decisión. Modelos de caja negra: Random Forest, XGBoost, Gradient Boosting y SVR. Ensamble por votación de RF, XGBoost y SVR.
- Evaluación con tres protocolos. Aleatorio: validación cruzada K=5 sobre el 80 % y prueba sobre el 20 %. Por país: la partición separa países completos y la validación cruzada usa `GroupKFold` por código ISO3, de modo que mide la generalización a países no vistos. Temporal: entrenamiento 2000-2015 y prueba 2016-2019. El holdout temporal se repite en tres orígenes (entrena hasta 2007, 2011 y 2015 y prueba los 4 años siguientes). Métricas: MAE, RMSE y R², más RMSLE y R² en escala logarítmica, que no están dominadas por los países de incidencia muy alta.
- Explicabilidad: SHAP (TreeExplainer) por indicador y por dimensión del exposoma.
- Comparación (`expotb.compare`): conjunto base con imputación por grupo de ingreso, conjunto base con mediana global y conjunto ampliado (añade consumo de alcohol, saneamiento básico, gasto en salud y cobertura BCG). Se evalúan con los mismos registros y modelos bajo los tres protocolos.
- Reproducibilidad: `random_state = 42`.

## Futuras implementaciones

- [ ] **Migrar la interfaz de Streamlit a React** (por ejemplo, con Next.js) para tener una interfaz más personalizable y de nivel producción. Se reutiliza la misma API FastAPI.
- [ ] Contenerizar la API y la interfaz con Docker y `docker-compose`.
- [ ] Registrar los experimentos y versionar los modelos con MLflow.
- [ ] Probar búsqueda bayesiana de hiperparámetros (la búsqueda aleatoria ya está implementada).
- [ ] Añadir rezagos temporales y datos subnacionales.
- [ ] Desplegar en la nube (Streamlit Community Cloud / Hugging Face Spaces / Render).
- [ ] Agente que consuma la API para responder consultas en lenguaje natural (ejercicio de API).

## Fuentes de datos

- World Health Organization, *Global Tuberculosis Report*, vía Our World in Data: https://ourworldindata.org/grapher/incidence-of-tuberculosis-sdgs (Kaggle: https://www.kaggle.com/datasets/imtkaggleteam/tuberculosis)
- World Bank, *World Development Indicators*: https://databank.worldbank.org/source/world-development-indicators

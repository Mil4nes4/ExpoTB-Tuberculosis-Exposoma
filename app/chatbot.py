"""Asistente de Inicio. Las estimaciones se calculan en la API de ExpoTB."""

import json
import math
import os
import unicodedata

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from api_client import get_dataset, predict
from expotb import config

GUIDE = """Eres el asistente de ExpoTB. Responde brevemente en español.
Inicio resume los datos. Explorador muestra mapas y evolución por país.
Predicción y escenarios estima incidencia poblacional al cambiar indicadores.
Comparación contrasta métricas de los modelos. Explicabilidad muestra SHAP de XGBoost.
Para una cifra, un cambio de escenario o los valores usados, llama a consultar_expotb.
Para un país nuevo usa estimar. Para modificar el último escenario usa cambiar.
No inventes valores ni hagas pronósticos. Para consultas históricas conserva el año solicitado,
incluso si supera el periodo del dataset. Usa códigos ISO3 cuando conozcas el país.
Si falta información imprescindible, pregunta. No ofrezcas diagnósticos ni inferencias causales."""

TOOL = {
    "type": "function",
    "function": {
        "name": "consultar_expotb",
        "description": "Ejecuta una estimación histórica, modifica el último escenario o muestra sus entradas.",
        "parameters": {
            "type": "object",
            "properties": {
                "accion": {"type": "string", "enum": ["estimar", "cambiar", "mostrar_datos"]},
                "pais": {"type": "string", "description": "ISO3 o nombre del país para estimar"},
                "anio": {"type": "integer", "description": "Año histórico solicitado, si se indica"},
                "cambios": {
                    "type": "object",
                    "properties": {
                        name: {"type": "number", "description": config.FEATURE_LABELS[name]}
                        for name in config.FEATURES
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["accion"],
            "additionalProperties": False,
        },
    },
}


def normalize(value):
    return "".join(
        letter for letter in unicodedata.normalize("NFKD", value.strip().casefold())
        if not unicodedata.combining(letter)
    )


def historical_record(country, year):
    if not isinstance(country, str) or not country.strip():
        raise ValueError("¿Para qué país quieres la estimación?")
    if year is not None and type(year) is not int:
        raise ValueError("Indica el año como un número entero.")
    data = get_dataset()
    key = normalize(country)
    records = data[
        data["iso3"].map(normalize).eq(key) | data["country"].map(normalize).eq(key)
    ]
    if records.empty:
        raise ValueError("No encontré ese país en ExpoTB. Indica su nombre completo o código ISO3.")
    if records["iso3"].nunique() != 1:
        raise ValueError("Hay más de un país con ese nombre. Indica su código ISO3.")
    years = sorted(int(value) for value in records["year"].unique())
    if year is not None and year not in years:
        raise ValueError(
            f"No hay datos históricos para {country} en {year}. "
            f"ExpoTB cubre {int(data['year'].min())}–{int(data['year'].max())}. "
            f"Puedes consultar {years[-1]}, el último año disponible para ese país, "
            "o simular cambios sobre ese registro."
        )
    return records[records["year"].eq(year if year is not None else years[-1])].iloc[0]


def scenario_inputs(scenario):
    return "\n".join(
        f"- {config.FEATURE_LABELS[name]}: "
        + (f"{value:g}" if value is not None else "dato faltante, imputado por ExpoTB")
        for name, value in scenario["inputs"].items()
        if name in config.FEATURES
    )


def run_tool(action, previous):
    kind = action.get("accion")
    changes = action.get("cambios", {})
    if kind not in {"estimar", "cambiar", "mostrar_datos"} or not isinstance(changes, dict):
        raise ValueError("No pude interpretar la solicitud. Prueba indicando un país y año.")
    if kind in {"cambiar", "mostrar_datos"} and previous is None:
        raise ValueError("Primero pide una estimación para un país y año.")
    if kind == "mostrar_datos":
        return f"Indicadores para **{previous['country']}, {previous['year']}**\n\n" + scenario_inputs(previous), previous
    if kind == "estimar":
        row = historical_record(action.get("pais"), action.get("anio"))
        inputs = {
            name: None if pd.isna(row[name]) else float(row[name])
            for name in config.FEATURES
        }
        income = row.get(config.INCOME_COL)
        inputs[config.INCOME_COL] = income if income in ("LIC", "LMC", "UMC", "HIC") else None
        base = {"country": row["country"], "year": int(row["year"]), "inputs": inputs}
    else:
        if not changes:
            raise ValueError("¿Qué indicador quieres cambiar y a qué valor?")
        base = previous
    for name, value in changes.items():
        if name not in config.FEATURE_RANGES or type(value) not in (int, float):
            raise ValueError("Indica un valor numérico para un indicador de ExpoTB.")
        low, high, _ = config.FEATURE_RANGES[name]
        if not low <= value <= high:
            raise ValueError(f"{config.FEATURE_LABELS[name]} debe estar entre {low:g} y {high:g}.")
    inputs = {**base["inputs"], **changes}
    estimate = predict(inputs).get("ensamble")
    if estimate is None or not math.isfinite(estimate):
        raise ValueError("El ensamble no devolvió una estimación válida.")
    scenario = {**base, "inputs": inputs, "estimate": estimate}
    label = "Escenario simulado" if changes else "Estimación histórica"
    result = (
        f"**{label} · {base['country']}, {base['year']}**\n\n"
        f"El ensamble estima **{estimate:.1f} casos por cada 100 000 habitantes**."
    )
    if changes:
        initial = base["estimate"] if kind == "cambiar" else predict(base["inputs"]).get("ensamble")
        if initial is None or not math.isfinite(initial):
            raise ValueError("El ensamble no devolvió una estimación válida para el escenario de referencia.")
        result += (
            f" Referencia: **{initial:.1f}**. Diferencia: **{estimate - initial:+.1f}** "
            "casos por cada 100 000 habitantes. Es un cambio en la estimación del modelo."
        )
    return result + "\n\n" + scenario_inputs(scenario), scenario


def reply(messages, scenario, key):
    context = GUIDE + ("\nHay un escenario anterior." if scenario else "\nNo hay escenario anterior.")
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.getenv("EXPOTB_CHAT_MODEL") or "deepseek-flash",
                "messages": [{"role": "system", "content": context}, *messages[-12:]],
                "tools": [TOOL],
                "thinking": {"type": "disabled"},
                "max_tokens": 600,
            },
            timeout=25,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return "El asistente no pudo conectarse con DeepSeek. Inténtalo de nuevo en un momento.", scenario
    calls = message.get("tool_calls") or []
    if not calls:
        return message.get("content") or "No obtuve una respuesta. Intenta reformular la pregunta.", scenario
    try:
        function = calls[0]["function"]
        action = json.loads(function["arguments"])
        if len(calls) != 1 or function["name"] != "consultar_expotb" or not isinstance(action, dict):
            return "Pide una estimación o un cambio de escenario por mensaje.", scenario
    except (ValueError, KeyError, TypeError):
        return "No pude interpretar la consulta. Prueba indicando un país y año.", scenario
    try:
        return run_tool(action, scenario)
    except requests.RequestException:
        return "La API de ExpoTB no está disponible para calcular la estimación. Inténtalo de nuevo.", scenario
    except ValueError as error:
        return str(error), scenario


@st.dialog("Pregúntale a ExpoTB", width="medium")
def chat_dialog():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY")
        except StreamlitSecretNotFoundError:
            key = None
    if not key:
        st.info("El asistente estará disponible cuando se configure su acceso a DeepSeek.")
        return
    messages = st.session_state.setdefault("expotb_chat_messages", [])
    if st.button("Nueva conversación"):
        messages.clear()
        st.session_state.pop("expotb_chat_scenario", None)
    if not messages:
        st.caption("Prueba «¿Cómo uso el explorador?» o «Estima la incidencia de Perú en 2019».")
    history = st.container(height=320)
    for message in messages:
        with history.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Escribe tu pregunta sobre ExpoTB", max_chars=1200)
    if prompt and prompt.strip():
        messages.append({"role": "user", "content": prompt.strip()})
        with history.chat_message("user"):
            st.markdown(prompt.strip())
        with history.chat_message("assistant"):
            with st.spinner("Consultando…"):
                answer, scenario = reply(messages, st.session_state.get("expotb_chat_scenario"), key)
            st.markdown(answer)
        messages.append({"role": "assistant", "content": answer})
        st.session_state["expotb_chat_scenario"] = scenario


def render_chat():
    st.html("""
    <style>
    .st-key-expotb_chat_launcher {
        position: fixed;
        right: max(1rem, env(safe-area-inset-right));
        bottom: max(1rem, env(safe-area-inset-bottom));
        width: max-content !important;
        max-width: calc(100vw - 2rem);
        z-index: 99;
    }
    .st-key-expotb_chat_launcher button {
        min-height: 44px;
        border-radius: 999px;
        padding: .55rem 1rem;
        box-shadow: 0 4px 18px #0003;
    }
    </style>
    """)
    with st.container(key="expotb_chat_launcher"):
        if st.button("Pregúntale a ExpoTB", icon=":material/chat:", type="primary"):
            chat_dialog()

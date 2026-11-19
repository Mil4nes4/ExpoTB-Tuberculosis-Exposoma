import json
from html import escape
from math import floor, log10
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import get_dataset, require_api
from expotb import config
from ui import PRIMARY, kpi_row, note, page_header, style_fig
from ui import csv_download, module_route, quick_guide

VARIABLES = {config.TARGET: "Incidencia de TB", **config.FEATURE_LABELS}
UNITS = {
    config.TARGET: "casos por 100 000 hab.", "pm25": "µg/m³", "clean_fuel": "%",
    "urban_pop": "%", "gdp_pc": "US$ corrientes", "hiv": "%", "undernourishment": "%",
}
VARIABLE_HELP = {
    config.TARGET: "Casos nuevos estimados de tuberculosis durante el año, por cada 100 000 habitantes.",
    "pm25": "Concentración media anual de partículas finas en el aire.",
    "clean_fuel": "Porcentaje de habitantes con acceso a combustibles y tecnologías limpias para cocinar.",
    "urban_pop": "Porcentaje de habitantes que vive en zonas urbanas según la definición de cada país.",
    "gdp_pc": "Producto interno bruto por habitante, expresado en dólares corrientes.",
    "hiv": "Porcentaje de personas de 15 a 49 años que viven con VIH.",
    "undernourishment": "Porcentaje de habitantes cuya ingesta habitual de alimentos es insuficiente.",
}
INCOME = {"LIC": "Bajo", "LMC": "Mediano bajo", "UMC": "Mediano alto", "HIC": "Alto"}
LEVEL_COLORS = ("#A9D2F0", "#6FB1E6", "#3F8FD6", "#2368B8", "#154587", "#0B2A63")
# descenso en cian, aumento en índigo (de menor a mayor cambio)
CHANGE_COLORS = ("#0E7490", "#67B7CF", "#C9E6EF", "#F5F7FA", "#C9C6F2", "#7F78D8", "#3B2FA0")
MISSING_COLOR = "#D3DADF"
WATER_COLOR = "#F0F5F7"

# Agrupaciones geográficas de la División de Estadística de las Naciones Unidas (M49).
REGION_ISO = {
    "América": """ABW AIA ARG ATG BES BHS BLM BLZ BMU BOL BRA BRB BVT CAN CHL COL CRI CUB CUW CYM
        DMA DOM ECU FLK GLP GRD GRL GTM GUF GUY HND HTI JAM KNA LCA MAF MEX MSR MTQ NIC PAN PER
        PRI PRY SGS SLV SPM SUR SXM TCA TTO URY USA VCT VEN VGB VIR""",
    "Europa": """ALA ALB AND AUT BEL BGR BIH BLR CHE CZE DEU DNK ESP EST FIN FRA FRO GBR GGY GIB GRC
        HRV HUN IMN IRL ISL ITA JEY LIE LTU LUX LVA MCO MDA MKD MLT MNE NLD NOR POL PRT ROU RUS
        SJM SMR SRB SVK SVN SWE UKR VAT""",
    "África": """AGO ATF BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI DZA EGY ERI ESH ETH GAB GHA
        GIN GMB GNB GNQ IOT KEN LBR LBY LSO MAR MDG MLI MOZ MRT MUS MWI MYT NAM NER NGA REU RWA
        SDN SEN SHN SLE SOM SSD STP SWZ SYC TCD TGO TUN TZA UGA ZAF ZMB ZWE""",
    "Asia": """AFG ARE ARM AZE BGD BHR BRN BTN CHN CYP GEO HKG IDN IND IRN IRQ ISR JOR JPN KAZ KGZ
        KHM KOR KWT LAO LBN LKA MAC MDV MMR MNG MYS NPL OMN PAK PHL PRK PSE QAT SAU SGP SYR
        THA TJK TKM TLS TUR UZB VNM YEM""",
    "Oceanía": """ASM AUS CCK COK CXR FJI FSM GUM HMD KIR MHL MNP NCL NFK NIU NRU NZL PCN PLW PNG
        PYF SLB TKL TON TUV UMI VUT WLF WSM""",
}
REGION_VIEWS = {
    "Mundo": (0, 12, 1.03), "América": (-86, 13, 1.48), "Europa": (25, 54, 2.75),
    "África": (20, 2, 2.33), "Asia": (95, 38, 1.77), "Oceanía": (158, -16, 2.12),
}
# vista (longitud, latitud, escala) de cada país para acercar el mapa; derivada del centro y la superficie
# del país (https://github.com/mledoze/countries)
COUNTRY_VIEWS = json.loads((Path(__file__).resolve().parents[1] / "country_views.json").read_text(encoding="utf-8"))
MAP_KEY = "explorador_mapa"
YEAR_KEY = "explorador_anio"
COUNTRY_KEY = "explorador_pais_clic"
MAP_STYLE = """
<style>
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-explorador_lienzo"]) {
  --expo-map-height: clamp(260px, 50vw, 480px);
  height: var(--expo-map-height);
  flex-basis: var(--expo-map-height);
}
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-explorador_lienzo_animado"]) {
  --expo-map-height: clamp(300px, 50vw, 480px);
}
@keyframes expo-acercar {
  from { opacity: .3; transform: scale(1.05); }
  to { opacity: 1; transform: none; }
}
[data-testid="stVerticalBlock"][class*="st-key-explorador_lienzo"] {
  animation: expo-acercar .65s cubic-bezier(.2, .7, .2, 1) both;
  padding: 0;
  border: 1px solid var(--expo-line);
  border-radius: var(--expo-radius);
  overflow: hidden;
}
.expo-map-legend {
  display: flex;
  flex-wrap: wrap;
  gap: .45rem 1rem;
  list-style: none;
  margin: 0;
  padding: 0;
  color: var(--expo-text);
  font-size: .875rem;
  font-variant-numeric: tabular-nums;
}
.expo-map-legend li { display: inline-flex; align-items: center; gap: .4rem; }
.expo-map-legend .expo-map-unit { font-weight: 600; }
.expo-map-legend span {
  display: inline-block;
  width: 1.15rem;
  height: .75rem;
  border: 1px solid #83969A55;
  border-radius: 2px;
}
</style>
"""


def number(value, digits=1, signed=False):
    if pd.isna(value):
        return "Sin datos"
    sign = "+" if signed and value > 0 else ""
    return sign + f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def with_unit(value, unit, signed=False):
    if pd.isna(value):
        return "Sin datos"
    return f"{number(value, signed=signed)} {unit}"


@st.cache_data(show_spinner=False)
def prepare_map(data, variable, show_change):
    """Se calculan las posiciones y los cortes en el conjunto mundial de años."""
    data = data.rename(columns={variable: "value"}).copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.drop_duplicates(["iso3", "year"]).sort_values(["iso3", "year"])
    first = data.dropna(subset=["value"]).drop_duplicates("iso3").set_index("iso3")
    data["base_year"] = data["iso3"].map(first["year"])
    data["change"] = data["value"] - data["iso3"].map(first["value"])
    earlier = data.set_index(["iso3", "year"])["value"].reindex(
        pd.MultiIndex.from_arrays([data["iso3"], data["year"] - 5])
    ).to_numpy()
    data["delta5"] = data["value"] - earlier
    data["pct5"] = data["delta5"] / np.where(earlier == 0, np.nan, earlier) * 100
    data["shown"] = data["change"] if show_change else data["value"]
    by_year = data.groupby("year")["shown"]
    data["position"] = by_year.rank(ascending=False, method="min")
    data["total"] = by_year.transform("count")

    # Se redondea el ancho a una magnitud legible y se dejan abiertas las colas.
    reference = data["shown"].abs().quantile(.85)
    raw_step = reference / (2.5 if show_change else 5)
    if pd.isna(raw_step) or raw_step <= 0:
        raw_step = 1
    power = 10.0 ** floor(log10(raw_step))
    step = next(factor * power for factor in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if factor * power >= raw_step)
    cuts = step * (np.arange(-2.5, 3) if show_change else np.arange(1, 6))
    data["class"] = np.searchsorted(cuts, data["shown"], side="right")
    data.loc[data["shown"].isna(), "class"] = len(cuts) + 1
    return data, cuts


def legend_labels(cuts, show_change):
    digits = max(0, min(8, 2 - floor(log10(abs(cuts[0])))))
    labels = [number(cut, digits, signed=show_change).rstrip("0").rstrip(",") if digits else
              number(cut, 0, signed=show_change) for cut in cuts]
    first = f"Menos de {labels[0]}" if show_change else f"0 a < {labels[0]}"
    return [first, *(f"{a} a < {b}" for a, b in zip(labels, labels[1:])), f"{labels[-1]} o más", "Sin datos"]


def position_label(row):
    return f"N.º {int(row.position)} de {int(row.total)}" if pd.notna(row.position) else "Sin posición"


def map_rows(data, registry, year):
    columns = ["iso3", "income", "value", "shown", "base_year", "delta5", "pct5", "position", "total", "class"]
    return registry.merge(data.loc[data["year"].eq(year), columns], on="iso3", how="left")


def map_payload(rows, year, unit, show_change, missing_class):
    details = []
    delta_unit = "p. p." if unit == "%" else unit
    for row in rows.itertuples(index=False):
        current = with_unit(row.value, unit)
        change = ""
        if show_change and pd.notna(row.shown):
            change = f"<br>Desde {int(row.base_year)}: {with_unit(row.shown, delta_unit, signed=True)}"
        change5 = ""
        if pd.notna(row.delta5):
            percent = f" ({number(row.pct5, signed=True)} %)" if pd.notna(row.pct5) else ""
            change5 = f"<br>Frente a {year - 5}: {with_unit(row.delta5, delta_unit, signed=True)}{percent}"
        details.append(
            f"<b>{escape(row.country)}</b> · {year}<br>{current}{change}<br>"
            f"Ingreso: {INCOME.get(row.income, 'Sin datos')}<br>{position_label(row)}{change5}"
        )
    return {
        "z": rows["class"].fillna(missing_class).astype(int).tolist(),
        "hovertext": details,
        "customdata": [[code, year] for code in rows["iso3"]],
    }


def view_map(data, registry, rows, year, years, region, unit, colors, highlight, show_change, animate):
    n = len(colors)
    scale = [(edge / n, color) for i, color in enumerate(colors) for edge in (i, i + 1)]
    borders = [code == highlight for code in registry["iso3"]]
    trace = go.Choropleth(
        locations=registry["iso3"].tolist(), locationmode="ISO-3",
        **map_payload(rows, year, unit, show_change, n - 1),
        zmin=-.5, zmax=n - .5, colorscale=scale, showscale=False,
        marker_line_color=["#0B1F3A" if marked else "#FFFFFF" for marked in borders],
        marker_line_width=[2.4 if marked else .55 for marked in borders],
        unselected_marker_opacity=1, selected_marker_opacity=1,
        hovertemplate="%{hovertext}<extra></extra>",
    )
    fig = style_fig(go.Figure(trace), 480)
    lon, lat, zoom = COUNTRY_VIEWS.get(highlight) or REGION_VIEWS[region]
    fig.update_geos(
        projection=dict(type="natural earth", scale=zoom, rotation=dict(lon=lon)),
        center=dict(lon=lon, lat=lat), resolution=50,
        showframe=False, bgcolor=WATER_COLOR, showocean=True, oceancolor=WATER_COLOR,
        showlakes=True, lakecolor=WATER_COLOR, showland=True, landcolor=MISSING_COLOR,
        showcountries=True, countrycolor="#FFFFFF", countrywidth=.6,
        showcoastlines=True, coastlinecolor="#9EADB3", coastlinewidth=.5,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=WATER_COLOR, clickmode="event+select",
        dragmode="pan", hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CBD7D9", align="left"),
        separators=", ", transition=dict(duration=0), uirevision=f"{region}-{highlight}",
    )
    if animate:
        fig.frames = [
            go.Frame(name=str(frame_year), traces=[0], data=[go.Choropleth(**map_payload(
                map_rows(data, registry, frame_year), frame_year, unit, show_change, n - 1,
            ))])
            for frame_year in years
        ]
        timing = dict(frame=dict(duration=480, redraw=True), transition=dict(duration=0), mode="immediate")
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=116),
            updatemenus=[dict(
                type="buttons", direction="right", showactive=False, x=.02, y=0,
                xanchor="left", yanchor="top", pad=dict(t=11),
                bgcolor="#FFFFFF", bordercolor="#D5E5E3",
                buttons=[
                    dict(label="Reproducir", method="animate", args=[None, dict(**timing, fromcurrent=True)]),
                    dict(label="Pausar", method="animate", args=[[None], dict(
                        mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                    )]),
                ],
            )],
            sliders=[dict(
                x=.02, len=.96, y=0, yanchor="top", active=years.index(year), pad=dict(t=34, b=0),
                currentvalue=dict(prefix="Año en el mapa: ", xanchor="right", offset=4),
                activebgcolor="#CFE3F7", bgcolor="#FFFFFF", bordercolor="#D5E5E3",
                steps=[
                    dict(method="animate", label=str(y), value=str(y),
                         args=[[str(y)], dict(**timing, fromcurrent=False)])
                    for y in years
                ],
            )],
        )
    return fig


def select_map_country():
    points = st.session_state[MAP_KEY]["selection"]["points"]
    if points and (location := points[-1].get("customdata")):
        st.session_state[COUNTRY_KEY] = location[0]
        st.session_state[YEAR_KEY] = int(location[1])
    else:
        st.session_state.pop(COUNTRY_KEY, None)


def country_profile(dataset, rows, code, variable, year, show_change):
    if code is None:
        st.info("Seleccione un país en el mapa o en el buscador para consultar su ficha.",
                icon=":material/touch_app:")
        return

    row = rows.set_index("iso3").loc[code]
    unit = UNITS[variable]
    delta_unit = "p. p." if unit == "%" else unit
    st.markdown(f"### {escape(row.country)}")
    st.caption(f"{year} · Posición mundial: {position_label(row)} · Ingreso: {INCOME.get(row.income, 'Sin datos')}")
    st.markdown(f"**{with_unit(row.value, unit)}**")
    if show_change and pd.notna(row.shown):
        st.caption(f"Desde {int(row.base_year)}: {with_unit(row.shown, delta_unit, signed=True)}")
    if pd.notna(row.delta5):
        percent = f" ({number(row.pct5, signed=True)} %)" if pd.notna(row.pct5) else ""
        st.caption(f"Frente a {year - 5}: {with_unit(row.delta5, delta_unit, signed=True)}{percent}")

    history = dataset.loc[dataset["iso3"].eq(code)].sort_values("year")
    median = dataset.groupby("year")[variable].median()
    timeline = history.set_index("year")[variable].reindex(median.index)
    trend = go.Figure()
    trend.add_scatter(x=median.index, y=median, name="Mediana mundial", mode="lines",
                      line=dict(color="#738890", dash="dot", width=2))
    trend.add_scatter(x=timeline.index, y=timeline, name=row.country, mode="lines",
                      line=dict(color=PRIMARY, width=2.5))
    trend.add_vline(x=year, line_color="#A1B1B6", line_width=1)
    if pd.notna(row.value):
        trend.add_scatter(x=[year], y=[row.value], mode="markers", showlegend=False, hoverinfo="skip",
                          marker=dict(color=PRIMARY, size=8))
    style_fig(trend, 190).update_layout(
        margin=dict(l=4, r=8, t=34, b=4), hovermode="x unified", separators=", ",
        xaxis=dict(tickformat="d", nticks=5), yaxis=dict(automargin=True, gridcolor="#E7EEEE"),
    )
    st.plotly_chart(trend, width="stretch", theme=None, config={"displayModeBar": False})

    st.markdown("**Indicadores del exposoma**")
    yearly = dataset.loc[dataset["year"].eq(year)].set_index("iso3")
    medians = yearly[config.FEATURES].median()
    country = yearly[config.FEATURES].reindex([code]).iloc[0]
    comparison = pd.DataFrame({
        "Indicador": [config.FEATURE_LABELS[f] for f in config.FEATURES],
        "País": [number(country[f]) for f in config.FEATURES],
        "Mediana mundial": [number(medians[f]) for f in config.FEATURES],
    })
    st.dataframe(comparison, hide_index=True, width="stretch")


page_header("Explorador del exposoma", "Mapa mundial, ranking y evolución de cada indicador por país.", "public")
require_api()
module_route(1)
quick_guide(
    ("Elegir una dimensión y una variable.",
     "Ajustar el año o la zona y seleccionar un país para abrir su ficha.",
     "Usar Evolución para comparar series o Datos para descargar el conjunto completo."),
    (("Exposoma", "Conjunto de exposiciones a lo largo de la vida."),
     ("Incidencia", "Casos nuevos estimados por cada 100 000 habitantes durante un año.")),
)

df = get_dataset()
variables = {**VARIABLES, config.TARGET: "Incidencia de TB (por 100 000 hab.)"}
years = sorted(int(y) for y in df["year"].dropna().unique())
dimensions = {"Tuberculosis": [config.TARGET], **config.DIMENSIONS}
saved = st.session_state.get("filtros_explorador", {})
saved_dimension = saved.get("dimension", "Tuberculosis")

with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    dimension = c1.selectbox("Dimensión", list(dimensions), index=list(dimensions).index(saved_dimension),
                             key="explorador_dimension", help="La incidencia de TB es el resultado estudiado. "
                             "Los factores del exposoma se agrupan en cuatro dimensiones.")
    var = c2.selectbox("Variable", dimensions[dimension], format_func=variables.get,
                       key=f"explorador_variable_{dimension}",
                       index=dimensions[dimension].index(saved["variable"]) if saved.get("variable") in dimensions[dimension] else 0,
                       help="El mapa, el ranking y la pestaña Evolución usan la variable seleccionada.")

tab_map, tab_line, tab_data = st.tabs([":material/map: Mapa y ranking", ":material/show_chart: Evolución",
                                       ":material/table: Datos"])

with tab_map:
    st.html(MAP_STYLE)
    unit = UNITS[var]
    a, b, c = st.columns([2, 1, 2])
    view = a.selectbox("Vista", ["Valor", "Cambio desde el primer año"],
                       help="El cambio se mide respecto del primer dato disponible de cada país.")
    show_change = view != "Valor"
    map_unit = "puntos porcentuales" if show_change and unit == "%" else unit
    saved_year = saved.get("year")
    year = b.select_slider("Año", years, value=saved_year if saved_year in years else years[-1], key=YEAR_KEY,
                           help="Año de referencia del mapa, el ranking y la ficha.") if len(years) > 1 else years[0]
    if len(years) == 1:
        b.caption(f"Año disponible: {year}")
    registry = df[["iso3", "country"]].dropna().drop_duplicates("iso3").sort_values("country")
    names = registry.set_index("iso3")["country"].to_dict()
    marked = c.selectbox("Resaltar país", list(names), key="explorador_resaltado", format_func=names.get,
                         index=list(names).index(saved["marked"]) if saved.get("marked") in names else None,
                         placeholder="Buscar país", help="Se contornea en el mapa y se marca en el ranking regional.",
                         on_change=st.session_state.pop, args=(COUNTRY_KEY, None))
    region = st.segmented_control("Zona", list(REGION_VIEWS), default=saved.get("region", "Mundo"),
                                  key="explorador_zona", selection_mode="single",
                                  help="Acerca el mapa y limita el ranking a la zona elegida.") or "Mundo"
    animate = st.toggle("Habilitar reproducción", disabled=len(years) < 2,
                        help="Carga la serie temporal para reproducirla en el mapa.")
    if animate:
        st.caption("La reproducción es una vista previa. Al seleccionar un país en un fotograma, "
                   "se actualizan el año, la ficha y el ranking.")

    prepared, cuts = prepare_map(df[["iso3", "year", "income", var]], var, show_change)
    rows = map_rows(prepared, registry, year)
    colors = (*(CHANGE_COLORS if show_change else LEVEL_COLORS), MISSING_COLOR)
    country = st.session_state.get(COUNTRY_KEY)
    highlight = marked or country
    st.session_state["filtros_explorador"] = {
        "dimension": dimension, "variable": var, "year": year, "region": region, "marked": marked,
        "country": names.get(country or marked),
    }

    with st.container(border=True, key="explorador_marco"):
        st.markdown(f"### {VARIABLES[var]} · {year}")
        st.caption(f"{region} · {view} · {map_unit}. {VARIABLE_HELP[var]}")
        # la clave cambia con el país: al montarse de nuevo el lienzo se reproduce la transición de acercamiento
        canvas_key = ("explorador_lienzo_animado" if animate else "explorador_lienzo") + (f"_{highlight}" if highlight else "")
        with st.container(height=480, key=canvas_key):
            st.plotly_chart(
                view_map(prepared, registry, rows, year, years, region, unit, colors, highlight, show_change, animate),
                key=MAP_KEY, on_select=select_map_country, selection_mode="points", theme=None,
                height="stretch", width="stretch", config={
                    "displaylogo": False, "scrollZoom": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                    "toImageButtonOptions": {"filename": "expotb_mapa", "format": "png"},
                },
            )
        chips = "".join(
            f'<li><span aria-hidden="true" style="background:{color}"></span>{escape(label)}</li>'
            for color, label in zip(colors, legend_labels(cuts, show_change))
        )
        st.html(
            f'<ul class="expo-map-legend" aria-label="Leyenda del mapa">'
            f'<li class="expo-map-unit">{escape(map_unit)}</li>{chips}</ul>'
        )
        st.caption("Fuentes: OMS vía Our World in Data (incidencia de TB) y Banco Mundial (exposoma e ingreso).")

    with st.expander("Cómo leer este mapa", icon=":material/info:"):
        st.markdown(
            "Cada color representa un intervalo de la variable. En **Valor**, los tonos más oscuros indican "
            "valores más altos. En **Cambio desde el primer año**, el cian indica una reducción, el índigo "
            "un aumento y el tono claro del centro un cambio cercano a cero. El año de referencia es el "
            "primero con un dato disponible en cada país.\n\n"
            "Los intervalos interiores tienen el mismo ancho. Los extremos agrupan los valores de las colas. "
            "El ancho se obtiene del percentil 85 de las magnitudes de todos los años y se redondea "
            "para facilitar la lectura. Los cortes se conservan al cambiar de zona. "
            "Son intervalos de visualización. El **gris** indica que falta el dato necesario para la vista.\n\n"
            "La incidencia de tuberculosis expresa los casos nuevos estimados en un año por cada 100 000 "
            "habitantes. Las estimaciones proceden de la OMS y se distribuyen a través de Our World in Data. "
            "Los indicadores del exposoma y los grupos de ingreso proceden del Banco Mundial.\n\n"
            "La posición del mapa y de la ficha se calcula entre todos los países con datos para el año y la "
            "vista elegidos. El ranking inferior se limita a la zona seleccionada."
        )

    available = rows.dropna(subset=["shown"])
    if not available.empty:
        highest = available.loc[available["shown"].idxmax()]
        lowest = available.loc[available["shown"].idxmin()]
        kpi_row([
            ("Promedio mundial", number(available["shown"].mean()), f"{len(available)} países con datos"),
            ("Mediana mundial", number(available["shown"].median()), "Valor central"),
            ("Más alto", number(highest.shown), escape(highest.country)),
            ("Más bajo", number(lowest.shown), escape(lowest.country)),
        ])

    rank_col, profile_col = st.columns([1, 1], gap="medium")
    with rank_col.container(border=True):
        st.subheader(f"Ranking · {region}")
        st.caption(f"{year} · {view} · {map_unit}")
        side = st.radio("Orden", ["Valores más altos", "Valores más bajos"], horizontal=True,
                        help="Elige qué extremo de la distribución se muestra primero.")
        eligible = available if region == "Mundo" else available[available["iso3"].isin(REGION_ISO[region].split())]
        if eligible.empty:
            st.info("No hay valores disponibles en esta zona para el año seleccionado.", icon=":material/info:")
        else:
            ascending = side == "Valores más bajos"
            ordered = eligible.sort_values(["shown", "country"], ascending=[ascending, True])
            top = ordered.loc[ordered.index.isin(ordered.head(10).index) | ordered["iso3"].eq(highlight)].iloc[::-1]
            flags = top["iso3"].eq(highlight)
            bar = go.Figure(go.Bar(
                x=top["shown"], y=top["country"], orientation="h",
                marker=dict(color=[colors[int(i)] for i in top["class"]],
                            line=dict(color="#0B1F3A", width=[2 if flag else 0 for flag in flags])),
                hovertext=[with_unit(v, map_unit, signed=show_change) for v in top["shown"]],
                text=[number(v, signed=show_change) if mark or v == 0 else "" for v, mark in zip(top["shown"], flags)],
                textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>%{hovertext}<extra></extra>",
            ))
            style_fig(bar, 365).update_layout(
                margin=dict(l=4, r=28, t=8, b=8), separators=", ",
                xaxis=dict(gridcolor="#E3EBEC", zerolinecolor="#A1B1B6", tickfont=dict(size=11), nticks=5),
                yaxis=dict(automargin=True, tickfont=dict(size=12)),
            )
            st.plotly_chart(bar, width="stretch", theme=None, config={"displayModeBar": False})
            csv_download(top.iloc[::-1][["country", "shown", "position"]].rename(columns={
                "country": "País", "shown": f"Valor mostrado ({map_unit})", "position": "Posición mundial"
            }), "expotb_ranking.csv", "Descargar ranking (CSV)")
            if highlight and highlight not in eligible["iso3"].values:
                st.caption("El país resaltado queda fuera de este ranking.")

    with profile_col.container(border=True):
        st.subheader("Ficha del país")
        country_profile(df, rows, country or marked, var, year, show_change)

with tab_line:
    countries = sorted(df["country"].unique())
    default = [c for c in ("Peru", "India", "South Africa") if c in countries]
    selected = st.multiselect("Países", countries, default=default, placeholder="Elige uno o más países",
                              help="Se muestra una serie temporal por cada país seleccionado.")
    show_median = st.checkbox("Mostrar la mediana mundial", value=True,
                              help="Añade la mediana de los países con datos para cada año.")
    if selected:
        line = px.line(df[df["country"].isin(selected)], x="year", y=var, color="country", markers=True,
                       labels={var: variables[var], "year": "Año", "country": "País"})
        if show_median:
            med = df.groupby("year")[var].median().reset_index()
            line.add_scatter(x=med["year"], y=med[var], mode="lines", name="Mediana mundial",
                             line=dict(color="#94A3B8", dash="dash"))
        st.plotly_chart(style_fig(line, 430), width="stretch")
    else:
        note("Elige al menos un país para ver su evolución.")

with tab_data:
    st.dataframe(df, width="stretch", height=420)
    st.download_button("Descargar datos (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="expotb_dataset.csv", mime="text/csv", icon=":material/download:",
                       on_click=st.toast, args=("Descarga solicitada.",), kwargs={"icon": ":material/download:"})

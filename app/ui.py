"""Estilo y componentes visuales compartidos por las páginas de la interfaz."""
import streamlit as st

from expotb import config

# color e ícono de cada dimensión del exposoma
DIM_COLORS = {"Externa": "#0EA5E9", "Hogar": "#F59E0B", "Social": "#8B5CF6", "Biológica": "#EF4444"}
DIM_ICONS = {"Externa": "air", "Hogar": "home", "Social": "groups", "Biológica": "biotech"}
FEATURE_DIM = {f: dim for dim, feats in config.DIMENSIONS.items() for f in feats}
PRIMARY = "#0F766E"
ACCENT = "#F97316"

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..500&display=swap");
:root {
  --expo-sans: "Atkinson Hyperlegible", "Source Sans", system-ui, sans-serif;
  --expo-serif: "Newsreader", Georgia, serif;
  --expo-accent: #0F766E;
  --expo-dark: #0B3B3A;
  --expo-text: #0F2A2A;
  --expo-muted: #4B6A69;
  --expo-line: #D5E5E3;
  --expo-orange: #E8822A;
  --expo-orange-soft: #FFF3E6;
  --expo-panel: #FFFFFF;
  --expo-radius: .375rem;
  --expo-gap: .875rem;
}
@keyframes xfade { from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;} }
[data-testid="stMainBlockContainer"] {
  padding-block: clamp(3.5rem, 6vw, 4.25rem) 3rem;
  padding-inline: clamp(1rem, 4vw, 3rem);
  max-width: 84rem;
  animation: xfade .4s ease both;
}

/* tipografía: texto en Atkinson Hyperlegible y títulos en Newsreader (los íconos conservan su fuente) */
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp [data-baseweb], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
  font-family: var(--expo-sans);
}
.stApp h1, .stApp h2, .stApp h3, .xhero h1, .xhead h1, .xcard .xvalue, [data-testid="stMetricValue"],
.xbrand b {
  font-family: var(--expo-serif);
  font-weight: 500;
  letter-spacing: -.01em;
}
.xcard h3 { font-family: var(--expo-sans); font-weight: 700; }

/* portada y encabezados de página */
.xhero {
  background: linear-gradient(110deg, var(--expo-dark) 0%, var(--expo-accent) 100%);
  color: #fff; border-radius: var(--expo-radius); padding: 1.8rem 2rem; margin-bottom: 1.2rem;
  border-bottom: 4px solid var(--expo-orange);
}
.xhero .xtag { display: inline-block; background: rgba(255,255,255,.14); border-radius: var(--expo-radius);
  padding: .15rem .6rem; font-size: .78rem; margin-bottom: .6rem; letter-spacing: .04em; }
.xhero h1 { color: #fff; margin: 0 0 .35rem; padding: 0; font-size: clamp(1.65rem, 3.5vw, 2.25rem);
  line-height: 1.15; font-weight: 650; letter-spacing: -.015em; overflow-wrap: anywhere; }
.xhero p { color: rgba(255,255,255,.9); margin: 0; max-width: 70ch; line-height: 1.5; }
.xhead { display: flex; align-items: center; gap: .8rem; padding-bottom: var(--expo-gap);
  border-bottom: 1px solid var(--expo-line); margin-bottom: .6rem; }
.xhead .xico { width: 42px; height: 42px; border-radius: var(--expo-radius); display: grid; place-items: center;
  background: var(--expo-orange); color: #fff; flex: none; }
.xhead h1 { margin: 0; padding: 0; color: var(--expo-text); font-size: clamp(1.5rem, 3vw, 1.9rem);
  line-height: 1.15; font-weight: 650; letter-spacing: -.015em; overflow-wrap: anywhere; }
.xhead p { margin: .15rem 0 0; color: var(--expo-muted); line-height: 1.45; max-width: 70ch; }

/* íconos Material (la fuente la registra Streamlit) */
.xi { font-family: "Material Symbols Rounded"; font-style: normal; font-weight: 400;
  font-feature-settings: "liga"; line-height: 1; letter-spacing: normal; display: inline-block;
  white-space: nowrap; direction: ltr; flex: none; width: 1em; overflow: hidden; }

/* tarjetas */
.xgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 10rem), 1fr));
  gap: var(--expo-gap); grid-auto-rows: 1fr; margin-bottom: .6rem; }
.xcard, [data-testid="stMetric"] { background: var(--expo-panel); border: 1px solid var(--expo-line);
  border-radius: var(--expo-radius); padding: .75rem .875rem; min-width: 0; overflow-wrap: anywhere;
  transition: border-color .2s ease, transform .2s ease; }
.xcard:hover, [data-testid="stMetric"]:hover { border-color: #F5AB62; transform: translateY(-1px); }
.xcard.xtop { border-top: 3px solid var(--expo-orange); }
.xcard .xlabel, .xcard .xsub, .xcard p { color: var(--expo-muted); font-size: .9rem; line-height: 1.4; }
.xcard .xlabel { font-weight: 600; }
.xcard .xvalue, [data-testid="stMetricValue"] { color: var(--expo-text); font-size: 1.75rem; font-weight: 650;
  font-variant-numeric: tabular-nums; line-height: 1.2; }
.xcard .xvalue { margin: .2rem 0; }
.xcard h3 { margin: .45rem 0 .2rem; padding: 0; font-size: 1rem; }
.xcard p { margin: 0; }
.xcard .xbadge { width: 34px; height: 34px; border-radius: var(--expo-radius); display: grid; place-items: center;
  color: #fff; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--expo-line); border-radius: var(--expo-radius); }
[data-testid="stMainBlockContainer"] h2 { font-size: 1.3rem; line-height: 1.25; font-weight: 650; padding: .65rem 0 .125rem; }

.xnote { border-inline-start: 3px solid var(--expo-orange); background: var(--expo-orange-soft); border-radius: var(--expo-radius);
  padding: .75rem 1rem; color: var(--expo-text); font-size: .95rem; margin: .3rem 0 .8rem; }
.xnote.warn { border-inline-start-color: var(--expo-accent); background: #FDE3C8; border-inline-start-color: #C25E10; }

/* controles y transiciones */
div[data-testid="stTabs"] button { transition: color .2s ease, background .2s ease; font-weight: 600; }
div[data-testid="stTabs"] button:hover { background: rgba(232,130,42,.10); }
.stButton button, .stDownloadButton button { border-radius: var(--expo-radius); box-shadow: none;
  transition: border-color .15s ease, background .15s ease, transform .15s ease; }
.stButton button:hover, .stDownloadButton button:hover { transform: translateY(-1px); }
section[data-testid="stSidebar"] a { transition: padding-left .2s ease; }
section[data-testid="stSidebar"] a:hover { padding-left: .9rem; }

/* marca en la barra lateral */
.xbrand { padding: .2rem 0 .8rem; }
.xbrand b { font-size: 1.4rem; letter-spacing: -.3px; color: #fff; border-bottom: 3px solid var(--expo-orange); padding-bottom: 2px; }
.xbrand span { display: block; font-size: .8rem; opacity: .75; margin-top: -2px; color: #fff; }


/* banner de inicio: título grande, decoración de círculos concéntricos y pasos del flujo */
.xbanner { position: relative; overflow: hidden; isolation: isolate; border-radius: var(--expo-radius);
  background: radial-gradient(120% 140% at 12% 20%, #14625D 0%, #0B3B3A 45%, #072423 100%);
  border-bottom: 4px solid var(--expo-orange); color: #fff; margin-bottom: 1.4rem;
  padding: clamp(1.6rem, 4vw, 2.8rem) clamp(1.2rem, 4vw, 3rem); min-height: 16rem;
  display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 1.6rem 2rem; }
.xbanner:before, .xbanner:after { content: ""; position: absolute; border-radius: 50%; z-index: -1;
  border: 1px solid rgba(255,255,255,.10); }
.xbanner:before { width: 38rem; height: 38rem; left: -16rem; top: -11rem;
  background: radial-gradient(circle, rgba(255,255,255,.05) 0%, rgba(255,255,255,0) 70%); }
.xbanner:after { width: 56rem; height: 56rem; left: -24rem; top: -17rem; border-color: rgba(255,255,255,.06); }
.xbanner .xline { position: absolute; left: 3.2rem; top: 0; width: 1px; height: 2.2rem; background: #7DD3FC; }
.xbanner .xmain { max-width: 46rem; min-width: 0; }
.xeyebrow { display: inline-flex; align-items: center; gap: .55rem; color: #7DD3FC; font-weight: 700;
  font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }
.xeyebrow i { width: 11px; height: 11px; border-radius: 50%; background: #7DD3FC;
  box-shadow: 0 0 0 4px rgba(125,211,252,.25); display: inline-block; }
.xbanner h1.xtitle { font-family: var(--expo-serif); font-weight: 500; color: #fff; margin: .35rem 0 .1rem;
  padding: 0; font-size: clamp(2.4rem, 6vw, 3.9rem); line-height: 1; letter-spacing: -.025em; }
.xbanner h1.xtitle em { font-style: normal; color: #7DD3FC; }
.xbanner .xheadline { font-family: var(--expo-serif); font-size: clamp(1.25rem, 2.6vw, 1.8rem); line-height: 1.25;
  color: #E8F5F4; margin: .3rem 0 .7rem; font-weight: 400; }
.xbanner .xlead { color: rgba(232,245,244,.82); margin: 0; max-width: 56ch; line-height: 1.5; }
.xsteps { display: flex; align-items: center; gap: .9rem; flex-wrap: wrap; }
.xstep { display: inline-flex; align-items: center; gap: .6rem; color: #E8F5F4; font-size: .98rem; }
.xstep b { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 50%; flex: none;
  border: 1px solid rgba(125,211,252,.6); color: #7DD3FC; font-size: .78rem; font-variant-numeric: tabular-nums; }
.xstep .xi { font-size: 18px; opacity: .85; }
.xsteps .xsep { width: 2.6rem; height: 1px; background: rgba(255,255,255,.25); }
@media (max-width: 700px) {
  .xbanner { min-height: 0; align-items: flex-start; }
  .xsteps .xsep { display: none; }
  .xsteps { gap: .6rem 1.2rem; }
}

/* pantallas angostas */
@media (max-width: 700px) {
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: var(--expo-gap); }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex: 1 1 100%; min-width: 0; }
  .xhero { padding: 1.3rem 1.2rem; }
}
@media (max-width: 380px) { .xgrid { grid-template-columns: 1fr; } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def _icon(name: str, size: int = 24) -> str:
    return f'<span class="xi" aria-hidden="true" style="font-size:{size}px">{name}</span>'


def hero(title: str, subtitle: str, tag: str = ""):
    tag_html = f'<span class="xtag">{tag}</span><br>' if tag else ""
    st.markdown(f'<div class="xhero">{tag_html}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def hero_banner(eyebrow: str, title: str, headline: str, lead: str, steps: list):
    """Banner de inicio. steps: lista de (ícono, etiqueta) que se numeran en orden."""
    parts = []
    for i, (icon, label) in enumerate(steps, 1):
        if i > 1:
            parts.append('<span class="xsep"></span>')
        parts.append(f'<span class="xstep"><b>{i:02d}</b>{_icon(icon, 18)}<span>{label}</span></span>')
    st.markdown(
        f'<div class="xbanner"><span class="xline"></span>'
        f'<div class="xmain"><span class="xeyebrow"><i></i>{eyebrow}</span>'
        f'<h1 class="xtitle">{title}</h1><div class="xheadline">{headline}</div><p class="xlead">{lead}</p></div>'
        f'<div class="xsteps">{"".join(parts)}</div></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, icon: str):
    st.markdown(f'<div class="xhead"><div class="xico">{_icon(icon, 24)}</div>'
                f'<div><h1>{title}</h1><p>{subtitle}</p></div></div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str = "") -> str:
    sub_html = f'<div class="xsub">{sub}</div>' if sub else ""
    style = f' style="border-top-color:{color}"' if color else ""
    cls = "xcard xtop" if color else "xcard"
    return (f'<div class="{cls}"{style}><div class="xlabel">{label}</div>'
            f'<div class="xvalue">{value}</div>{sub_html}</div>')


def card_grid(cards):
    """Muestra tarjetas de altura uniforme en una cuadrícula adaptable."""
    st.markdown(f'<div class="xgrid">{"".join(cards)}</div>', unsafe_allow_html=True)


def kpi_row(items: list):
    """items: lista de (etiqueta, valor, subtítulo) o (etiqueta, valor, subtítulo, color de acento)."""
    card_grid(kpi(*item) for item in items)


def dim_card(name: str, indicators: str) -> str:
    color = DIM_COLORS[name]
    return (f'<div class="xcard"><div class="xbadge" style="background:{color}">{_icon(DIM_ICONS[name], 20)}</div>'
            f'<h3>{name}</h3><p>{indicators}</p></div>')


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="xnote{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int | None = None):
    """Aplica el aspecto común a las figuras de Plotly."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Atkinson Hyperlegible, Source Sans, Segoe UI, sans-serif", size=13, color="#0F2A2A"),
        colorway=[PRIMARY, ACCENT, "#0EA5E9", "#8B5CF6", "#EF4444", "#14B8A6", "#F59E0B"],
        margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    if height:
        fig.update_layout(height=height)
    return fig

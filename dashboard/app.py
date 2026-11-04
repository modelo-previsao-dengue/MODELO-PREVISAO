"""
DengueCast — Sistema Preditivo de Surtos de Dengue
"""
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "model_ready_v4"
REG_DIR = ROOT / "models" / "regression_v4"
CLS_DIR = ROOT / "models" / "classification_v4"
MAP_PATH = ROOT / "data" / "model_ready_v3" / "municipio_mesorregiao.csv"
GEO_PATH = ROOT / "data" / "geojson" / "mesorregioes.geojson"

RISK = {0: "Baixo", 1: "Moderado", 2: "Alto", 3: "Surto"}
RISK_CLR = {"Baixo": "#22c55e", "Moderado": "#eab308", "Alto": "#f97316", "Surto": "#ef4444"}
UF = {
    11:"RO",12:"AC",13:"AM",14:"RR",15:"PA",16:"AP",17:"TO",
    21:"MA",22:"PI",23:"CE",24:"RN",25:"PB",26:"PE",27:"AL",
    28:"SE",29:"BA",31:"MG",32:"ES",33:"RJ",35:"SP",
    41:"PR",42:"SC",43:"RS",50:"MS",51:"MT",52:"GO",53:"DF",
}

st.set_page_config(page_title="DengueCast", page_icon="🦟", layout="wide",
                    initial_sidebar_state="expanded")

# ╔══════════════════════════════════════════════════════════╗
# ║                   DESIGN SYSTEM                         ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --font: 'Plus Jakarta Sans', -apple-system, sans-serif;
  --bg: #f5f6f8;
  --card: #ffffff;
  --border: #e4e7ec;
  --text: #101828;
  --muted: #98a2b3;
  --sub: #667085;
  --ink: #344054;
  --primary: #5046e5;
  --primary-hover: #4338ca;
  --primary-soft: #eef2ff;
  --surface: #f9fafb;
  --radius: 14px;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.05);
  --shadow-md: 0 4px 16px rgba(16,24,40,.06);
  --shadow-lg: 0 8px 32px rgba(16,24,40,.08);
}

/* ─ Base ─ */
html, body, [class*="css"], .stApp {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
#MainMenu, footer { visibility: hidden !important; height: 0 !important }
header[data-testid="stHeader"] {
  background: transparent !important; border: none !important;
  box-shadow: none !important;
}
[data-testid="stHeader"] [data-testid="stStatusWidget"],
[data-testid="stHeader"] .stDeployButton,
[data-testid="stHeader"] button[kind="header"],
[data-testid="stToolbar"] {
  display: none !important;
}
.block-container {
  padding: 1rem 2.2rem 3rem !important;
  max-width: 1380px;
}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {
  background: #fafbfc !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  padding: 1.2rem 1rem !important;
}
section[data-testid="stSidebar"] * { color: var(--sub) !important }
section[data-testid="stSidebar"] .stSelectbox label {
  font-size: .6rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .05em !important;
  color: var(--muted) !important; margin-bottom: .15rem !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
  font-size: .8rem !important; min-height: 36px !important;
  border-color: #e4e7ec !important; background: #fff !important;
}
section[data-testid="stSidebar"] hr {
  border-color: #eef0f3 !important; margin: .6rem 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: var(--text) !important; font-size: .85rem !important; font-weight: 800 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
  font-size: .55rem !important; color: var(--muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: #fff !important; border: 1px solid #eef0f3 !important;
  border-radius: 10px !important; padding: .55rem .7rem !important;
}

/* ─ Sidebar expand button — BIG and visible ─ */
[data-testid="collapsedControl"] {
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 0 12px 12px 0 !important;
  width: 36px !important;
  height: 36px !important;
  left: 0 !important;
  top: 14px !important;
  box-shadow: var(--shadow-md) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  z-index: 999 !important;
}
[data-testid="collapsedControl"]:hover {
  background: var(--primary-hover) !important;
}
[data-testid="collapsedControl"] svg {
  fill: #fff !important;
  stroke: #fff !important;
  width: 18px !important;
  height: 18px !important;
}

/* ─ Metrics ─ */
[data-testid="stMetricValue"] {
  font-size: 1.05rem !important; font-weight: 700 !important; color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  font-size: .62rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .05em !important;
  color: var(--muted) !important;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important; background: transparent !important;
  border-bottom: 2px solid var(--border) !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 0 !important; font-weight: 600 !important; font-size: .78rem !important;
  padding: .65rem 1.2rem !important; color: var(--muted) !important;
  border-bottom: 2px solid transparent !important; background: transparent !important;
  margin-bottom: -2px !important; letter-spacing: -.01em !important;
  transition: color .15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--ink) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--primary) !important; border-bottom-color: var(--primary) !important;
  font-weight: 700 !important;
}

/* ─ Components ─ */
.stSelectbox > div > div {
  border-radius: 10px !important; border-color: var(--border) !important;
  font-size: .82rem !important;
}
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important; overflow: hidden;
  box-shadow: var(--shadow-sm) !important; border: 1px solid var(--border) !important;
}

/* ─ Plotly ─ */
.js-plotly-plot .plotly .modebar { display: none !important }

/* ─ Download button ─ */
.stDownloadButton > button {
  background: var(--primary) !important; color: #fff !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: .76rem !important;
  padding: .55rem 1.3rem !important; letter-spacing: -.01em !important;
  box-shadow: 0 2px 8px rgba(80,70,229,.25) !important;
  transition: all .15s !important;
}
.stDownloadButton > button:hover {
  background: var(--primary-hover) !important;
  box-shadow: 0 4px 16px rgba(80,70,229,.35) !important;
  transform: translateY(-1px) !important;
}

/* ─ Slider ─ */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--primary) !important;
}

/* ─ Risk filter pills ─ */
div[data-testid="stRadio"] > div { gap: 0 !important }
div[data-testid="stRadio"] [role="radiogroup"] {
  gap: .3rem !important; background: #f2f4f7; padding: .25rem .3rem;
  border-radius: 10px; display: inline-flex !important;
}
div[data-testid="stRadio"] label {
  font-size: .72rem !important; font-weight: 600 !important;
  padding: .35rem .85rem !important; border-radius: 8px !important;
  color: var(--sub) !important; cursor: pointer !important;
  transition: all .15s !important; white-space: nowrap !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
  background: #fff !important; color: var(--text) !important;
  box-shadow: 0 1px 4px rgba(16,24,40,.08) !important; font-weight: 700 !important;
}
div[data-testid="stRadio"] label input { display: none !important }
div[data-testid="stRadio"] label p { margin: 0 !important }
div[data-testid="stRadio"] > label:first-child { display: none !important }

/* ─ Iframe no scroll ─ */
iframe[title="streamlit_app.components.v1.html"] {
  overflow: hidden !important;
}
</style>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                    COMPONENTS                           ║
# ╚══════════════════════════════════════════════════════════╝

PLOTLY_FONT = "Plus Jakarta Sans, -apple-system, sans-serif"

def plo(h=380, **kw):
    d = dict(
        height=h, template="plotly_white",
        font=dict(family=PLOTLY_FONT, size=11, color="#667085"),
        margin=dict(l=48, r=16, t=32, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10, family=PLOTLY_FONT)),
    )
    d.update(kw)
    return d


def sec(title):
    st.markdown(
        f'<p style="font-size:.7rem;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:.08em;color:#98a2b3;margin:2rem 0 .65rem;'
        f'display:flex;align-items:center;gap:.5rem">'
        f'<span style="width:16px;height:2px;background:#5046e5;border-radius:1px"></span>'
        f'{title}</p>',
        unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    accent_map = {"#ef4444":"#fef2f2","#f97316":"#fff7ed","#eab308":"#fefce8","#22c55e":"#f0fdf4"}
    for col, (val, label, color) in zip(cols, items):
        bg = accent_map.get(color, "#f9fafb")
        col.markdown(
            f'<div style="background:{bg};border:1px solid {color}22;border-radius:14px;'
            f'padding:1.15rem 1.3rem;position:relative;overflow:hidden">'
            f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
            f'background:{color};border-radius:14px 14px 0 0"></div>'
            f'<div style="font-size:2rem;font-weight:800;color:{color};letter-spacing:-.04em;'
            f'line-height:1">{val:,}</div>'
            f'<div style="font-size:.62rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;color:#667085;margin-top:.35rem">{label}</div></div>',
            unsafe_allow_html=True)


def card_start():
    st.markdown(
        '<div style="background:#fff;border:1px solid #e4e7ec;border-radius:14px;'
        'padding:1.2rem 1.5rem;box-shadow:0 1px 3px rgba(16,24,40,.04);margin-bottom:.5rem">',
        unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                     DATA LOADING                        ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data
def load_meso_names():
    if MAP_PATH.exists():
        df = pd.read_csv(MAP_PATH).drop_duplicates("cod_mesorregiao")
        return df.set_index("cod_mesorregiao")[["nome_mesorregiao", "uf_sigla"]].to_dict("index")
    return {}

@st.cache_data
def load_geojson():
    if GEO_PATH.exists():
        with open(GEO_PATH) as f:
            return json.load(f)
    return None

@st.cache_data
def load_all_data(var):
    dfs = []
    for split in ["train", "val", "test"]:
        p = DATA / f"splits_{var.lower()}" / f"{split}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["split"] = split
            dfs.append(df)
    if not dfs:
        return None
    full = pd.concat(dfs, ignore_index=True)
    full["uf_code"] = full["cod_mesorregiao"] // 100
    full["uf"] = full["uf_code"].map(UF)
    return full

@st.cache_resource
def load_models(var, sub):
    rm = cm = None
    rp = REG_DIR / f"model_{var}_{sub}.ubj"
    cp = CLS_DIR / f"model_{var}_{sub}_cls.ubj"
    if rp.exists():
        rm = xgb.Booster(); rm.load_model(str(rp))
    if cp.exists():
        cm = xgb.Booster(); cm.load_model(str(cp))
    return rm, cm

@st.cache_data
def run_predictions(_rm, _cm, df, _var, _sub):
    feats = _rm.feature_names
    dm = xgb.DMatrix(df[feats], feature_names=feats)
    pred_log = _rm.predict(dm)
    pred_notif = np.maximum(np.expm1(pred_log), 0)
    probs = _cm.predict(dm)
    pred_cls = probs.argmax(axis=1)
    out = df[["cod_mesorregiao","ano","semana_epidemiologica",
              "notificacoes","uf","notificacoes_t4","risco_surto_t4","split"]].copy()
    out["pred_notif"] = pred_notif
    out["pred_risco"] = pred_cls
    out["pred_label"] = pd.Series(pred_cls).map(RISK).values
    out["real_label"] = df["risco_surto_t4"].map(RISK).values
    for i, nm in RISK.items():
        out[f"p_{nm.lower()}"] = probs[:, i]
    out["periodo"] = out["ano"].astype(str) + "-S" + out["semana_epidemiologica"].astype(str).str.zfill(2)
    return out

@st.cache_data
def load_reports():
    reps = {}
    for key, fn in [("reg","46_regression_v4_report.json"),("wf","47_walkforward_v4_report.json"),
                    ("cls","54_classification_v4_report.json")]:
        p = DATA / fn
        if p.exists():
            with open(p) as f: reps[key] = json.load(f)
    sp = REG_DIR / "shap_importance_v4.csv"
    if sp.exists(): reps["shap"] = pd.read_csv(sp)
    return reps

def mn(cod, names):
    info = names.get(cod, {})
    nm = info.get("nome_mesorregiao", "")
    uf = info.get("uf_sigla", "")
    return f"{nm} ({uf})" if nm else str(cod)


# ╔══════════════════════════════════════════════════════════╗
# ║                     PDF EXPORT                          ║
# ╚══════════════════════════════════════════════════════════╝

def make_pdf(counts, high_df, names, var, sub, sel_ano, sel_uf):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_fill_color(80, 70, 229)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(6)
    pdf.cell(0, 10, "DengueCast - Relatorio de Alertas", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5, f"Gerado: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Var {var} / {sub} | Ano: {sel_ano} | UF: {sel_uf}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(34)
    pdf.set_text_color(16, 24, 40)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Resumo", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for label, (r, g, b) in [("Surto",(239,68,68)),("Alto",(249,115,22)),("Moderado",(234,179,8)),("Baixo",(34,197,94))]:
        pdf.set_fill_color(r, g, b)
        pdf.rect(pdf.get_x(), pdf.get_y()+1.5, 3, 3, "F")
        pdf.set_x(pdf.get_x()+6)
        pdf.cell(0, 6, f"{label}: {counts.get(label,0)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    if not high_df.empty:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Top 20 Mesorregioes em Alerta", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(241, 245, 249)
        ws = [65, 14, 26, 22, 22]
        for h, w in zip(["Mesorregiao","UF","Prev.t+4","Risco","P(Surto)"], ws):
            pdf.cell(w, 6, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for _, row in high_df.head(20).iterrows():
            nm = mn(row["cod_mesorregiao"], names)
            if len(nm) > 32: nm = nm[:30]+".."
            pdf.cell(ws[0], 5, nm, border=1)
            pdf.cell(ws[1], 5, str(row["uf"]), border=1, align="C")
            pdf.cell(ws[2], 5, f"{row['pred_notif']:,.0f}", border=1, align="R")
            pdf.cell(ws[3], 5, str(row["pred_label"]), border=1, align="C")
            pdf.cell(ws[4], 5, f"{row['p_surto']*100:.1f}%", border=1, align="C")
            pdf.ln()
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ╔══════════════════════════════════════════════════════════╗
# ║                        MAIN                             ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    names = load_meso_names()
    geojson = load_geojson()
    reports = load_reports()

    # ── Header ──
    st.markdown(
        '<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;padding-bottom:.4rem">'
        '<div style="width:28px;height:28px;border-radius:7px;display:flex;align-items:center;'
        'justify-content:center;background:#101828;position:relative;overflow:hidden">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="#5046e5" opacity=".3"/>'
        '<path d="M12 6v6l4.24 2.54" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
        '<circle cx="12" cy="12" r="2" fill="#ef4444"/>'
        '</svg></div>'
        '<div>'
        '<div style="font-size:.88rem;font-weight:800;color:#101828;letter-spacing:-.03em;line-height:1.1">'
        'Dengue<span style="color:#5046e5">Cast</span></div>'
        '<div style="font-size:.52rem;color:#98a2b3;font-weight:500;letter-spacing:.02em">'
        'Vigilancia epidemiologica preditiva</div>'
        '</div></div>',
        unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.45rem;margin-bottom:.1rem">'
            '<div style="width:24px;height:24px;border-radius:6px;background:#101828;display:flex;'
            'align-items:center;justify-content:center">'
            '<svg width="13" height="13" viewBox="0 0 24 24" fill="none">'
            '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="#5046e5" opacity=".3"/>'
            '<path d="M12 6v6l4.24 2.54" stroke="#fff" stroke-width="2" stroke-linecap="round"/>'
            '<circle cx="12" cy="12" r="2" fill="#ef4444"/></svg></div>'
            '<div style="font-size:.78rem;font-weight:800;color:#101828!important;letter-spacing:-.02em">'
            'Dengue<span style="color:#5046e5!important">Cast</span></div></div>',
            unsafe_allow_html=True)
        st.markdown("---")

        var = st.selectbox("Modelo", ["B","C","A"],
                           format_func={"B":"Var B — Completa (22a)","C":"Var C — Decay",
                                        "A":"Var A — Baseline (4a)"}.get)
        sub = st.selectbox("Features", ["inmet_enriquecido","inmet_bruto","sinan_only"],
                           format_func={"inmet_enriquecido":"INMET enriq. (266)",
                                        "inmet_bruto":"INMET bruto (122)",
                                        "sinan_only":"SINAN-only (110)"}.get)

    full_df = load_all_data(var)
    if full_df is None:
        st.error("Dados nao encontrados.")
        return
    rm, cm = load_models(var, sub)
    if rm is None or cm is None:
        st.error(f"Modelos {var}_{sub} nao encontrados.")
        return

    with st.spinner("Gerando previsoes..."):
        preds = run_predictions(rm, cm, full_df, var, sub)

    with st.sidebar:
        st.markdown("---")
        all_anos = sorted(preds["ano"].unique())
        sel_ano = st.selectbox("Ano", ["Todos"]+[str(a) for a in all_anos], index=0)
        ufs = sorted(preds["uf"].dropna().unique())
        sel_uf = st.selectbox("UF", ["Todos"]+list(ufs))
        st.markdown("---")
        st.markdown(
            f'<div style="display:grid;gap:.4rem">'
            f'<div style="background:#fff;border:1px solid #eef0f3;border-radius:10px;padding:.5rem .65rem">'
            f'<div style="font-size:.5rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#98a2b3">Previsoes</div>'
            f'<div style="font-size:.88rem;font-weight:800;color:#101828">{len(preds):,}</div></div>'
            f'<div style="background:#fff;border:1px solid #eef0f3;border-radius:10px;padding:.5rem .65rem">'
            f'<div style="font-size:.5rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#98a2b3">Cobertura</div>'
            f'<div style="font-size:.88rem;font-weight:800;color:#101828">{all_anos[0]}–{all_anos[-1]}</div></div>'
            f'<div style="background:#fff;border:1px solid #eef0f3;border-radius:10px;padding:.5rem .65rem">'
            f'<div style="font-size:.5rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#98a2b3">Mesorregioes</div>'
            f'<div style="font-size:.88rem;font-weight:800;color:#101828">137</div></div>'
            f'</div>',
            unsafe_allow_html=True)

    view = preds.copy()
    if sel_ano != "Todos":
        view = view[view["ano"]==int(sel_ano)]
    if sel_uf != "Todos":
        view = view[view["uf"]==sel_uf]

    t1, t2, t3, t4, t5 = st.tabs(["Alertas", "Regiao", "Temporal", "Modelo", "Sobre"])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — ALERTAS
    # ════════════════════════════════════════════════════════════
    with t1:
        latest = view.sort_values(["ano","semana_epidemiologica"]).groupby("cod_mesorregiao").last().reset_index()
        counts = {r: int((latest["pred_risco"]==i).sum()) for i, r in RISK.items()}

        kpi_row([
            (counts["Surto"], "Surto", "#ef4444"),
            (counts["Alto"], "Alto", "#f97316"),
            (counts["Moderado"], "Moderado", "#eab308"),
            (counts["Baixo"], "Baixo", "#22c55e"),
        ])

        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        risk_filter = st.radio(
            "Filtro de risco", ["Todos", "Surto", "Alto", "Moderado", "Baixo"],
            horizontal=True, label_visibility="collapsed", key="risk_filter")

        risk_id_map = {"Surto": 3, "Alto": 2, "Moderado": 1, "Baixo": 0}
        if risk_filter != "Todos":
            latest = latest[latest["pred_risco"] == risk_id_map[risk_filter]]

        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)

        if geojson:
            sec("Mapa de risco por mesorregiao")
            map_df = latest[["cod_mesorregiao","pred_risco","pred_label","pred_notif","p_surto","uf"]].copy()
            map_df["cod_str"] = map_df["cod_mesorregiao"].astype(str)
            map_df["nome"] = map_df["cod_mesorregiao"].apply(lambda c: mn(c, names))
            map_df["p_surto_pct"] = (map_df["p_surto"] * 100).round(1)

            fig_map = px.choropleth_map(
                map_df, geojson=geojson,
                locations="cod_str", featureidkey="properties.codarea",
                color="pred_risco",
                color_continuous_scale=[[0,"#16a34a"],[0.33,"#eab308"],[0.66,"#ea580c"],[1,"#dc2626"]],
                range_color=[0, 3],
                map_style="white-bg",
                zoom=3.2, center={"lat": -14.5, "lon": -53},
                opacity=0.88,
                hover_name="nome",
                hover_data={"cod_str": False, "pred_risco": False,
                            "pred_label": True, "pred_notif": ":.0f",
                            "p_surto_pct": True, "uf": True},
                labels={"pred_label": "Risco", "pred_notif": "Prev. t+4",
                        "p_surto_pct": "P(Surto) %", "uf": "UF"},
            )
            fig_map.update_traces(marker_line_width=0.5, marker_line_color="#ffffff")
            fig_map.update_layout(
                height=540,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family=PLOTLY_FONT, size=11),
                dragmode=False,
                coloraxis_colorbar=dict(
                    title=dict(text="Risco", font=dict(size=11)),
                    tickvals=[0,1,2,3],
                    ticktext=["Baixo","Moderado","Alto","Surto"],
                    tickfont=dict(size=10),
                    len=.45, thickness=10, x=1.01,
                    outlinewidth=0,
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True,
                            config={"scrollZoom": False, "doubleClick": False,
                                    "displayModeBar": False})

        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        sec("Risco medio por UF")
        col_chart, col_table = st.columns([1, 1])

        with col_chart:
            uf_risk = latest.groupby("uf")["pred_risco"].mean().sort_values(ascending=True).reset_index()
            uf_risk.columns = ["uf", "risco"]
            fig_uf = go.Figure(go.Bar(
                y=uf_risk["uf"], x=uf_risk["risco"], orientation="h",
                marker=dict(
                    color=uf_risk["risco"],
                    colorscale=[[0,"#16a34a"],[.33,"#eab308"],[.66,"#ea580c"],[1,"#dc2626"]],
                    cmin=0, cmax=3),
                hovertemplate="<b>%{y}</b><br>Risco: %{x:.2f}<extra></extra>",
            ))
            fig_uf.update_layout(**plo(h=520,
                                       xaxis=dict(range=[0,3.3], showgrid=False, title="",
                                                  tickfont=dict(size=10)),
                                       yaxis=dict(title="", tickfont=dict(size=10))))
            st.plotly_chart(fig_uf, use_container_width=True)

        with col_table:
            high = latest[latest["pred_risco"]>=2].nlargest(20, "pred_notif").copy()
            if high.empty:
                st.info("Nenhum alerta Alto/Surto no periodo selecionado.")
            else:
                sec("Top alertas")
                tbl = high[["cod_mesorregiao","uf","pred_notif","pred_label","p_surto"]].copy()
                tbl["Mesorregiao"] = tbl["cod_mesorregiao"].apply(lambda c: mn(c, names))
                tbl["Prev"] = tbl["pred_notif"].round(0).astype(int)
                tbl["Risco"] = tbl["pred_label"]
                tbl["P(Surto)%"] = (tbl["p_surto"] * 100).round(1)
                show = tbl[["Mesorregiao","uf","Prev","Risco","P(Surto)%"]].rename(columns={"uf":"UF"})
                st.dataframe(show, use_container_width=True, hide_index=True,
                             height=min(520, 36*len(show)+38),
                             column_config={
                                 "P(Surto)%": st.column_config.ProgressColumn(
                                     "P(Surto)", min_value=0, max_value=100, format="%.1f%%"),
                                 "Prev": st.column_config.NumberColumn("Prev. t+4", format="%d"),
                             })

        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
        c_dl, _ = st.columns([1, 3])
        with c_dl:
            st.download_button("Exportar relatorio PDF",
                               data=make_pdf(counts, high, names, var, sub, sel_ano, sel_uf),
                               file_name=f"denguecast_{datetime.now().strftime('%Y%m%d')}.pdf",
                               mime="application/pdf")

        sec("Evolucao semanal de alertas")
        wk = view.groupby(["ano","semana_epidemiologica"]).agg(
            surto=("pred_risco", lambda x: (x==3).sum()),
            alto=("pred_risco", lambda x: (x==2).sum()),
        ).reset_index()
        wk["per"] = wk["ano"].astype(str)+"-S"+wk["semana_epidemiologica"].astype(str).str.zfill(2)
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(
            x=wk["per"], y=wk["surto"], name="Surto",
            line=dict(color="#dc2626", width=2),
            fill="tozeroy", fillcolor="rgba(220,38,38,.05)"))
        fig_evo.add_trace(go.Scatter(
            x=wk["per"], y=wk["alto"], name="Alto",
            line=dict(color="#ea580c", width=1.5, dash="dot")))
        fig_evo.update_layout(**plo(h=280,
                                     xaxis=dict(showgrid=False, title=""),
                                     yaxis=dict(gridcolor="#f2f4f7", gridwidth=1, title="")))
        st.plotly_chart(fig_evo, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 2 — REGIAO
    # ════════════════════════════════════════════════════════════
    with t2:
        mesos = sorted(view["cod_mesorregiao"].unique())
        opts = {m: mn(m, names) for m in mesos}
        sel_meso = st.selectbox("Buscar mesorregiao", mesos,
                                format_func=lambda x: opts[x], key="meso")
        if not sel_meso:
            return

        df_m = preds[preds["cod_mesorregiao"]==sel_meso].sort_values(
            ["ano","semana_epidemiologica"]).copy()
        last = df_m.iloc[-1]
        rl = last["pred_label"]

        pill_bg = {"Surto":"#fef2f2","Alto":"#fff7ed","Moderado":"#fefce8","Baixo":"#f0fdf4"}
        pill_fg = {"Surto":"#dc2626","Alto":"#ea580c","Moderado":"#a16207","Baixo":"#16a34a"}
        pill_border = {"Surto":"#fecaca","Alto":"#fed7aa","Moderado":"#fef08a","Baixo":"#bbf7d0"}
        st.markdown(
            f'<div style="background:#fff;border:1px solid #e4e7ec;border-radius:14px;'
            f'padding:1.1rem 1.4rem;margin-bottom:1rem;display:flex;align-items:center;gap:.8rem;'
            f'box-shadow:0 1px 3px rgba(16,24,40,.04)">'
            f'<span style="font-size:1.08rem;font-weight:800;color:#101828;letter-spacing:-.02em">'
            f'{opts[sel_meso]}</span>'
            f'<span style="font-size:.72rem;color:#98a2b3;font-weight:500">'
            f'{int(last["ano"])}-S{int(last["semana_epidemiologica"]):02d}</span>'
            f'<span style="padding:.2rem .6rem;border-radius:6px;font-size:.65rem;'
            f'font-weight:700;background:{pill_bg[rl]};color:{pill_fg[rl]};'
            f'border:1px solid {pill_border[rl]}">{rl}</span></div>',
            unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Notificacoes", f"{last['notificacoes']:,.0f}")
        c2.metric("Previsao t+4", f"{last['pred_notif']:,.0f}")
        c3.metric("Risco", rl)
        c4.metric("P(Surto)", f"{last['p_surto']*100:.1f}%")

        sec("Serie temporal")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_m["periodo"], y=df_m["notificacoes_t4"], name="Real (t+4)",
            line=dict(color="#5046e5", width=2)))
        fig_ts.add_trace(go.Scatter(
            x=df_m["periodo"], y=df_m["pred_notif"], name="Previsto (t+4)",
            line=dict(color="#dc2626", width=2, dash="dash")))
        fig_ts.add_trace(go.Scatter(
            x=df_m["periodo"], y=df_m["notificacoes"], name="Atual",
            line=dict(color="#d0d5dd", width=1)))
        fig_ts.update_layout(**plo(h=360,
                                    xaxis=dict(showgrid=False, title=""),
                                    yaxis=dict(gridcolor="#f2f4f7", title="")))
        st.plotly_chart(fig_ts, use_container_width=True)

        sec("Probabilidade de risco")
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=df_m["periodo"], y=df_m["p_surto"], name="P(Surto)",
            marker_color="rgba(220,38,38,.45)"))
        fig_p.add_trace(go.Bar(
            x=df_m["periodo"], y=df_m["p_alto"], name="P(Alto)",
            marker_color="rgba(234,88,12,.25)"))
        fig_p.update_layout(**plo(h=240, barmode="overlay",
                                   yaxis=dict(range=[0,1], gridcolor="#f2f4f7", title=""),
                                   xaxis=dict(showgrid=False, title="")))
        st.plotly_chart(fig_p, use_container_width=True)

        from sklearn.metrics import r2_score, mean_absolute_error
        act = df_m["notificacoes_t4"].values
        pred = df_m["pred_notif"].values
        mk = ~np.isnan(act) & ~np.isnan(pred) & (act>=0)
        if mk.sum() > 5:
            sec("Metricas locais")
            m1,m2,m3 = st.columns(3)
            m1.metric("R² (log)", f"{r2_score(np.log1p(act[mk]), np.log1p(pred[mk])):.3f}")
            m2.metric("MAE", f"{mean_absolute_error(act[mk], pred[mk]):.0f}")
            m3.metric("Acuracia risco",
                      f"{(df_m['pred_risco'].values==df_m['risco_surto_t4'].values).mean():.0%}")

    # ════════════════════════════════════════════════════════════
    # TAB 3 — TEMPORAL
    # ════════════════════════════════════════════════════════════
    with t3:
        sec("Previsto vs Real (log scale)")
        samp = view.sample(min(5000, len(view)), random_state=42)
        samp["log_real"] = np.log1p(samp["notificacoes_t4"])
        samp["log_pred"] = np.log1p(samp["pred_notif"])
        fig_sc = px.scatter(
            samp, x="log_real", y="log_pred",
            color="pred_label", color_discrete_map=RISK_CLR, opacity=.3,
            labels={"log_real":"log(Real+1)","log_pred":"log(Previsto+1)","pred_label":"Risco"})
        mx = max(samp["log_real"].quantile(.99), samp["log_pred"].quantile(.99))
        fig_sc.add_trace(go.Scatter(
            x=[0,mx], y=[0,mx], mode="lines",
            line=dict(color="#d0d5dd", dash="dash", width=1.5), showlegend=False))
        fig_sc.update_layout(**plo(h=440,
                                    xaxis=dict(gridcolor="#f2f4f7"),
                                    yaxis=dict(gridcolor="#f2f4f7")))
        st.plotly_chart(fig_sc, use_container_width=True)

        sec("Total nacional semanal")
        wk_agg = view.groupby("periodo").agg(
            real=("notificacoes_t4","sum"),
            prev=("pred_notif","sum")).reset_index().sort_values("periodo")
        fig_nat = go.Figure()
        fig_nat.add_trace(go.Scatter(
            x=wk_agg["periodo"], y=wk_agg["real"], name="Real",
            line=dict(color="#5046e5", width=2)))
        fig_nat.add_trace(go.Scatter(
            x=wk_agg["periodo"], y=wk_agg["prev"], name="Previsto",
            line=dict(color="#dc2626", width=2, dash="dash")))
        fig_nat.update_layout(**plo(h=320,
                                     xaxis=dict(showgrid=False, title=""),
                                     yaxis=dict(gridcolor="#f2f4f7", title="")))
        st.plotly_chart(fig_nat, use_container_width=True)

        sec("R² por ano")
        from sklearn.metrics import r2_score
        yr_stats = []
        for yr in sorted(view["ano"].unique()):
            sub_df = view[view["ano"]==yr]
            a2 = sub_df["notificacoes_t4"].values
            p2 = sub_df["pred_notif"].values
            mk2 = ~np.isnan(a2) & (a2>=0)
            if mk2.sum()>10:
                yr_stats.append({
                    "Ano": yr,
                    "R2": round(r2_score(np.log1p(a2[mk2]), np.log1p(p2[mk2])), 4),
                    "split": sub_df["split"].iloc[0],
                })
        if yr_stats:
            df_yr = pd.DataFrame(yr_stats)
            fig_yr = px.bar(df_yr, x="Ano", y="R2", color="split",
                            color_discrete_map={"train":"#818cf8","val":"#fbbf24","test":"#f87171"},
                            labels={"split":"Split","R2":"R²"})
            fig_yr.update_layout(**plo(h=300,
                                       xaxis=dict(showgrid=False, title=""),
                                       yaxis=dict(gridcolor="#f2f4f7", title="R²")))
            st.plotly_chart(fig_yr, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — MODELO
    # ════════════════════════════════════════════════════════════
    with t4:
        mk_key = f"{var}_{sub}"
        reg = reports.get("reg", {})
        if mk_key in reg.get("modelos", {}):
            m = reg["modelos"][mk_key]
            sec("Regressao — modelo selecionado")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("R² (log)", f"{m['R2_log']:.4f}")
            k2.metric("RMSE", f"{m['RMSE']:,.0f}")
            k3.metric("MAE", f"{m['MAE']:,.1f}")
            k4.metric("IC 95%", f"[{m['R2_log_CI95_lo']:.4f}, {m['R2_log_CI95_hi']:.4f}]")

        sec("Comparativo variantes")
        cv, cc = st.columns(2)
        with cv:
            if reg:
                bars = []
                for v in ["A","B","C"]:
                    k = f"{v}_inmet_enriquecido"
                    if k in reg.get("modelos",{}):
                        mm = reg["modelos"][k]
                        bars.append({"Var":f"Var {v}","R2":mm["R2_log"],
                                     "lo":mm["R2_log_CI95_lo"],"hi":mm["R2_log_CI95_hi"]})
                if bars:
                    clr = {"Var A":"#93c5fd","Var B":"#5046e5","Var C":"#a78bfa"}
                    fig_v = go.Figure()
                    for b in bars:
                        fig_v.add_trace(go.Bar(
                            x=[b["Var"]], y=[b["R2"]],
                            error_y=dict(type="data", symmetric=False,
                                         array=[b["hi"]-b["R2"]],
                                         arrayminus=[b["R2"]-b["lo"]]),
                            marker_color=clr.get(b["Var"]), name=b["Var"],
                            text=f"{b['R2']:.4f}", textposition="outside",
                            textfont=dict(size=11, family=PLOTLY_FONT)))
                    fig_v.update_layout(**plo(h=340, showlegend=False,
                                              yaxis=dict(range=[0.84,0.92], gridcolor="#f2f4f7"),
                                              xaxis=dict(showgrid=False, title="")))
                    st.plotly_chart(fig_v, use_container_width=True)
        with cc:
            cls_r = reports.get("cls")
            if cls_r:
                rows = []
                for v in ["A","B","C"]:
                    k = f"{v}_inmet_enriquecido"
                    if k in cls_r.get("modelos",{}):
                        mm = cls_r["modelos"][k]
                        rows.append({
                            "Variante": f"Var {v}",
                            "F1 macro": f"{mm['f1_macro']:.4f}",
                            "AUC": f"{mm['auc_macro_ovr']:.4f}",
                            "F1 surto": f"{mm['f1_per_class']['surto']:.4f}",
                        })
                if rows:
                    sec("Classificacao de risco")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        wf = reports.get("wf")
        if wf and "modelos" in wf:
            sec("Walk-Forward Validation")
            wfr = []
            for v in ["A","B","C"]:
                k = f"{v}_inmet_enriquecido"
                if k in wf["modelos"]:
                    for i, fold in enumerate(wf["modelos"][k].get("folds",[]),1):
                        yr = fold["test_years"][0] if "test_years" in fold else "?"
                        wfr.append({"Var":f"Var {v}","Fold":f"F{i} ({yr})","R2":fold["R2_log"]})
            if wfr:
                fig_wf = px.line(pd.DataFrame(wfr), x="Fold", y="R2", color="Var",
                                 markers=True,
                                 color_discrete_map={"Var A":"#93c5fd","Var B":"#5046e5",
                                                     "Var C":"#a78bfa"})
                fig_wf.update_layout(**plo(h=300,
                                           xaxis=dict(showgrid=False),
                                           yaxis=dict(gridcolor="#f2f4f7", title="R²")))
                st.plotly_chart(fig_wf, use_container_width=True)

        shap_df = reports.get("shap")
        if shap_df is not None:
            shap_col = "shap_mean_abs_B" if "shap_mean_abs_B" in shap_df.columns else shap_df.columns[1]
            sec("SHAP Feature Importance")
            n = st.slider("Top N", 10, 50, 15, key="shap_n")
            top_n = shap_df.nlargest(n, shap_col)
            fig_sh = px.bar(top_n, x=shap_col, y="feature", orientation="h",
                            color=shap_col,
                            color_continuous_scale=["#e0e7ff","#5046e5"])
            fig_sh.update_layout(**plo(h=max(260, n*22),
                                       yaxis=dict(autorange="reversed", title=""),
                                       xaxis=dict(title="SHAP", gridcolor="#f2f4f7"),
                                       coloraxis_showscale=False))
            st.plotly_chart(fig_sh, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 5 — SOBRE
    # ════════════════════════════════════════════════════════════
    with t5:
        about_html = """<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Plus Jakarta Sans',-apple-system,sans-serif;color:#101828;background:#f5f6f8;
  line-height:1.65;-webkit-font-smoothing:antialiased}

/* Hero */
.hero{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#4338ca 100%);
  border-radius:18px;padding:2.2rem 2.5rem;margin-bottom:1.8rem;position:relative;overflow:hidden}
.hero::after{content:'';position:absolute;top:-40%;right:-10%;width:300px;height:300px;
  background:radial-gradient(circle,rgba(255,255,255,.06) 0%,transparent 70%);border-radius:50%}
.hero h1{font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-.04em;margin-bottom:.5rem;line-height:1.2}
.hero p{font-size:.86rem;color:rgba(255,255,255,.75);max-width:560px;line-height:1.7}
.hero b{color:#c7d2fe}
.hero-stats{display:flex;gap:.7rem;margin-top:1.3rem}
.hs{background:rgba(255,255,255,.1);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.12);
  border-radius:12px;padding:.7rem 1rem;text-align:center;flex:1}
.hs .n{font-size:1.2rem;font-weight:800;color:#fff;letter-spacing:-.02em}
.hs .l{font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:rgba(255,255,255,.5);margin-top:.1rem}

/* Grid layout */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:16px;padding:1.4rem 1.5rem;
  position:relative;overflow:hidden}
.card.full{grid-column:1/-1}
.card-label{font-size:.6rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
  color:#98a2b3;margin-bottom:.8rem;display:flex;align-items:center;gap:.4rem}
.card-label::before{content:'';width:12px;height:2px;background:#5046e5;border-radius:1px}
.card p,.card li{font-size:.82rem;color:#667085;line-height:1.6}
.card b{color:#344054}

/* Pipeline */
.pipe{display:flex;align-items:center;gap:.3rem;flex-wrap:wrap;margin:.6rem 0}
.ps{padding:.35rem .7rem;border-radius:8px;font-size:.7rem;font-weight:700;color:#667085;background:#f2f4f7}
.ps.ml{background:#eef2ff;color:#4338ca}
.ar{color:#d0d5dd;font-size:.75rem}

/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.tbl th{text-align:left;font-size:.58rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em;
  color:#98a2b3;padding:.45rem .6rem;border-bottom:2px solid #e4e7ec}
.tbl td{padding:.45rem .6rem;border-bottom:1px solid #f2f4f7;color:#667085}
.tbl tr:hover td{background:#f9fafb}
.tbl .bld{font-weight:700;color:#101828}

/* Highlight */
.hl{background:linear-gradient(135deg,#eef2ff,#e8defb);border-radius:12px;
  padding:.9rem 1.2rem;margin:.7rem 0;border-left:4px solid #5046e5}
.hl p{color:#3730a3;font-weight:600;font-size:.82rem;margin:0}

/* Badges */
.badge{display:inline-flex;padding:.15rem .45rem;border-radius:5px;font-size:.62rem;font-weight:700;margin:0 .1rem}
.bs{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
.ba{background:#fff7ed;color:#ea580c;border:1px solid #fed7aa}
.bm{background:#fefce8;color:#a16207;border:1px solid #fef08a}
.bb{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}

/* Tech chips */
.techs{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}
.chip{background:#f2f4f7;border:1px solid #e4e7ec;border-radius:8px;padding:.4rem .8rem;
  font-size:.72rem;font-weight:700;color:#344054}

/* Metrics row */
.mrow{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin:.6rem 0}
.mc{text-align:center;padding:.6rem;background:#f9fafb;border-radius:10px;border:1px solid #f2f4f7}
.mc .v{font-size:1rem;font-weight:800;color:#5046e5}
.mc .k{font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#98a2b3;margin-top:.1rem}

/* Validation list */
.vl{list-style:none;padding:0;margin:.5rem 0}
.vl li{padding:.45rem 0;border-bottom:1px solid #f2f4f7;font-size:.82rem;color:#667085;
  display:flex;align-items:flex-start;gap:.5rem}
.vl li::before{content:'✓';color:#5046e5;font-weight:800;font-size:.7rem;margin-top:.1rem}

/* Warning */
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:.8rem 1.1rem}
.warn ul{padding-left:1rem;margin:0}
.warn li{color:#92400e;font-size:.8rem;margin-bottom:.25rem}
</style>
</head><body>

<!-- Hero -->
<div class="hero">
  <h1>DengueCast</h1>
  <p>Sistema preditivo de surtos de dengue desenvolvido como <b>TCC em Engenharia de Software</b>
  na Universidade de Brasilia (UnB). Previsoes de notificacoes e risco de surto
  <b>4 semanas a frente</b> em <b>137 mesorregioes</b> brasileiras.</p>
  <div class="hero-stats">
    <div class="hs"><div class="n">180,703</div><div class="l">Registros</div></div>
    <div class="hs"><div class="n">2001–2026</div><div class="l">Cobertura</div></div>
    <div class="hs"><div class="n">137</div><div class="l">Mesorregioes</div></div>
    <div class="hs"><div class="n">18</div><div class="l">Modelos</div></div>
  </div>
</div>

<!-- Cards grid -->
<div class="grid">

  <!-- Pipeline -->
  <div class="card full">
    <div class="card-label">Pipeline de Dados</div>
    <div class="pipe">
      <span class="ps">SINAN</span><span class="ar">→</span>
      <span class="ps">Limpeza</span><span class="ar">→</span>
      <span class="ps">Feature Eng.</span><span class="ar">→</span>
      <span class="ps">INMET</span><span class="ar">→</span>
      <span class="ps">Merge</span><span class="ar">→</span>
      <span class="ps">Splits</span><span class="ar">→</span>
      <span class="ps ml">XGBoost</span>
    </div>
    <p>Features: lags (1–8 sem.), medias moveis, diferencas semanais,
    sazonalidade ciclica e variaveis climaticas (temperatura, precipitacao, umidade).</p>
  </div>

  <!-- Variantes -->
  <div class="card">
    <div class="card-label">Variantes experimentais</div>
    <table class="tbl">
      <tr><th>Variante</th><th>Treino</th><th>Descricao</th></tr>
      <tr><td>A (Baseline)</td><td>2019–2022</td><td>4 anos recentes</td></tr>
      <tr><td class="bld">B (Completa)</td><td class="bld">2001–2022</td><td class="bld">Serie completa 22a</td></tr>
      <tr><td>C (Decay)</td><td>2001–2022</td><td>Completa + pesos exp.</td></tr>
    </table>
  </div>

  <!-- Sub-modelos -->
  <div class="card">
    <div class="card-label">Sub-modelos de features</div>
    <table class="tbl">
      <tr><th>Sub-modelo</th><th>N</th><th>Composicao</th></tr>
      <tr><td>SINAN-only</td><td>110</td><td>Epidemiologicas</td></tr>
      <tr><td>INMET bruto</td><td>122</td><td>+ clima brutas</td></tr>
      <tr><td class="bld">INMET enriq.</td><td class="bld">266</td><td class="bld">+ features derivadas</td></tr>
    </table>
  </div>

  <!-- Resultados -->
  <div class="card">
    <div class="card-label">Resultados — Regressao</div>
    <div class="hl">
      <p>Variante B supera A com ΔR²=+0.0155 (p &lt; 0.001).
      Decay (C) nao melhora sobre B.</p>
    </div>
    <table class="tbl" style="margin-top:.6rem">
      <tr><th>Var</th><th>R² (log)</th><th>RMSE</th><th>MAE</th></tr>
      <tr><td>A</td><td>0.8723</td><td>2,063</td><td>253</td></tr>
      <tr><td class="bld">B</td><td class="bld">0.8878</td><td class="bld">2,010</td><td class="bld">246</td></tr>
      <tr><td>C</td><td>0.8870</td><td>1,945</td><td>237</td></tr>
    </table>
  </div>

  <!-- Classificacao -->
  <div class="card">
    <div class="card-label">Classificacao de risco</div>
    <p style="margin-bottom:.5rem">4 niveis baseados em percentis historicos:</p>
    <div style="display:flex;gap:.3rem;margin-bottom:.7rem">
      <span class="badge bb">Baixo</span>
      <span class="badge bm">Moderado</span>
      <span class="badge ba">Alto</span>
      <span class="badge bs">Surto</span>
    </div>
    <div class="mrow">
      <div class="mc"><div class="v">0.578</div><div class="k">F1 macro</div></div>
      <div class="mc"><div class="v">0.830</div><div class="k">AUC</div></div>
      <div class="mc"><div class="v">0.725</div><div class="k">F1 Surto</div></div>
    </div>
    <p style="font-size:.72rem;color:#98a2b3;margin-top:.3rem">Var B — INMET enriquecido</p>
  </div>

  <!-- Validacao -->
  <div class="card">
    <div class="card-label">Validacao</div>
    <ul class="vl">
      <li><b>Walk-forward</b> — treino expandido, teste ano a ano (2023→2026)</li>
      <li><b>Bootstrap IC 95%</b> — 1000 reamostragens por modelo</li>
      <li><b>Significancia</b> — B vs A: p = 8.29×10⁻⁴⁰</li>
    </ul>
  </div>

  <!-- Tecnologias -->
  <div class="card">
    <div class="card-label">Stack tecnologico</div>
    <div class="techs">
      <span class="chip">XGBoost</span>
      <span class="chip">Python</span>
      <span class="chip">Streamlit</span>
      <span class="chip">Plotly</span>
      <span class="chip">Pandas</span>
      <span class="chip">scikit-learn</span>
      <span class="chip">NumPy</span>
      <span class="chip">SHAP</span>
    </div>
    <p style="margin-top:.6rem">Dados: <b>SINAN</b> (notificacoes epidemiologicas) e <b>INMET</b> (estacoes meteorologicas).</p>
  </div>

  <!-- Limitacoes -->
  <div class="card full">
    <div class="card-label">Limitacoes conhecidas</div>
    <div class="warn"><ul>
      <li>INMET Silver ausente para 2009, 2020 e 2022 reduz cobertura climatica</li>
      <li>Variante A tem cobertura INMET de ~48% (vs ~90% original) devido a falta de 2020/2022</li>
      <li>2026 e ano parcial (17 semanas epidemiologicas no corte)</li>
      <li>Assume estabilidade da distribuicao temporal entre periodos de treino e teste</li>
    </ul></div>
  </div>

</div>
</body></html>"""
        st.components.v1.html(about_html, height=1500, scrolling=False)


if __name__ == "__main__":
    main()

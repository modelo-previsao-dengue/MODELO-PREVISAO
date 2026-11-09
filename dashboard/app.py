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
CACHE_DIR = Path(__file__).resolve().parent / "data"
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&display=swap');

:root {
  --font: 'DM Sans', -apple-system, sans-serif;
  --bg: #f5f6f8;
  --card: #ffffff;
  --border: #e4e7ec;
  --border-light: #f2f4f7;
  --text: #101828;
  --muted: #98a2b3;
  --sub: #667085;
  --ink: #344054;
  --primary: #5046e5;
  --primary-hover: #4338ca;
  --primary-soft: #eef2ff;
  --primary-muted: #818cf8;
  --surface: #f9fafb;
  --success: #22c55e;
  --success-soft: #f0fdf4;
  --warning: #eab308;
  --warning-soft: #fefce8;
  --danger: #ef4444;
  --danger-soft: #fef2f2;
  --orange: #f97316;
  --orange-soft: #fff7ed;
  --sp-1: .25rem;
  --sp-2: .5rem;
  --sp-3: .75rem;
  --sp-4: 1rem;
  --sp-5: 1.25rem;
  --sp-6: 1.5rem;
  --sp-8: 2rem;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 14px;
  --radius-xl: 18px;
  --shadow-sm: 0 1px 2px rgba(16,24,40,.05);
  --shadow-md: 0 4px 16px rgba(16,24,40,.06);
  --shadow-lg: 0 8px 32px rgba(16,24,40,.08);
  --shadow-xl: 0 12px 48px rgba(16,24,40,.10);
  --transition: all .2s ease;
}

/* ─ Base ─ */
html, body, [class*="css"], .stApp {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
  font-size: 15px !important;
}
#MainMenu, footer { visibility: hidden !important; height: 0 !important }
header[data-testid="stHeader"] {
  background: transparent !important; border: none !important;
  box-shadow: none !important;
}
[data-testid="stHeader"] [data-testid="stStatusWidget"],
[data-testid="stHeader"] .stDeployButton,
.stDeployButton, [data-testid="stAppDeployButton"],
button[kind="header"], .stAppDeployButton {
  display: none !important; visibility: hidden !important;
  width: 0 !important; height: 0 !important; overflow: hidden !important;
}
.block-container {
  padding: 1rem 2.5rem 3rem !important;
  max-width: 1440px;
}

/* ─ Typography ─ */
.stMarkdown p {
  font-size: .92rem !important; line-height: 1.65 !important;
}
.dc-h1 {
  font-size: 1.3rem; font-weight: 800; color: var(--text);
  letter-spacing: -.03em; line-height: 1.2; margin: 0;
}
.dc-h1 span { color: var(--primary); }
.dc-tagline {
  font-size: .72rem; color: var(--muted); font-weight: 500;
  letter-spacing: -.01em;
}
.dc-intro {
  font-size: .92rem; color: var(--sub); line-height: 1.7;
  max-width: 680px; margin-bottom: var(--sp-4);
}
.dc-intro b { color: var(--ink); }
.dc-subtle {
  font-size: .85rem; color: var(--muted); line-height: 1.6;
  margin-bottom: var(--sp-3); max-width: 680px;
}

/* ─ Section Header ─ */
.sec-header {
  font-size: .78rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: .06em; color: var(--sub); margin: var(--sp-8) 0 var(--sp-3);
  display: flex; align-items: center; gap: var(--sp-2);
}
.sec-header::before {
  content: ''; width: 18px; height: 3px;
  background: var(--primary); border-radius: 2px; flex-shrink: 0;
}

/* ─ Cards ─ */
.dc-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: var(--sp-5) var(--sp-6);
  box-shadow: var(--shadow-sm); margin-bottom: var(--sp-3);
  transition: var(--transition);
}
.dc-card:hover {
  box-shadow: var(--shadow-md);
}
.dc-card-title {
  font-size: .72rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); margin-bottom: var(--sp-3);
  display: flex; align-items: center; gap: .4rem;
}
.dc-card-title::before {
  content: ''; width: 12px; height: 2px;
  background: var(--primary); border-radius: 1px;
}

/* ─ Chart / DataFrame card wrappers (CSS-only) ─ */
[data-testid="stPlotlyChart"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: .6rem .8rem .2rem;
  box-shadow: var(--shadow-sm); margin-bottom: var(--sp-3);
  transition: var(--transition);
}
[data-testid="stPlotlyChart"]:hover {
  box-shadow: var(--shadow-md);
}
[data-testid="stDataFrame"] {
  background: var(--card); padding: var(--sp-2);
  margin-bottom: var(--sp-3);
}
/* remove card from map (already transparent bg) */
[data-testid="stPlotlyChart"]:has(.choroplethmapbox) {
  border: none; box-shadow: none; padding: 0;
  background: transparent;
}
[data-testid="stPlotlyChart"]:has(.choroplethmapbox):hover {
  box-shadow: none;
}

/* ─ KPI Cards ─ */
.kpi-card {
  border-radius: var(--radius-lg); padding: 1.15rem 1.3rem;
  position: relative; overflow: hidden;
  transition: var(--transition); cursor: default;
}
.kpi-card:hover {
  transform: scale(1.01); box-shadow: var(--shadow-md);
}
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: var(--radius-lg) var(--radius-lg) 0 0; }
.kpi-value { font-size: 2.2rem; font-weight: 800; letter-spacing: -.04em; line-height: 1; }
.kpi-label { font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--sub); margin-top: .35rem; }

/* ─ Header ─ */
.dc-header {
  display: flex; align-items: center; gap: var(--sp-4);
  padding-bottom: var(--sp-4); margin-bottom: var(--sp-2);
  border-bottom: 1px solid var(--border-light);
}
.dc-header-brand {
  display: flex; align-items: center; gap: var(--sp-3);
}
.dc-header-badge {
  font-size: .65rem; font-weight: 600; color: var(--primary);
  background: var(--primary-soft); padding: .2rem .6rem;
  border-radius: var(--radius-sm); border: 1px solid #c7d2fe;
  white-space: nowrap;
}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {
  background: #fafaf9 !important;
  border-right: none !important;
}
section[data-testid="stSidebar"] hr {
  border: none !important; height: 1px !important;
  background: #e7e5e4 !important; margin: var(--sp-3) 0 !important;
}
.sb-section-label {
  font-size: .6rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: .08em; color: #a8a29e; margin-bottom: var(--sp-2);
  display: flex; align-items: center; gap: .35rem;
}
section[data-testid="stSidebar"] .stSelectbox label {
  font-size: .7rem !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: .06em !important;
  color: #78716c !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
  font-size: .85rem !important;
  border: 1.5px solid #e7e5e4 !important; background: #fff !important;
  border-radius: 10px !important; color: #1c1917 !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
  border-color: #a8a29e !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(80,70,229,.1) !important;
}
section[data-testid="stSidebar"] .stCaption p {
  font-size: .72rem !important; color: #a8a29e !important;
  font-style: italic !important;
}
.sb-stat {
  background: rgba(255,255,255,.85); backdrop-filter: blur(8px);
  border: 1px solid rgba(231,229,228,.7); border-radius: 10px;
  padding: .5rem .55rem; text-align: center;
  position: relative; overflow: hidden;
  box-shadow: 0 1px 2px rgba(28,25,23,.04);
  transition: var(--transition);
}
.sb-stat:hover { box-shadow: 0 2px 8px rgba(28,25,23,.08); }
.sb-stat-bar { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.sb-stat-value { font-size: 1rem; font-weight: 800; color: #1c1917; letter-spacing: -.02em; }
.sb-stat-label { font-size: .58rem; font-weight: 600; color: #a8a29e; margin-top: .05rem; text-transform: uppercase; letter-spacing: .04em; }
.sb-footer {
  font-size: .62rem; color: #d6d3d1; font-weight: 500;
  text-align: center; padding: var(--sp-4) 0 var(--sp-2);
  border-top: 1px solid #e7e5e4; margin-top: var(--sp-4);
  letter-spacing: .02em;
}

/* ─ Region cards ─ */
.region-header {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 1.1rem 1.4rem;
  margin-bottom: var(--sp-4); display: flex;
  align-items: center; gap: .8rem; flex-wrap: wrap;
  box-shadow: var(--shadow-sm);
}
.region-name { font-size: 1.08rem; font-weight: 800; color: var(--text); letter-spacing: -.02em; }
.region-period { font-size: .72rem; color: var(--muted); font-weight: 500; }
.risk-badge {
  padding: .2rem .6rem; border-radius: 6px; font-size: .65rem;
  font-weight: 700; display: inline-flex; align-items: center;
}
.region-metric-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: var(--sp-4) var(--sp-5);
  text-align: center; box-shadow: var(--shadow-sm);
  transition: var(--transition);
}
.region-metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.region-metric-value { font-size: 1.4rem; font-weight: 800; color: var(--text); letter-spacing: -.03em; }
.region-metric-label { font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin-top: .2rem; }
.region-metric-accent { font-size: 1.4rem; font-weight: 800; color: var(--primary); letter-spacing: -.03em; }

/* ─ Explore card ─ */
.explore-card {
  background: linear-gradient(135deg, var(--primary-soft), #f5f3ff);
  border: 1px solid #c7d2fe; border-radius: var(--radius-lg);
  padding: var(--sp-6) 1.8rem; margin: var(--sp-4) 0;
}
.explore-title { font-size: 1rem; font-weight: 700; color: #1e1b4b; margin-bottom: .3rem; }
.explore-desc { font-size: .88rem; color: var(--sub); margin-bottom: 0; }

/* ─ Risk mini-card ─ */
.risk-mini {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: .8rem 1rem;
  margin-bottom: var(--sp-2); box-shadow: var(--shadow-sm);
  transition: var(--transition);
}
.risk-mini:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.risk-mini-name { font-size: .85rem; font-weight: 700; color: var(--text); margin-bottom: .3rem; }
.risk-mini-meta { font-size: .78rem; color: var(--muted); }

/* ─ Sobre (About) ─ */
.about-hero {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
  border-radius: var(--radius-xl); padding: 2.2rem 2.5rem;
  margin-bottom: var(--sp-6); position: relative; overflow: hidden;
}
.about-hero::after {
  content: ''; position: absolute; top: -40%; right: -10%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(255,255,255,.06) 0%, transparent 70%);
  border-radius: 50%;
}
.about-hero-title { font-size: 1.6rem; font-weight: 800; color: #fff; letter-spacing: -.04em; margin-bottom: .5rem; line-height: 1.2; }
.about-hero-desc { font-size: .86rem; color: rgba(255,255,255,.75); max-width: 560px; line-height: 1.7; }
.about-hero-desc b { color: #c7d2fe; }
.about-stats { display: flex; gap: .7rem; margin-top: 1.3rem; flex-wrap: wrap; }
.about-stat {
  background: rgba(255,255,255,.1); backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,.12); border-radius: var(--radius-md);
  padding: .7rem 1rem; text-align: center; flex: 1; min-width: 80px;
}
.about-stat-n { font-size: 1.2rem; font-weight: 800; color: #fff; letter-spacing: -.02em; }
.about-stat-l { font-size: .52rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: rgba(255,255,255,.5); margin-top: .1rem; }

.about-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-4); margin-bottom: var(--sp-4); }
.about-grid .dc-card.full { grid-column: 1 / -1; }

.pipe { display: flex; align-items: center; gap: .3rem; flex-wrap: wrap; margin: .6rem 0; }
.pipe-step { padding: .35rem .7rem; border-radius: var(--radius-sm); font-size: .7rem; font-weight: 700; color: var(--sub); background: var(--border-light); }
.pipe-step.ml { background: var(--primary-soft); color: var(--primary-hover); }
.pipe-arrow { color: #d0d5dd; font-size: .75rem; }
.about-hl {
  background: linear-gradient(135deg, var(--primary-soft), #e8defb);
  border-radius: var(--radius-md); padding: .9rem 1.2rem;
  margin: .7rem 0; border-left: 4px solid var(--primary);
}
.about-hl p { color: #3730a3; font-weight: 600; font-size: .82rem; margin: 0; }
.about-tbl { width: 100%; border-collapse: collapse; font-size: .78rem; }
.about-tbl th {
  text-align: left; font-size: .58rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  padding: .45rem .6rem; border-bottom: 2px solid var(--border);
}
.about-tbl td { padding: .45rem .6rem; border-bottom: 1px solid var(--border-light); color: var(--sub); }
.about-tbl tr:hover td { background: var(--surface); }
.about-tbl .bld { font-weight: 700; color: var(--text); }
.about-badge { display: inline-flex; padding: .15rem .45rem; border-radius: 5px; font-size: .62rem; font-weight: 700; margin: 0 .1rem; }
.about-badge.surto { background: var(--danger-soft); color: #dc2626; border: 1px solid #fecaca; }
.about-badge.alto { background: var(--orange-soft); color: #ea580c; border: 1px solid #fed7aa; }
.about-badge.moderado { background: var(--warning-soft); color: #a16207; border: 1px solid #fef08a; }
.about-badge.baixo { background: var(--success-soft); color: #16a34a; border: 1px solid #bbf7d0; }
.about-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: .6rem 0; }
.about-mc {
  text-align: center; padding: .6rem; background: var(--surface);
  border-radius: 10px; border: 1px solid var(--border-light);
}
.about-mc-v { font-size: 1rem; font-weight: 800; color: var(--primary); }
.about-mc-k { font-size: .52rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-top: .1rem; }
.about-vl { list-style: none; padding: 0; margin: .5rem 0; }
.about-vl li {
  padding: .45rem 0; border-bottom: 1px solid var(--border-light);
  font-size: .82rem; color: var(--sub); display: flex;
  align-items: flex-start; gap: .5rem;
}
.about-vl li::before { content: '✓'; color: var(--primary); font-weight: 800; font-size: .7rem; margin-top: .1rem; }
.about-chip {
  background: var(--border-light); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: .4rem .8rem;
  font-size: .72rem; font-weight: 700; color: var(--ink);
  display: inline-flex;
}
.about-warn {
  background: #fffbeb; border: 1px solid #fde68a;
  border-radius: var(--radius-md); padding: .8rem 1.1rem;
}
.about-warn ul { padding-left: 1rem; margin: 0; }
.about-warn li { color: #92400e; font-size: .8rem; margin-bottom: .25rem; }
.about-credits {
  background: linear-gradient(135deg, #fafafe, var(--primary-soft));
  border: 1px solid #c7d2fe; border-radius: var(--radius-lg);
  padding: 1.4rem 1.5rem;
}
.about-author-name { font-size: .95rem; font-weight: 800; color: #1e1b4b; margin-bottom: .15rem; }
.about-author-role { font-size: .78rem; color: var(--primary-hover); font-weight: 600; }

/* ─ Divider ─ */
.dc-divider {
  border: none; height: 1px; background: var(--border-light);
  margin: var(--sp-8) 0;
}

/* ─ Metrics ─ */
[data-testid="stMetricValue"] {
  font-size: 1.15rem !important; font-weight: 700 !important; color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  font-size: .72rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .04em !important;
  color: var(--muted) !important;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
  gap: .4rem !important; background: var(--surface) !important;
  border-bottom: none !important; padding: .4rem !important;
  border-radius: var(--radius-md) !important;
}
.stTabs [data-baseweb="tab-list"]::before,
.stTabs [data-baseweb="tab-list"]::after { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab"] {
  border-radius: 9px !important; font-weight: 600 !important; font-size: .88rem !important;
  padding: .55rem 1.3rem !important; color: var(--sub) !important;
  border: none !important; background: transparent !important;
  letter-spacing: -.01em !important; transition: var(--transition) !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--ink) !important; background: rgba(80,70,229,.06) !important;
}
.stTabs [aria-selected="true"] {
  color: #fff !important; background: var(--primary) !important;
  font-weight: 700 !important; box-shadow: 0 2px 8px rgba(80,70,229,.25) !important;
}

/* ─ Components ─ */
.stSelectbox > div > div {
  border-radius: 10px !important; border-color: var(--border) !important;
  font-size: .82rem !important;
}
[data-testid="stDataFrame"] {
  border-radius: var(--radius-lg) !important; overflow: hidden;
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
  gap: .3rem !important; background: var(--border-light); padding: .25rem .3rem;
  border-radius: 10px; display: inline-flex !important;
}
div[data-testid="stRadio"] label {
  font-size: .72rem !important; font-weight: 600 !important;
  padding: .35rem .85rem !important; border-radius: var(--radius-sm) !important;
  color: var(--sub) !important; cursor: pointer !important;
  transition: all .15s !important; white-space: nowrap !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
  background: #fff !important; color: var(--text) !important;
  box-shadow: 0 1px 4px rgba(16,24,40,.08) !important; font-weight: 700 !important;
}
div[data-testid="stRadio"] label input,
div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] input,
div[data-testid="stRadio"] [role="radiogroup"] input[type="radio"],
div[data-testid="stRadio"] label > div:first-child {
  display: none !important; width: 0 !important; height: 0 !important;
  opacity: 0 !important; position: absolute !important;
}
div[data-testid="stRadio"] label p { margin: 0 !important }
div[data-testid="stRadio"] > label:first-child { display: none !important }

/* ─ Animations ─ */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(12px) }
  to { opacity: 1; transform: translateY(0) }
}
.kpi-card,
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
.dc-card {
  animation: fadeInUp .4s ease both;
}
.stColumns:first-of-type [data-testid="column"]:nth-child(1) .kpi-card { animation-delay: 0s }
.stColumns:first-of-type [data-testid="column"]:nth-child(2) .kpi-card { animation-delay: .06s }
.stColumns:first-of-type [data-testid="column"]:nth-child(3) .kpi-card { animation-delay: .12s }
.stColumns:first-of-type [data-testid="column"]:nth-child(4) .kpi-card { animation-delay: .18s }
[data-testid="stPlotlyChart"] { animation-delay: .25s }
.dc-card { animation-delay: .15s }
</style>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                    COMPONENTS                           ║
# ╚══════════════════════════════════════════════════════════╝

PLOTLY_FONT = "DM Sans, -apple-system, sans-serif"

_RISK_BG = {"Surto": "var(--danger-soft)", "Alto": "var(--orange-soft)",
            "Moderado": "var(--warning-soft)", "Baixo": "var(--success-soft)"}
_RISK_FG = {"Surto": "#dc2626", "Alto": "#ea580c", "Moderado": "#a16207", "Baixo": "#16a34a"}
_RISK_BORDER = {"Surto": "#fecaca", "Alto": "#fed7aa", "Moderado": "#fef08a", "Baixo": "#bbf7d0"}


def plo(h=380, **kw):
    d = dict(
        height=h, template="plotly_white",
        font=dict(family=PLOTLY_FONT, size=13, color="#667085"),
        margin=dict(l=48, r=16, t=32, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12, family=PLOTLY_FONT)),
    )
    d.update(kw)
    return d


def sec(title):
    st.markdown(f'<p class="sec-header">{title}</p>', unsafe_allow_html=True)


def kpi_row(items):
    accent_map = {"#ef4444": "var(--danger-soft)", "#f97316": "var(--orange-soft)",
                  "#eab308": "var(--warning-soft)", "#22c55e": "var(--success-soft)"}
    cols = st.columns(len(items))
    for col, (val, label, color) in zip(cols, items):
        bg = accent_map.get(color, "var(--surface)")
        col.markdown(
            f'<div class="kpi-card" style="background:{bg};border:1px solid {color}22">'
            f'<div class="kpi-accent" style="background:{color}"></div>'
            f'<div class="kpi-value" style="color:{color}">{val:,}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True)


def card_title(title):
    st.markdown(f'<div class="dc-card-title">{title}</div>', unsafe_allow_html=True)


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

def _has_train(var):
    return (DATA / f"splits_{var.lower()}" / "train.parquet").exists()

@st.cache_data
def load_cached_predictions(var, sub):
    p = CACHE_DIR / f"predictions_{var}_{sub}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

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

    _logo_dir = Path(__file__).resolve().parent
    _logo_full = str(_logo_dir / "logo_full.png")
    _logo_icon = str(_logo_dir / "logo_icon.png")
    st.logo(_logo_full, icon_image=_logo_icon, size="medium")

    # ── Sidebar ──
    _var_labels = {
        "B": "Serie completa (22 anos)",
        "C": "Serie completa + Decay",
        "A": "Apenas recente (4 anos)",
    }
    _sub_labels = {
        "inmet_enriquecido": "Epidemiologico + Clima detalhado",
        "inmet_bruto": "Epidemiologico + Clima basico",
        "sinan_only": "Somente epidemiologico",
    }
    _var_desc = {
        "B": "Treinado com dados de 2001 a 2022 — maior volume historico",
        "C": "Treinado com 2001–2022, priorizando dados mais recentes",
        "A": "Treinado apenas com 2019–2022 — referencia rapida",
    }
    _sub_desc = {
        "inmet_enriquecido": "266 variaveis — dados de saude + clima com indicadores derivados",
        "inmet_bruto": "122 variaveis — dados de saude + clima bruto",
        "sinan_only": "110 variaveis — apenas dados epidemiologicos (sem clima)",
    }
    with st.sidebar:
        st.markdown('<div class="sb-section-label">Configuracao do modelo</div>',
                    unsafe_allow_html=True)
        var = st.selectbox("Periodo", ["B","C","A"],
                           format_func=lambda v: _var_labels[v],
                           help="Janela temporal dos dados de treinamento")
        st.caption(_var_desc[var])
        sub = st.selectbox("Dados", ["inmet_enriquecido","inmet_bruto","sinan_only"],
                           format_func=lambda s: _sub_labels[s],
                           help="Conjunto de variaveis usadas pelo modelo")
        st.caption(_sub_desc[sub])

    live_mode = _has_train(var)

    if live_mode:
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
    else:
        preds = load_cached_predictions(var, sub)
        if preds is None:
            st.error(f"Previsoes pre-computadas nao encontradas para {var}/{sub}.")
            return

    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="sb-section-label">Filtros</div>',
                    unsafe_allow_html=True)
        _fc, _fu = st.columns(2)
        with _fc:
            all_anos = sorted(preds["ano"].unique())
            sel_ano = st.selectbox("Ano", ["Todos"]+[str(a) for a in all_anos], index=0)
        with _fu:
            ufs = sorted(preds["uf"].dropna().unique())
            sel_uf = st.selectbox("UF", ["Todos"]+list(ufs))

        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;margin-top:.5rem">'
            f'<div class="sb-stat"><div class="sb-stat-bar" style="background:linear-gradient(90deg,#5046e5,#818cf8)"></div>'
            f'<div class="sb-stat-value">{len(preds):,}</div>'
            f'<div class="sb-stat-label">Previsoes</div></div>'
            f'<div class="sb-stat"><div class="sb-stat-bar" style="background:linear-gradient(90deg,#818cf8,#a78bfa)"></div>'
            f'<div class="sb-stat-value" style="font-size:.88rem">{all_anos[0]}–{all_anos[-1]}</div>'
            f'<div class="sb-stat-label">Periodo</div></div>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr;gap:.4rem;margin-top:.4rem">'
            f'<div class="sb-stat"><div class="sb-stat-bar" style="background:linear-gradient(90deg,#a78bfa,#c4b5fd)"></div>'
            f'<div class="sb-stat-value">137 mesorregioes &middot; {len(preds["uf"].dropna().unique())} UFs</div>'
            f'<div class="sb-stat-label">Cobertura nacional</div></div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sb-footer">TCC &middot; UnB FGA &middot; 2024–2025</div>',
                    unsafe_allow_html=True)

    # ── Header ──
    _sub_display = _sub_labels[sub].split(" + ")[-1] if " + " in _sub_labels[sub] else _sub_labels[sub]
    st.markdown(
        f'<div class="dc-header">'
        f'<div class="dc-header-brand">'
        f'<div><div class="dc-h1">Dengue<span>Cast</span></div>'
        f'<div class="dc-tagline">Sistema preditivo de vigilancia epidemiologica</div></div>'
        f'</div>'
        f'<div class="dc-header-badge">{_var_labels[var]} &middot; {_sub_display}</div>'
        f'</div>',
        unsafe_allow_html=True)

    view = preds.copy()
    if sel_ano != "Todos":
        view = view[view["ano"]==int(sel_ano)]
    if sel_uf != "Todos":
        view = view[view["uf"]==sel_uf]

    t1, t2, t3, t4 = st.tabs(["Painel", "Regioes", "Analise", "Sobre"])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — PAINEL (Alertas)
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
                hovertemplate="<b>%{y}</b><br>Risco medio: %{x:.2f}<extra></extra>",
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
                card_title("Top alertas")
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
            line=dict(color="#dc2626", width=2.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,.04)",
            hovertemplate="<b>%{x}</b><br>Surto: %{y} mesorregioes<extra></extra>"))
        fig_evo.add_trace(go.Scatter(
            x=wk["per"], y=wk["alto"], name="Alto",
            line=dict(color="#ea580c", width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Alto: %{y} mesorregioes<extra></extra>"))
        fig_evo.update_layout(**plo(h=280,
                                     xaxis=dict(showgrid=False, title=""),
                                     yaxis=dict(gridcolor="#f2f4f7", gridwidth=1, title="")))
        st.plotly_chart(fig_evo, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 2 — REGIOES
    # ════════════════════════════════════════════════════════════
    with t2:
        mesos = sorted(view["cod_mesorregiao"].unique())
        opts = {m: mn(m, names) for m in mesos}
        sel_meso = st.selectbox("Buscar mesorregiao", mesos,
                                format_func=lambda x: opts[x], key="meso",
                                index=None, placeholder="Digite ou selecione uma mesorregiao...")
        if sel_meso is None:
            _latest_r = view.sort_values(["ano","semana_epidemiologica"]).groupby("cod_mesorregiao").last().reset_index()
            _top_risk = _latest_r.nlargest(8, "pred_risco").head(8)
            st.markdown(
                '<div class="explore-card">'
                '<p class="explore-title">Explore uma mesorregiao</p>'
                '<p class="explore-desc">'
                'Selecione uma das 137 mesorregioes no campo acima para ver previsoes detalhadas, '
                'serie temporal e probabilidades de risco por semana epidemiologica.</p>'
                '</div>',
                unsafe_allow_html=True)
            sec("Mesorregioes com maior risco atual")
            _cols = st.columns(4)
            for i, (_, row) in enumerate(_top_risk.iterrows()):
                _rl = row["pred_label"]
                _nm = mn(row["cod_mesorregiao"], names)
                _cols[i % 4].markdown(
                    f'<div class="risk-mini">'
                    f'<p class="risk-mini-name">{_nm}</p>'
                    f'<span class="risk-badge" style="background:{_RISK_BG[_rl]};color:{_RISK_FG[_rl]};'
                    f'border:1px solid {_RISK_BORDER[_rl]};font-size:.72rem;padding:.15rem .5rem">{_rl}</span>'
                    f'&nbsp;<span class="risk-mini-meta">P(Surto) {row["p_surto"]*100:.0f}%</span></div>',
                    unsafe_allow_html=True)

        if sel_meso is not None:
            df_m = preds[preds["cod_mesorregiao"]==sel_meso].sort_values(
                ["ano","semana_epidemiologica"]).copy()
            last = df_m.iloc[-1]
            rl = last["pred_label"]

            st.markdown(
                f'<div class="region-header">'
                f'<span class="region-name">{opts[sel_meso]}</span>'
                f'<span class="region-period">{int(last["ano"])}-S{int(last["semana_epidemiologica"]):02d}</span>'
                f'<span class="risk-badge" style="background:{_RISK_BG[rl]};color:{_RISK_FG[rl]};'
                f'border:1px solid {_RISK_BORDER[rl]}">{rl}</span></div>',
                unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value">{last["notificacoes"]:,.0f}</div>'
                    f'<div class="region-metric-label">Casos atuais</div></div>',
                    unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-accent">{last["pred_notif"]:,.0f}</div>'
                    f'<div class="region-metric-label">Previsao 4 semanas</div></div>',
                    unsafe_allow_html=True)
            with mc3:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value" style="color:{_RISK_FG[rl]}">{rl}</div>'
                    f'<div class="region-metric-label">Nivel de risco</div></div>',
                    unsafe_allow_html=True)
            with mc4:
                psurto_pct = last["p_surto"] * 100
                bar_color = "#ef4444" if psurto_pct > 50 else "#f97316" if psurto_pct > 25 else "#eab308" if psurto_pct > 10 else "#22c55e"
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value">{psurto_pct:.1f}%</div>'
                    f'<div style="background:#f2f4f7;border-radius:4px;height:4px;margin-top:.4rem;overflow:hidden">'
                    f'<div style="background:{bar_color};height:100%;width:{min(psurto_pct, 100):.0f}%;'
                    f'border-radius:4px;transition:width .3s ease"></div></div>'
                    f'<div class="region-metric-label">Chance de surto</div></div>',
                    unsafe_allow_html=True)

            sec("Serie temporal")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_m["periodo"], y=df_m["notificacoes_t4"], name="Real (t+4)",
                line=dict(color="#5046e5", width=2.5),
                hovertemplate="<b>%{x}</b><br>Real: %{y:,.0f} casos<extra></extra>"))
            fig_ts.add_trace(go.Scatter(
                x=df_m["periodo"], y=df_m["pred_notif"], name="Previsto (t+4)",
                line=dict(color="#dc2626", width=2, dash="dash"),
                hovertemplate="<b>%{x}</b><br>Previsto: %{y:,.0f} casos<extra></extra>"))
            fig_ts.add_trace(go.Scatter(
                x=df_m["periodo"], y=df_m["notificacoes"], name="Atual",
                line=dict(color="#d0d5dd", width=1),
                hovertemplate="<b>%{x}</b><br>Atual: %{y:,.0f} casos<extra></extra>"))
            fig_ts.update_layout(**plo(h=360,
                                        xaxis=dict(showgrid=False, title=""),
                                        yaxis=dict(gridcolor="#f2f4f7", title="")))
            st.plotly_chart(fig_ts, use_container_width=True)

            sec("Probabilidade de risco")
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=df_m["periodo"], y=df_m["p_surto"], name="P(Surto)",
                marker_color="rgba(220,38,38,.45)",
                hovertemplate="<b>%{x}</b><br>P(Surto): %{y:.1%}<extra></extra>"))
            fig_p.add_trace(go.Bar(
                x=df_m["periodo"], y=df_m["p_alto"], name="P(Alto)",
                marker_color="rgba(234,88,12,.25)",
                hovertemplate="<b>%{x}</b><br>P(Alto): %{y:.1%}<extra></extra>"))
            fig_p.update_layout(**plo(h=240, barmode="overlay",
                                       yaxis=dict(range=[0,1], gridcolor="#f2f4f7", title=""),
                                       xaxis=dict(showgrid=False, title="")))
            st.plotly_chart(fig_p, use_container_width=True)

            from sklearn.metrics import r2_score, mean_absolute_error
            act = df_m["notificacoes_t4"].values
            pred = df_m["pred_notif"].values
            mk = ~np.isnan(act) & ~np.isnan(pred) & (act>=0)
            if mk.sum() > 5:
                sec("Desempenho do modelo nesta regiao")
                r2_val = r2_score(np.log1p(act[mk]), np.log1p(pred[mk]))
                mae_val = mean_absolute_error(act[mk], pred[mk])
                acc_val = (df_m['pred_risco'].values==df_m['risco_surto_t4'].values).mean()
                mm1, mm2, mm3 = st.columns(3)
                with mm1:
                    st.markdown(
                        f'<div class="region-metric-card">'
                        f'<div class="region-metric-accent">{r2_val:.3f}</div>'
                        f'<div class="region-metric-label">Qualidade (R²)</div></div>',
                        unsafe_allow_html=True)
                with mm2:
                    st.markdown(
                        f'<div class="region-metric-card">'
                        f'<div class="region-metric-value">{mae_val:,.0f}</div>'
                        f'<div class="region-metric-label">Erro medio (casos)</div></div>',
                        unsafe_allow_html=True)
                with mm3:
                    st.markdown(
                        f'<div class="region-metric-card">'
                        f'<div class="region-metric-accent">{acc_val:.0%}</div>'
                        f'<div class="region-metric-label">Acerto no risco</div></div>',
                        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # TAB 3 — ANALISE (Temporal + Modelo merged)
    # ════════════════════════════════════════════════════════════
    with t3:
        st.markdown(
            '<p class="dc-intro">'
            'Analise completa do desempenho do modelo. Compara previsoes com dados reais ao longo do tempo '
            'e detalha as metricas de qualidade do modelo de inteligencia artificial selecionado.</p>',
            unsafe_allow_html=True)

        # ── Previsão vs Realidade ──
        sec("Previsao vs Realidade")
        st.markdown(
            '<p class="dc-subtle">'
            'Cada ponto representa uma mesorregiao em uma semana. Pontos proximos a linha tracejada '
            'indicam previsao precisa. Escala logaritmica para comparar regioes de diferentes tamanhos.</p>',
            unsafe_allow_html=True)
        samp = view.sample(min(5000, len(view)), random_state=42)
        samp["log_real"] = np.log1p(samp["notificacoes_t4"])
        samp["log_pred"] = np.log1p(samp["pred_notif"])
        fig_sc = px.scatter(
            samp, x="log_real", y="log_pred",
            color="pred_label", color_discrete_map=RISK_CLR, opacity=.3,
            labels={"log_real":"Notificacoes reais (log)","log_pred":"Previsao do modelo (log)","pred_label":"Risco"})
        fig_sc.update_traces(marker=dict(size=5))
        mx = max(samp["log_real"].quantile(.99), samp["log_pred"].quantile(.99))
        fig_sc.add_trace(go.Scatter(
            x=[0,mx], y=[0,mx], mode="lines",
            line=dict(color="#d0d5dd", dash="dash", width=1.5), showlegend=False))
        fig_sc.update_layout(**plo(h=440,
                                    xaxis=dict(gridcolor="#f2f4f7"),
                                    yaxis=dict(gridcolor="#f2f4f7")))
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Evolução Nacional ──
        sec("Total de casos no Brasil por semana")
        st.markdown(
            '<p class="dc-subtle">'
            'Soma de todas as mesorregioes por semana. '
            'Linha solida = dados reais, tracejada = previsao do modelo.</p>',
            unsafe_allow_html=True)
        wk_agg = view.groupby("periodo").agg(
            real=("notificacoes_t4","sum"),
            prev=("pred_notif","sum")).reset_index().sort_values("periodo")
        fig_nat = go.Figure()
        fig_nat.add_trace(go.Scatter(
            x=wk_agg["periodo"], y=wk_agg["real"], name="Real",
            line=dict(color="#5046e5", width=2.5),
            hovertemplate="<b>%{x}</b><br>Real: %{y:,.0f}<extra></extra>"))
        fig_nat.add_trace(go.Scatter(
            x=wk_agg["periodo"], y=wk_agg["prev"], name="Previsto",
            line=dict(color="#dc2626", width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Previsto: %{y:,.0f}<extra></extra>"))
        fig_nat.update_layout(**plo(h=320,
                                     xaxis=dict(showgrid=False, title=""),
                                     yaxis=dict(gridcolor="#f2f4f7", title="")))
        st.plotly_chart(fig_nat, use_container_width=True)

        # ── Desempenho por Ano ──
        sec("Qualidade da previsao por ano")
        st.markdown(
            '<p class="dc-subtle">'
            'Desempenho do modelo em cada ano. Cores indicam se o ano foi usado para treino, '
            'validacao ou teste. Barras mais altas indicam melhor previsao.</p>',
            unsafe_allow_html=True)
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
            _split_labels = {"train":"Treino","val":"Validacao","test":"Teste"}
            df_yr["Tipo"] = df_yr["split"].map(_split_labels)
            fig_yr = px.bar(df_yr, x="Ano", y="R2", color="Tipo",
                            color_discrete_map={"Treino":"#818cf8","Validacao":"#fbbf24","Teste":"#f87171"},
                            labels={"R2":"Qualidade da previsao"},
                            text=df_yr["R2"].apply(lambda v: f"{v:.2f}"))
            fig_yr.update_traces(
                textposition="outside",
                textfont=dict(size=11, family=PLOTLY_FONT, color="#344054"),
                width=0.7)
            fig_yr.update_layout(**plo(h=400,
                                       xaxis=dict(showgrid=False, title="", dtick=1,
                                                  tickfont=dict(size=11)),
                                       yaxis=dict(gridcolor="#f2f4f7", title="Qualidade (R²)",
                                                  range=[0, 1.08]),
                                       bargap=0.15))
            st.plotly_chart(fig_yr, use_container_width=True)

        # ── Separador visual ──
        st.markdown('<hr class="dc-divider">', unsafe_allow_html=True)

        # ── Métricas do Modelo ──
        mk_key = f"{var}_{sub}"
        reg = reports.get("reg", {})
        if mk_key in reg.get("modelos", {}):
            m = reg["modelos"][mk_key]
            sec("Metricas do modelo selecionado")
            st.markdown(
                '<p class="dc-subtle">'
                '<b>Qualidade (R²)</b> indica o quanto o modelo consegue explicar os dados '
                '(mais perto de 1 = melhor). <b>Erro medio</b> indica quantas notificacoes '
                'o modelo erra em media.</p>',
                unsafe_allow_html=True)
            mk1, mk2, mk3, mk4 = st.columns(4)
            with mk1:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-accent">{m["R2_log"]:.4f}</div>'
                    f'<div class="region-metric-label">Qualidade (R²)</div></div>',
                    unsafe_allow_html=True)
            with mk2:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value">{m["RMSE"]:,.0f}</div>'
                    f'<div class="region-metric-label">Erro medio (RMSE)</div></div>',
                    unsafe_allow_html=True)
            with mk3:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value">{m["MAE"]:,.1f}</div>'
                    f'<div class="region-metric-label">Desvio medio (MAE)</div></div>',
                    unsafe_allow_html=True)
            with mk4:
                st.markdown(
                    f'<div class="region-metric-card">'
                    f'<div class="region-metric-value" style="font-size:1rem">{m["R2_log_CI95_lo"]:.4f} – {m["R2_log_CI95_hi"]:.4f}</div>'
                    f'<div class="region-metric-label">Intervalo confianca 95%</div></div>',
                    unsafe_allow_html=True)

        # ── Comparativo de Variantes ──
        sec("Comparativo de variantes")
        st.markdown(
            '<p class="dc-subtle">'
            'Comparacao entre as 3 estrategias de treinamento. '
            'Barras de erro mostram a margem de incerteza do resultado.</p>',
            unsafe_allow_html=True)
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
                            textfont=dict(size=11, family=PLOTLY_FONT),
                            hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"))
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
                            "Precisao geral": f"{mm['f1_macro']:.4f}",
                            "Capacidade de distincao": f"{mm['auc_macro_ovr']:.4f}",
                            "Acerto em surtos": f"{mm['f1_per_class']['surto']:.4f}",
                        })
                if rows:
                    card_title("Classificacao de risco")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Walk-Forward ──
        wf = reports.get("wf")
        if wf and "modelos" in wf:
            sec("Validacao Walk-Forward")
            st.markdown(
                '<p class="dc-subtle">'
                'Simulacao de uso real: o modelo e treinado com dados ate um ano e testado no seguinte, '
                'avancando progressivamente. Mostra como o modelo se comportaria em producao.</p>',
                unsafe_allow_html=True)
            wfr = []
            for v in ["A","B","C"]:
                k = f"{v}_inmet_enriquecido"
                if k in wf["modelos"]:
                    for i, fold in enumerate(wf["modelos"][k].get("folds",[]),1):
                        yr = fold["test_years"][0] if "test_years" in fold else "?"
                        wfr.append({"Variante":f"Var {v}","Periodo de teste":f"Teste {yr}","Qualidade":fold["R2_log"]})
            if wfr:
                fig_wf = px.line(pd.DataFrame(wfr), x="Periodo de teste", y="Qualidade", color="Variante",
                                 markers=True,
                                 color_discrete_map={"Var A":"#93c5fd","Var B":"#5046e5",
                                                     "Var C":"#a78bfa"})
                fig_wf.update_traces(
                    hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>")
                fig_wf.update_layout(**plo(h=300,
                                           xaxis=dict(showgrid=False),
                                           yaxis=dict(gridcolor="#f2f4f7", title="Qualidade (R²)")))
                st.plotly_chart(fig_wf, use_container_width=True)

        # ── SHAP ──
        shap_df = reports.get("shap")
        if shap_df is not None:
            shap_col = "shap_mean_abs_B" if "shap_mean_abs_B" in shap_df.columns else shap_df.columns[1]
            sec("Importancia das variaveis (SHAP)")
            st.markdown(
                '<p class="dc-subtle">'
                'Quais dados mais influenciam a previsao do modelo? '
                'Barras maiores indicam variaveis com maior peso na decisao.</p>',
                unsafe_allow_html=True)
            _feat_names = {
                "notificacoes_lag_1w": "Notificacoes (1 sem. atras)",
                "notificacoes_lag_2w": "Notificacoes (2 sem. atras)",
                "notificacoes_lag_3w": "Notificacoes (3 sem. atras)",
                "notificacoes_lag_4w": "Notificacoes (4 sem. atras)",
                "notificacoes_lag_6w": "Notificacoes (6 sem. atras)",
                "notificacoes_lag_8w": "Notificacoes (8 sem. atras)",
                "notificacoes_diff_1w": "Variacao semanal de casos",
                "notificacoes_pct_change_1w": "Variacao % semanal",
                "notificacoes_media_movel_2_4w": "Media movel 2-4 sem.",
                "notificacoes_media_movel_4_8w": "Media movel 4-8 sem.",
                "week_of_year_cos": "Sazonalidade (cosseno)",
                "week_of_year_sin": "Sazonalidade (seno)",
                "qt_confirmados_provaveis": "Casos confirmados/provaveis",
                "prop_confirmado_provavel": "Proporcao confirmados",
                "atraso_notificacao_medio_dias": "Atraso medio notificacao",
                "prop_raca_parda": "Proporcao raca parda",
                "prop_criterio_lab": "Proporcao criterio laboratorial",
            }
            n = st.slider("Quantidade de variaveis", 10, 50, 15, key="shap_n")
            top_n = shap_df.nlargest(n, shap_col).copy()
            top_n["nome"] = top_n["feature"].map(lambda f: _feat_names.get(f, f.replace("_"," ").title()))
            fig_sh = px.bar(top_n, x=shap_col, y="nome", orientation="h",
                            color=shap_col,
                            color_continuous_scale=["#e0e7ff","#5046e5"])
            fig_sh.update_traces(
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>")
            fig_sh.update_layout(**plo(h=max(280, n*24),
                                       yaxis=dict(autorange="reversed", title=""),
                                       xaxis=dict(title="Importancia (SHAP)", gridcolor="#f2f4f7"),
                                       coloraxis_showscale=False))
            st.plotly_chart(fig_sh, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — SOBRE (Native Streamlit)
    # ════════════════════════════════════════════════════════════
    with t4:
        # Hero
        st.markdown(
            '<div class="about-hero">'
            '<div class="about-hero-title">DengueCast</div>'
            '<p class="about-hero-desc">Sistema preditivo de surtos de dengue desenvolvido como '
            '<b>TCC em Engenharia de Software</b> na Universidade de Brasilia (UnB). '
            'Previsoes de notificacoes e risco de surto <b>4 semanas a frente</b> em '
            '<b>137 mesorregioes</b> brasileiras.</p>'
            '<div class="about-stats">'
            '<div class="about-stat"><div class="about-stat-n">180,703</div><div class="about-stat-l">Registros</div></div>'
            '<div class="about-stat"><div class="about-stat-n">2001–2026</div><div class="about-stat-l">Cobertura</div></div>'
            '<div class="about-stat"><div class="about-stat-n">137</div><div class="about-stat-l">Mesorregioes</div></div>'
            '<div class="about-stat"><div class="about-stat-n">18</div><div class="about-stat-l">Modelos</div></div>'
            '</div></div>',
            unsafe_allow_html=True)

        # Pipeline
        st.markdown(
            '<div class="about-grid">'
            '<div class="dc-card full">'
            '<div class="dc-card-title">Pipeline de Dados</div>'
            '<div class="pipe">'
            '<span class="pipe-step">SINAN</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step">Limpeza</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step">Feature Eng.</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step">INMET</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step">Merge</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step">Splits</span><span class="pipe-arrow">→</span>'
            '<span class="pipe-step ml">XGBoost</span>'
            '</div>'
            '<p style="font-size:.82rem;color:var(--sub);line-height:1.6">'
            'Features: lags (1–8 sem.), medias moveis, diferencas semanais, '
            'sazonalidade ciclica e variaveis climaticas (temperatura, precipitacao, umidade).</p>'
            '</div>'

            # Variantes
            '<div class="dc-card">'
            '<div class="dc-card-title">Variantes experimentais</div>'
            '<table class="about-tbl">'
            '<tr><th>Variante</th><th>Treino</th><th>Descricao</th></tr>'
            '<tr><td>A (Baseline)</td><td>2019–2022</td><td>4 anos recentes</td></tr>'
            '<tr><td class="bld">B (Completa)</td><td class="bld">2001–2022</td><td class="bld">Serie completa 22a</td></tr>'
            '<tr><td>C (Decay)</td><td>2001–2022</td><td>Completa + pesos exp.</td></tr>'
            '</table></div>'

            # Sub-modelos
            '<div class="dc-card">'
            '<div class="dc-card-title">Sub-modelos de features</div>'
            '<table class="about-tbl">'
            '<tr><th>Sub-modelo</th><th>N</th><th>Composicao</th></tr>'
            '<tr><td>SINAN-only</td><td>110</td><td>Epidemiologicas</td></tr>'
            '<tr><td>INMET bruto</td><td>122</td><td>+ clima brutas</td></tr>'
            '<tr><td class="bld">INMET enriq.</td><td class="bld">266</td><td class="bld">+ features derivadas</td></tr>'
            '</table></div>'

            # Resultados
            '<div class="dc-card">'
            '<div class="dc-card-title">Resultados — Regressao</div>'
            '<div class="about-hl"><p>Variante B supera A com ΔR²=+0.0155 (p &lt; 0.001). '
            'Decay (C) nao melhora sobre B.</p></div>'
            '<table class="about-tbl" style="margin-top:.6rem">'
            '<tr><th>Var</th><th>R² (log)</th><th>RMSE</th><th>MAE</th></tr>'
            '<tr><td>A</td><td>0.8723</td><td>2,063</td><td>253</td></tr>'
            '<tr><td class="bld">B</td><td class="bld">0.8878</td><td class="bld">2,010</td><td class="bld">246</td></tr>'
            '<tr><td>C</td><td>0.8870</td><td>1,945</td><td>237</td></tr>'
            '</table></div>'

            # Classificação
            '<div class="dc-card">'
            '<div class="dc-card-title">Classificacao de risco</div>'
            '<p style="margin-bottom:.5rem;font-size:.82rem;color:var(--sub)">4 niveis baseados em percentis historicos:</p>'
            '<div style="display:flex;gap:.3rem;margin-bottom:.7rem">'
            '<span class="about-badge baixo">Baixo</span>'
            '<span class="about-badge moderado">Moderado</span>'
            '<span class="about-badge alto">Alto</span>'
            '<span class="about-badge surto">Surto</span>'
            '</div>'
            '<div class="about-metrics">'
            '<div class="about-mc"><div class="about-mc-v">0.578</div><div class="about-mc-k">F1 macro</div></div>'
            '<div class="about-mc"><div class="about-mc-v">0.830</div><div class="about-mc-k">AUC</div></div>'
            '<div class="about-mc"><div class="about-mc-v">0.725</div><div class="about-mc-k">F1 Surto</div></div>'
            '</div>'
            '<p style="font-size:.72rem;color:var(--muted);margin-top:.3rem">Var B — INMET enriquecido</p>'
            '</div>'

            # Validação
            '<div class="dc-card">'
            '<div class="dc-card-title">Validacao</div>'
            '<ul class="about-vl">'
            '<li><b>Walk-forward</b> — treino expandido, teste ano a ano (2023→2026)</li>'
            '<li><b>Bootstrap IC 95%</b> — 1000 reamostragens por modelo</li>'
            '<li><b>Significancia</b> — B vs A: p = 8.29×10⁻⁴⁰</li>'
            '</ul></div>'

            # Tecnologias
            '<div class="dc-card">'
            '<div class="dc-card-title">Stack tecnologico</div>'
            '<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0">'
            '<span class="about-chip">XGBoost</span>'
            '<span class="about-chip">Python</span>'
            '<span class="about-chip">Streamlit</span>'
            '<span class="about-chip">Plotly</span>'
            '<span class="about-chip">Pandas</span>'
            '<span class="about-chip">scikit-learn</span>'
            '<span class="about-chip">NumPy</span>'
            '<span class="about-chip">SHAP</span>'
            '</div>'
            '<p style="margin-top:.6rem;font-size:.82rem;color:var(--sub)">'
            'Dados: <b style="color:var(--ink)">SINAN</b> (notificacoes epidemiologicas) e '
            '<b style="color:var(--ink)">INMET</b> (estacoes meteorologicas).</p>'
            '</div>'

            # Limitações
            '<div class="dc-card full">'
            '<div class="dc-card-title">Limitacoes conhecidas</div>'
            '<div class="about-warn"><ul>'
            '<li>INMET Silver ausente para 2009, 2020 e 2022 reduz cobertura climatica</li>'
            '<li>Variante A tem cobertura INMET de ~48% (vs ~90% original) devido a falta de 2020/2022</li>'
            '<li>2026 e ano parcial (17 semanas epidemiologicas no corte)</li>'
            '<li>Assume estabilidade da distribuicao temporal entre periodos de treino e teste</li>'
            '</ul></div></div>'

            # Créditos
            '<div class="about-credits full" style="grid-column:1/-1">'
            '<div class="dc-card-title">Autores</div>'
            '<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap">'
            '<div style="flex:1;min-width:200px">'
            '<p class="about-author-name">Pedro Lucas Figueiredo Santana</p>'
            '<p class="about-author-role">Engenharia de Software — UnB</p></div>'
            '<div style="flex:1;min-width:200px">'
            '<p class="about-author-name">Thiago Ribeiro Freitas</p>'
            '<p class="about-author-role">Engenharia de Software — UnB</p></div>'
            '</div>'
            '<p style="margin-top:.8rem;font-size:.76rem;color:var(--sub)">'
            'Trabalho de Conclusao de Curso (TCC) — Faculdade do Gama (FGA), '
            'Universidade de Brasilia, 2024–2025.</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True)


if __name__ == "__main__":
    main()

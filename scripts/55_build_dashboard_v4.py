#!/usr/bin/env python3
"""v4-55: Gera landing page HTML estática — Sistema Preditivo de Surtos de Dengue.

Carrega modelos XGBoost, gera previsões no conjunto de teste, e produz
docs/dashboard_v4.html com painel de alertas + resultados do modelo.
Auto-contido (Chart.js + Tailwind via CDN, dados embutidos como JSON).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
MODEL_DIR_CLS = BASE_DIR / "models" / "classification_v4"
MAPPING_PATH = BASE_DIR / "data" / "model_ready_v3" / "municipio_mesorregiao.csv"
OUT_PATH = BASE_DIR / "docs" / "dashboard_v4.html"

CLASS_NAMES = {0: "Baixo", 1: "Moderado", 2: "Alto", 3: "Surto"}
UF_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}


def load_data():
    data = {}
    final_path = DATA_DIR / "52_final_results_v4.json"
    if final_path.exists():
        with open(final_path) as f:
            data["final"] = json.load(f)

    reg_path = DATA_DIR / "46_regression_v4_report.json"
    if reg_path.exists():
        with open(reg_path) as f:
            data["regression"] = json.load(f)

    wf_path = DATA_DIR / "47_walkforward_v4_report.json"
    if wf_path.exists():
        with open(wf_path) as f:
            data["walkforward"] = json.load(f)

    shap_path = MODEL_DIR / "shap_importance_v4.csv"
    if shap_path.exists():
        df = pd.read_csv(shap_path).head(20)
        data["shap_top20"] = df.to_dict(orient="records")

    drift_path = DATA_DIR / "50_drift_report.json"
    if drift_path.exists():
        with open(drift_path) as f:
            data["drift"] = json.load(f)

    lc_path = DATA_DIR / "51_learning_curve_report.json"
    if lc_path.exists():
        with open(lc_path) as f:
            data["learning_curve"] = json.load(f)

    integ_path = DATA_DIR / "42_integration_report.json"
    if integ_path.exists():
        with open(integ_path) as f:
            data["integration"] = json.load(f)

    cls_path = DATA_DIR / "54_classification_v4_report.json"
    if cls_path.exists():
        with open(cls_path) as f:
            data["classification"] = json.load(f)

    uf_path = MODEL_DIR / "metrics_por_uf_v4.csv"
    if uf_path.exists():
        df = pd.read_csv(uf_path)
        enriq = df[df["modelo"].str.contains("enriquecido")]
        data["uf_metrics"] = enriq.to_dict(orient="records")

    # Load models and generate predictions
    data["predictions"] = generate_predictions()

    return data


def generate_predictions():
    reg_path = MODEL_DIR / "model_B_inmet_enriquecido.ubj"
    cls_path = MODEL_DIR_CLS / "model_B_inmet_enriquecido_cls.ubj"
    test_path = DATA_DIR / "splits_b" / "test.parquet"

    if not all(p.exists() for p in [reg_path, cls_path, test_path]):
        return {}

    reg_model = xgb.Booster()
    reg_model.load_model(str(reg_path))
    cls_model = xgb.Booster()
    cls_model.load_model(str(cls_path))

    test = pd.read_parquet(test_path)
    test["uf"] = (test["cod_mesorregiao"] // 100).map(UF_MAP)

    feat_names = reg_model.feature_names
    dmat = xgb.DMatrix(test[feat_names], feature_names=feat_names)

    pred_log = reg_model.predict(dmat)
    pred_notif = np.maximum(np.expm1(pred_log), 0)
    cls_probs = cls_model.predict(dmat)
    pred_risk = cls_probs.argmax(axis=1)

    test["pred_notif"] = pred_notif
    test["pred_risk"] = pred_risk

    meso_names = {}
    if MAPPING_PATH.exists():
        mapping = pd.read_csv(MAPPING_PATH)
        meso_names = mapping.drop_duplicates("cod_mesorregiao").set_index("cod_mesorregiao")["nome_mesorregiao"].to_dict()

    latest = test.sort_values(["ano", "semana_epidemiologica"]).groupby("cod_mesorregiao").last().reset_index()

    risk_dist = {CLASS_NAMES[i]: int((latest["pred_risk"] == i).sum()) for i in range(4)}

    top_alerts = latest[latest["pred_risk"] >= 2].nlargest(15, "pred_notif")
    alerts = []
    for _, row in top_alerts.iterrows():
        cod = int(row["cod_mesorregiao"])
        alerts.append({
            "meso": f"{cod} - {meso_names.get(cod, '?')}",
            "uf": row["uf"],
            "pred_notif": int(round(row["pred_notif"])),
            "risk": CLASS_NAMES[int(row["pred_risk"])],
            "prob_surto": round(float(cls_probs[test.index.get_loc(row.name), 3]) if hasattr(test.index, 'get_loc') else 0, 3),
        })

    uf_risk = latest.groupby("uf")["pred_risk"].mean().sort_values(ascending=False)
    uf_risk_data = {"labels": uf_risk.index.tolist(), "values": [round(v, 2) for v in uf_risk.values]}

    weekly = test.groupby(["ano", "semana_epidemiologica"]).agg(
        n_surto=("pred_risk", lambda x: int((x == 3).sum())),
        n_alto=("pred_risk", lambda x: int((x == 2).sum())),
    ).reset_index()
    weekly["periodo"] = weekly["ano"].astype(str) + "-S" + weekly["semana_epidemiologica"].astype(str).str.zfill(2)

    from sklearn.metrics import r2_score
    r2 = r2_score(np.log1p(test["notificacoes_t4"].values), pred_log)

    return {
        "risk_distribution": risk_dist,
        "alerts": alerts,
        "uf_risk": uf_risk_data,
        "weekly_alerts": {
            "periodos": weekly["periodo"].tolist(),
            "surto": weekly["n_surto"].tolist(),
            "alto": weekly["n_alto"].tolist(),
        },
        "total_predictions": len(test),
        "r2_log": round(r2, 4),
        "n_mesos": int(latest["cod_mesorregiao"].nunique()),
    }


def build_html(data):
    reg = data.get("regression", {})
    wf = data.get("walkforward", {})
    final = data.get("final", {})
    integ = data.get("integration", {})

    models = reg.get("modelos", {})
    a = models.get("A_inmet_enriquecido", {})
    b = models.get("B_inmet_enriquecido", {})
    c = models.get("C_inmet_enriquecido", {})

    ttests = reg.get("testes_t", {})
    boot = reg.get("bootstrap_diffs", {})

    wf_resumo = wf.get("resumo_enriquecido", {})
    shap = data.get("shap_top20", [])
    drift = data.get("drift", {})
    lc = data.get("learning_curve", {})
    cls = data.get("classification", {})
    uf_mets = data.get("uf_metrics", [])

    cov_by_year = integ.get("cobertura_por_ano", {})

    data_json = json.dumps({
        "variantes": {
            "labels": ["Var A", "Var B", "Var C"],
            "r2_log": [a.get("R2_log", 0), b.get("R2_log", 0), c.get("R2_log", 0)],
            "ci_lo": [a.get("R2_log_CI95_lo", 0), b.get("R2_log_CI95_lo", 0), c.get("R2_log_CI95_lo", 0)],
            "ci_hi": [a.get("R2_log_CI95_hi", 0), b.get("R2_log_CI95_hi", 0), c.get("R2_log_CI95_hi", 0)],
            "rmse": [a.get("RMSE", 0), b.get("RMSE", 0), c.get("RMSE", 0)],
            "mae": [a.get("MAE", 0), b.get("MAE", 0), c.get("MAE", 0)],
        },
        "walkforward": {
            "A": wf_resumo.get("mean_R2_log_A", 0),
            "B": wf_resumo.get("mean_R2_log_B", 0),
            "C": wf_resumo.get("mean_R2_log_C", 0),
            "folds": {},
        },
        "shap": {
            "features": [s.get("feature", "") for s in shap],
            "values": [round(s.get("mean_abs_shap", 0), 4) for s in shap],
        },
        "coverage": {
            "years": sorted(int(y) for y in cov_by_year.keys()) if cov_by_year else [],
            "pcts": [cov_by_year.get(str(y), 0) for y in sorted(int(y) for y in cov_by_year.keys())] if cov_by_year else [],
        },
        "uf": {
            "labels": [m.get("uf", "") for m in uf_mets if "Var B" in m.get("modelo", "")],
            "r2_log": [m.get("R2_log", 0) for m in uf_mets if "Var B" in m.get("modelo", "")],
        },
    }, ensure_ascii=False)

    # Walk-forward folds
    wf_folds_data = {}
    if "modelos" in wf:
        for var in ["A", "B", "C"]:
            key = f"{var}_inmet_enriquecido"
            if key in wf["modelos"]:
                folds = wf["modelos"][key].get("folds", [])
                wf_folds_data[var] = {
                    "labels": [f"F{i+1} ({f['test_years'][0]})" for i, f in enumerate(folds)],
                    "values": [f["R2_log"] for f in folds],
                }

    lc_data = {}
    if "windows" in lc:
        lc_data = {
            "labels": [p.get("label", "") for p in lc["windows"]],
            "r2": [p.get("R2_log", 0) for p in lc["windows"]],
        }

    drift_data = {}
    psi_key = "mean_psi_by_year" if "mean_psi_by_year" in drift else "psi_por_ano"
    if psi_key in drift:
        psi = drift[psi_key]
        drift_data = {
            "years": sorted(int(y) for y in psi.keys()),
            "psi": [psi.get(str(y), 0) for y in sorted(int(y) for y in psi.keys())],
        }

    extra_json = json.dumps({
        "wf_folds": wf_folds_data,
        "learning_curve": lc_data,
        "drift": drift_data,
    }, ensure_ascii=False)

    # t-test data
    ab = ttests.get("A_vs_B", {})
    ac = ttests.get("A_vs_C", {})
    bc = ttests.get("B_vs_C", {})
    bab = boot.get("A_vs_B", {})
    bac = boot.get("A_vs_C", {})
    bbc = boot.get("B_vs_C", {})

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    cls_b = {}
    if cls and "modelos" in cls:
        cls_b = cls["modelos"].get("B_inmet_enriquecido", {})
    cls_f1_surto = cls_b.get("f1_per_class", {}).get("surto", 0) if cls_b else 0

    preds = data.get("predictions", {})
    risk_dist = preds.get("risk_distribution", {})
    alerts = preds.get("alerts", [])
    pred_json = json.dumps({
        "uf_risk": preds.get("uf_risk", {}),
        "weekly": preds.get("weekly_alerts", {}),
    }, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DengueCast v4</title>
<script src="https://cdn.tailwindcss.com/3.4.1"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {{ --bg: #f8fafc; --card: #fff; --text: #1e293b; --muted: #64748b; --border: #e2e8f0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8; --border: #334155; }} }}
:root[data-theme="dark"] {{ --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8; --border: #334155; }}
body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }}
.kpi {{ text-align: center; }}
.kpi .value {{ font-size: 2rem; font-weight: 700; color: #2563eb; }}
.kpi .label {{ font-size: 0.85rem; color: var(--muted); }}
canvas {{ max-height: 400px; }}
</style>
</head>
<body class="min-h-screen">
<div class="max-w-6xl mx-auto px-4 py-8">

<!-- Header -->
<header class="text-center mb-10">
<h1 class="text-3xl md:text-4xl font-bold mb-2">🦟 DengueCast v4 — Sistema Preditivo de Surtos de Dengue</h1>
<p class="text-lg" style="color:var(--muted)">Modelo XGBoost treinado com SINAN + INMET (2001–2022) — previsão de notificações e risco 4 semanas à frente</p>
<p class="text-sm mt-1" style="color:var(--muted)">137 mesorregiões | 266 features | {preds.get('total_predictions', 0):,} previsões geradas</p>
<button id="themeToggle" onclick="toggleTheme()" class="mt-3 px-3 py-1 rounded border text-sm" style="border-color:var(--border);color:var(--muted)">🌙 / ☀️</button>
</header>

<!-- Prediction Alert KPIs -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
<div class="card kpi"><div class="value" style="color:#F44336">{risk_dist.get('Surto', 0)}</div><div class="label">Mesorregiões em Surto</div></div>
<div class="card kpi"><div class="value" style="color:#FF9800">{risk_dist.get('Alto', 0)}</div><div class="label">Risco Alto</div></div>
<div class="card kpi"><div class="value" style="color:#FFC107">{risk_dist.get('Moderado', 0)}</div><div class="label">Risco Moderado</div></div>
<div class="card kpi"><div class="value" style="color:#4CAF50">{risk_dist.get('Baixo', 0)}</div><div class="label">Risco Baixo</div></div>
</div>

<!-- Alerts Table -->
{"" if not alerts else f'''
<section class="card mb-8">
<h2 class="text-xl font-semibold mb-4" style="color:#F44336">Alertas de Surto — Top 15 Mesorregiões</h2>
<p class="text-sm mb-3" style="color:var(--muted)">Mesorregiões com risco previsto Alto ou Surto, ordenadas por volume de notificações previstas (t+4 semanas).</p>
<div class="overflow-x-auto">
<table class="w-full text-sm">
<thead><tr class="border-b" style="border-color:var(--border)">
<th class="py-2 text-left">Mesorregião</th><th>UF</th><th>Prev. Notif.</th><th>Risco</th>
</tr></thead>
<tbody>
''' + ''.join(f'<tr class="border-b" style="border-color:var(--border)"><td class="py-1">{al["meso"]}</td><td class="text-center">{al["uf"]}</td><td class="text-center">{al["pred_notif"]:,}</td><td class="text-center"><span style="color:{"#F44336" if al["risk"]=="Surto" else "#FF9800"};font-weight:600">{al["risk"]}</span></td></tr>' for al in alerts) + '''
</tbody></table>
</div>
</section>
'''}

<!-- Prediction Charts -->
<div class="grid md:grid-cols-2 gap-8 mb-8">
<section class="card"><h2 class="text-lg font-semibold mb-3">Evolução de Alertas por Semana</h2><canvas id="chartWeeklyAlerts"></canvas></section>
<section class="card"><h2 class="text-lg font-semibold mb-3">Risco Médio por UF</h2><canvas id="chartUFRisk"></canvas></section>
</div>

<!-- Model Performance KPIs -->
<div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
<div class="card kpi"><div class="value">{b.get('R2_log', 0):.4f}</div><div class="label">R²_log (Var B)</div></div>
<div class="card kpi"><div class="value">{b.get('RMSE', 0):.0f}</div><div class="label">RMSE</div></div>
<div class="card kpi"><div class="value">137</div><div class="label">Mesorregiões</div></div>
<div class="card kpi"><div class="value">22</div><div class="label">Anos de treino</div></div>
<div class="card kpi"><div class="value">266</div><div class="label">Features</div></div>
</div>

<!-- Variantes -->
<section class="card mb-8">
<h2 class="text-xl font-semibold mb-4">Comparação de Variantes (INMET enriquecido)</h2>
<div class="overflow-x-auto">
<table class="w-full text-sm">
<thead><tr class="border-b" style="border-color:var(--border)">
<th class="py-2 text-left">Variante</th><th>R²_log</th><th>IC 95%</th><th>RMSE</th><th>MAE</th>
</tr></thead>
<tbody>
<tr class="border-b" style="border-color:var(--border)"><td class="py-2">A (baseline 4yr)</td><td class="text-center">{a.get('R2_log',0):.4f}</td><td class="text-center">[{a.get('R2_log_CI95_lo',0):.4f}, {a.get('R2_log_CI95_hi',0):.4f}]</td><td class="text-center">{a.get('RMSE',0):.0f}</td><td class="text-center">{a.get('MAE',0):.1f}</td></tr>
<tr class="border-b font-semibold" style="border-color:var(--border)"><td class="py-2">B (série 22yr) ⭐</td><td class="text-center">{b.get('R2_log',0):.4f}</td><td class="text-center">[{b.get('R2_log_CI95_lo',0):.4f}, {b.get('R2_log_CI95_hi',0):.4f}]</td><td class="text-center">{b.get('RMSE',0):.0f}</td><td class="text-center">{b.get('MAE',0):.1f}</td></tr>
<tr><td class="py-2">C (série + peso)</td><td class="text-center">{c.get('R2_log',0):.4f}</td><td class="text-center">[{c.get('R2_log_CI95_lo',0):.4f}, {c.get('R2_log_CI95_hi',0):.4f}]</td><td class="text-center">{c.get('RMSE',0):.0f}</td><td class="text-center">{c.get('MAE',0):.1f}</td></tr>
</tbody></table>
</div>
<div class="mt-4" style="max-width:500px;margin:0 auto"><canvas id="chartVariantes"></canvas></div>
</section>

<!-- Statistical Tests -->
<section class="card mb-8">
<h2 class="text-xl font-semibold mb-4">Testes Estatísticos</h2>
<div class="overflow-x-auto">
<table class="w-full text-sm">
<thead><tr class="border-b" style="border-color:var(--border)">
<th class="py-2 text-left">Comparação</th><th>ΔR²_log</th><th>IC 95%</th><th>p-value</th><th>Sig.</th>
</tr></thead>
<tbody>
<tr class="border-b" style="border-color:var(--border)"><td class="py-2">B vs A</td><td class="text-center">{bab.get('mean_diff',0):+.4f}</td><td class="text-center">[{bab.get('CI95_lo',0):.4f}, {bab.get('CI95_hi',0):.4f}]</td><td class="text-center">{ab.get('p_value',1):.2e}</td><td class="text-center">{sig(ab.get('p_value',1))}</td></tr>
<tr class="border-b" style="border-color:var(--border)"><td class="py-2">C vs A</td><td class="text-center">{bac.get('mean_diff',0):+.4f}</td><td class="text-center">[{bac.get('CI95_lo',0):.4f}, {bac.get('CI95_hi',0):.4f}]</td><td class="text-center">{ac.get('p_value',1):.2e}</td><td class="text-center">{sig(ac.get('p_value',1))}</td></tr>
<tr><td class="py-2">C vs B</td><td class="text-center">{bbc.get('mean_diff',0):+.4f}</td><td class="text-center">[{bbc.get('CI95_lo',0):.4f}, {bbc.get('CI95_hi',0):.4f}]</td><td class="text-center">{bc.get('p_value',1):.2e}</td><td class="text-center">{sig(bc.get('p_value',1))}</td></tr>
</tbody></table>
</div>
</section>

<!-- Charts grid -->
<div class="grid md:grid-cols-2 gap-8 mb-8">
<section class="card"><h2 class="text-lg font-semibold mb-3">Walk-Forward</h2><canvas id="chartWF"></canvas></section>
<section class="card"><h2 class="text-lg font-semibold mb-3">SHAP — Top 20 Features</h2><canvas id="chartSHAP"></canvas></section>
<section class="card"><h2 class="text-lg font-semibold mb-3">Cobertura INMET por Ano</h2><canvas id="chartCoverage"></canvas></section>
<section class="card"><h2 class="text-lg font-semibold mb-3">R²_log por UF (Var B)</h2><canvas id="chartUF"></canvas></section>
</div>

<div class="grid md:grid-cols-2 gap-8 mb-8">
<section class="card"><h2 class="text-lg font-semibold mb-3">Learning Curve</h2><canvas id="chartLC"></canvas></section>
<section class="card"><h2 class="text-lg font-semibold mb-3">Data Drift (PSI por Ano)</h2><canvas id="chartDrift"></canvas></section>
</div>

<!-- Classification -->
{"" if not cls_b else f'''
<section class="card mb-8">
<h2 class="text-xl font-semibold mb-4">Classificação de Risco (Var B INMET enriquecido)</h2>
<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
<div class="kpi"><div class="value">{cls_b.get("f1_macro",0):.4f}</div><div class="label">F1 macro</div></div>
<div class="kpi"><div class="value">{cls_b.get("auc_macro_ovr",0):.4f}</div><div class="label">AUC macro OVR</div></div>
<div class="kpi"><div class="value">{cls_b.get("accuracy",0):.4f}</div><div class="label">Acurácia</div></div>
<div class="kpi"><div class="value">{cls_f1_surto:.4f}</div><div class="label">F1 surto</div></div>
</div>
</section>
'''}

<!-- Conclusions -->
<section class="card mb-8">
<h2 class="text-xl font-semibold mb-4">Conclusões</h2>
<ol class="list-decimal pl-5 space-y-2">
<li><strong>H1 — Série temporal completa melhora:</strong> SIM. ΔR²_log = {bab.get('mean_diff',0):+.4f}, p &lt; 0.001.</li>
<li><strong>H2 — Decaimento exponencial melhora:</strong> NÃO. ΔR²_log C vs B = {bbc.get('mean_diff',0):+.4f}, ns.</li>
<li><strong>Melhor variante: B</strong> (série completa sem peso).</li>
<li>Walk-forward confirma B &gt; A (Δ = {wf_resumo.get('delta_B_vs_A',0):+.4f}).</li>
</ol>
</section>

<!-- Footer -->
<footer class="text-center py-6 text-sm" style="color:var(--muted)">
DengueCast v4 — Sistema Preditivo de Surtos de Dengue — TCC Eng. Software UnB
<br>Gerado em <span id="genDate"></span> | 16 scripts, 9 modelos regressão + 9 classificação, 137 mesorregiões
</footer>

</div>

<script>
const D = {data_json};
const E = {extra_json};
const P = {pred_json};
document.getElementById('genDate').textContent = new Date().toLocaleDateString('pt-BR');

function toggleTheme() {{
  const r = document.documentElement;
  const cur = r.getAttribute('data-theme');
  r.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
}}

const colors = {{ A: '#1976D2', B: '#FF9800', C: '#E53935' }};

// Variantes bar
new Chart(document.getElementById('chartVariantes'), {{
  type: 'bar',
  data: {{
    labels: D.variantes.labels,
    datasets: [{{
      data: D.variantes.r2_log,
      backgroundColor: [colors.A, colors.B, colors.C],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ beginAtZero: false, title: {{ display: true, text: 'R²_log' }} }} }},
  }}
}});

// Walk-forward
if (E.wf_folds && E.wf_folds.B) {{
  const datasets = [];
  for (const v of ['A','B','C']) {{
    if (E.wf_folds[v]) {{
      datasets.push({{
        label: 'Var ' + v,
        data: E.wf_folds[v].values,
        borderColor: colors[v],
        backgroundColor: colors[v] + '33',
        tension: 0.3,
        pointRadius: 5,
      }});
    }}
  }}
  new Chart(document.getElementById('chartWF'), {{
    type: 'line',
    data: {{ labels: E.wf_folds.B.labels, datasets }},
    options: {{ responsive: true, scales: {{ y: {{ title: {{ display: true, text: 'R²_log' }} }} }} }}
  }});
}}

// SHAP
if (D.shap.features.length) {{
  new Chart(document.getElementById('chartSHAP'), {{
    type: 'bar',
    data: {{
      labels: D.shap.features.slice().reverse(),
      datasets: [{{ data: D.shap.values.slice().reverse(), backgroundColor: '#2563eb88', borderColor: '#2563eb', borderWidth: 1 }}]
    }},
    options: {{ indexAxis: 'y', responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Mean |SHAP|' }} }} }} }}
  }});
}}

// Coverage
if (D.coverage.years.length) {{
  new Chart(document.getElementById('chartCoverage'), {{
    type: 'bar',
    data: {{
      labels: D.coverage.years,
      datasets: [{{ data: D.coverage.pcts, backgroundColor: D.coverage.pcts.map(p => p >= 80 ? '#4CAF50' : p >= 40 ? '#FF9800' : '#E53935') }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'Cobertura (%)' }}, max: 100 }} }} }}
  }});
}}

// UF
if (D.uf.labels.length) {{
  new Chart(document.getElementById('chartUF'), {{
    type: 'bar',
    data: {{ labels: D.uf.labels, datasets: [{{ data: D.uf.r2_log, backgroundColor: '#2563eb88' }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'R²_log' }} }} }} }}
  }});
}}

// Learning Curve
if (E.learning_curve && E.learning_curve.labels) {{
  new Chart(document.getElementById('chartLC'), {{
    type: 'line',
    data: {{
      labels: E.learning_curve.labels,
      datasets: [{{ label: 'R²_log', data: E.learning_curve.r2, borderColor: '#2563eb', tension: 0.3, pointRadius: 5 }}]
    }},
    options: {{ responsive: true, scales: {{ y: {{ title: {{ display: true, text: 'R²_log' }} }} }} }}
  }});
}}

// Drift
if (E.drift && E.drift.years) {{
  new Chart(document.getElementById('chartDrift'), {{
    type: 'bar',
    data: {{
      labels: E.drift.years,
      datasets: [{{ data: E.drift.psi, backgroundColor: E.drift.psi.map(p => p > 0.2 ? '#E53935' : p > 0.1 ? '#FF9800' : '#4CAF50') }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'PSI médio' }} }} }} }}
  }});
}}

// Weekly Alerts
if (P.weekly && P.weekly.periodos) {{
  new Chart(document.getElementById('chartWeeklyAlerts'), {{
    type: 'line',
    data: {{
      labels: P.weekly.periodos,
      datasets: [
        {{ label: 'Surto', data: P.weekly.surto, borderColor: '#F44336', backgroundColor: 'rgba(244,67,54,0.15)', fill: true, tension: 0.3 }},
        {{ label: 'Alto', data: P.weekly.alto, borderColor: '#FF9800', backgroundColor: 'rgba(255,152,0,0.1)', fill: true, tension: 0.3 }}
      ]
    }},
    options: {{ responsive: true, scales: {{ y: {{ title: {{ display: true, text: 'Nº Mesorregiões' }} }} }} }}
  }});
}}

// UF Risk
if (P.uf_risk && P.uf_risk.labels) {{
  new Chart(document.getElementById('chartUFRisk'), {{
    type: 'bar',
    data: {{
      labels: P.uf_risk.labels,
      datasets: [{{ data: P.uf_risk.values, backgroundColor: P.uf_risk.values.map(v => v >= 2 ? '#F44336' : v >= 1.5 ? '#FF9800' : v >= 1 ? '#FFC107' : '#4CAF50') }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'Risco médio (0=Baixo→3=Surto)' }}, max: 3 }} }} }}
  }});
}}
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 70)
    print("  v4-55: Build Landing Page HTML Estática")
    print("=" * 70)

    data = load_data()
    loaded = [k for k in data.keys()]
    print(f"  Dados carregados: {loaded}")

    html = build_html(data)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n  Salvo: {OUT_PATH}")
    print(f"  Tamanho: {len(html)/1024:.1f} KB")

    print(f"\n{'=' * 70}")
    print(f"  Landing page gerada: docs/dashboard_v4.html")
    print(f"  Abra com: open docs/dashboard_v4.html")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

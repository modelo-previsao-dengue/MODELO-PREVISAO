#!/usr/bin/env python3
"""US-113: Regressão v2 — 3 modelos comparativos.

A. SINAN-only
B. SINAN + INMET bruto (30 features originais)
C. SINAN + INMET enriquecido (bruto + lags + anomalias = 174 features)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "regression_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA",
    "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS",
    "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}

REGIAO_MAP = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte",
    "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
}


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def categorize_features(all_features):
    """Split features into SINAN, INMET bruto, and INMET enriched."""
    sinan = []
    inmet_bruto = []
    inmet_enriched = []

    for c in all_features:
        if not is_inmet_feature(c):
            sinan.append(c)
        elif ("_lag_" in c and c.endswith("w")) or "_anomalia" in c or "_mm_2_4w" in c or "_mm_4_8w" in c:
            inmet_enriched.append(c)
        else:
            inmet_bruto.append(c)

    return sinan, inmet_bruto, inmet_enriched


def train_and_eval(train, val, test, features, label):
    """Train XGBoost regressor and evaluate."""
    print(f"\n  Treinando modelo {label} ({len(features)} features)...")

    y_train = np.log1p(train[TARGET])
    y_val = np.log1p(val[TARGET])
    y_test_log = np.log1p(test[TARGET])
    y_test_orig = test[TARGET].values

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[features], y_train,
        eval_set=[(val[features], y_val)],
        verbose=0,
    )

    pred_log = model.predict(test[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)
    r2_orig = r2_score(y_test_orig, pred_orig)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, pred_log))
    r2_log = r2_score(y_test_log, pred_log)

    metrics = {
        "label": label,
        "n_features": len(features),
        "R2_log": round(float(r2_log), 4),
        "R2": round(float(r2_orig), 4),
        "RMSE": round(float(rmse), 2),
        "RMSE_log": round(float(rmse_log), 4),
        "MAE": round(float(mae), 4),
        "best_iteration": int(model.best_iteration),
    }

    print(f"    R²_log={r2_log:.4f}  R²={r2_orig:.4f}  RMSE={rmse:.2f}  MAE={mae:.4f}")

    errors_sq = (y_test_log.values - pred_log) ** 2

    return model, metrics, pred_log, errors_sq


def eval_by_uf(test, pred_log, label):
    """Evaluate per UF."""
    test_tmp = test.copy()
    test_tmp["pred_log"] = pred_log
    test_tmp["y_log"] = np.log1p(test_tmp[TARGET])
    test_tmp["uf_code"] = test_tmp["ibge_municipio"].str[:2]
    test_tmp["uf"] = test_tmp["uf_code"].map(UF_MAP)
    test_tmp["regiao"] = test_tmp["uf"].map(REGIAO_MAP)

    uf_metrics = []
    for uf in sorted(test_tmp["uf"].dropna().unique()):
        sub = test_tmp[test_tmp["uf"] == uf]
        if len(sub) < 50:
            continue
        r2 = r2_score(sub["y_log"], sub["pred_log"])
        rmse = np.sqrt(mean_squared_error(sub[TARGET], np.maximum(np.expm1(sub["pred_log"]), 0)))
        uf_metrics.append({
            "uf": uf,
            "regiao": REGIAO_MAP.get(uf, "?"),
            "modelo": label,
            "R2_log": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "n": len(sub),
        })

    return pd.DataFrame(uf_metrics)


def plot_comparison(results, out_path):
    """Barplot comparing 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))

    for ax, metric, title in zip(axes,
                                  ["R2_log", "RMSE", "MAE"],
                                  ["R² (log)", "RMSE", "MAE"]):
        vals = [r[metric] for r in results]
        colors = ["#1976D2", "#FF9800", "#E53935"]
        bars = ax.bar(x, vals, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}" if metric != "RMSE" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Comparação de Modelos — Regressão v2 (log1p target)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_delta_uf(uf_all, out_path):
    """Delta R² per UF: model C vs model A."""
    a = uf_all[uf_all["modelo"] == "A: SINAN-only"].set_index("uf")
    c = uf_all[uf_all["modelo"] == "C: INMET enriquecido"].set_index("uf")
    common = sorted(set(a.index) & set(c.index))
    if not common:
        return

    deltas = [(uf, c.loc[uf, "R2_log"] - a.loc[uf, "R2_log"]) for uf in common]
    deltas.sort(key=lambda x: x[1])
    ufs = [d[0] for d in deltas]
    vals = [d[1] for d in deltas]
    colors = ["#4CAF50" if v > 0 else "#E53935" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(ufs)), vals, color=colors)
    ax.set_yticks(range(len(ufs)))
    ax.set_yticklabels(ufs, fontsize=9)
    ax.set_xlabel("ΔR²_log (INMET enriquecido − SINAN-only)", fontsize=10)
    ax.set_title("Impacto do INMET Enriquecido por UF", fontsize=12)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-113: Regressão v2 — 3 Modelos Comparativos")
    print("=" * 60)

    print("\nCarregando splits v2...")
    train = pd.read_parquet(DATA_DIR / "train_v2.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v2.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v2.parquet")
    for d in [train, val, test]:
        d["ibge_municipio"] = d["ibge_municipio"].astype(str)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    all_features = [c for c in train.columns
                    if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                    and train[c].dtype != "object" and not str(train[c].dtype).startswith("datetime")]

    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(all_features)
    print(f"\n  Features: {len(sinan_feats)} SINAN + {len(inmet_bruto)} INMET bruto + {len(inmet_enriched)} INMET enriquecido")

    feats_a = sinan_feats
    feats_b = sinan_feats + inmet_bruto
    feats_c = sinan_feats + inmet_bruto + inmet_enriched

    model_a, met_a, pred_a, err_a = train_and_eval(train, val, test, feats_a, "A: SINAN-only")
    model_b, met_b, pred_b, err_b = train_and_eval(train, val, test, feats_b, "B: INMET bruto")
    model_c, met_c, pred_c, err_c = train_and_eval(train, val, test, feats_c, "C: INMET enriquecido")

    print("\n  Testes t pareados:")
    pairs = [("A vs B", err_a, err_b), ("A vs C", err_a, err_c), ("B vs C", err_b, err_c)]
    ttests = {}
    for name, e1, e2 in pairs:
        t_stat, p_val = ttest_rel(e1, e2)
        ttests[name] = {"t_stat": round(float(t_stat), 4), "p_value": float(p_val)}
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"    {name}: t={t_stat:.4f}, p={p_val:.2e} ({sig})")

    print("\n  Avaliando por UF...")
    uf_a = eval_by_uf(test, pred_a, "A: SINAN-only")
    uf_b = eval_by_uf(test, pred_b, "B: INMET bruto")
    uf_c = eval_by_uf(test, pred_c, "C: INMET enriquecido")
    uf_all = pd.concat([uf_a, uf_b, uf_c], ignore_index=True)
    uf_all.to_csv(MODEL_DIR / "metrics_por_uf_v2.csv", index=False)

    by_regiao = uf_all.groupby(["modelo", "regiao"])["R2_log"].mean().unstack(level=0)
    print("\n  R²_log médio por região:")
    print(by_regiao.to_string())

    print("\n  Salvando modelos...")
    model_a.save_model(str(MODEL_DIR / "model_a_sinan_only.json"))
    model_b.save_model(str(MODEL_DIR / "model_b_inmet_bruto.json"))
    model_c.save_model(str(MODEL_DIR / "model_c_inmet_enriquecido.json"))

    print("\nGerando figuras...")
    plot_comparison([met_a, met_b, met_c], FIG_DIR / "fig_5yr_v2_regression_comparison.png")
    plot_delta_uf(uf_all, FIG_DIR / "fig_5yr_v2_regression_delta_uf.png")

    delta_ab = met_b["R2_log"] - met_a["R2_log"]
    delta_ac = met_c["R2_log"] - met_a["R2_log"]
    delta_bc = met_c["R2_log"] - met_b["R2_log"]

    report = {
        "modelos": {
            "A_sinan_only": met_a,
            "B_inmet_bruto": met_b,
            "C_inmet_enriquecido": met_c,
        },
        "deltas": {
            "B_vs_A": round(delta_ab, 4),
            "C_vs_A": round(delta_ac, 4),
            "C_vs_B": round(delta_bc, 4),
        },
        "testes_t": ttests,
        "conclusao": {
            "inmet_bruto_ajuda": delta_ab > 0,
            "inmet_enriquecido_ajuda": delta_ac > 0,
            "feature_eng_ajuda": delta_bc > 0,
        },
    }
    with open(DATA_DIR / "13_regression_v2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-113:")
    print(f"  A (SINAN-only):        R²_log = {met_a['R2_log']:.4f}")
    print(f"  B (INMET bruto):       R²_log = {met_b['R2_log']:.4f}  (Δ = {delta_ab:+.4f})")
    print(f"  C (INMET enriquecido): R²_log = {met_c['R2_log']:.4f}  (Δ vs A = {delta_ac:+.4f})")
    print(f"  Feature eng. (C vs B): Δ = {delta_bc:+.4f}")
    print(f"  INMET bruto ajuda? {'SIM' if delta_ab > 0 else 'NÃO'}")
    print(f"  INMET enriquecido ajuda? {'SIM' if delta_ac > 0 else 'NÃO'}")
    print(f"  Feature engineering ajuda? {'SIM' if delta_bc > 0 else 'NÃO'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

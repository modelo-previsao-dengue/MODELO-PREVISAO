#!/usr/bin/env python3
"""v4-50: Análise de data drift — PSI e KS para top-20 features por SHAP.

RF-06/R-06: Calcular PSI e KS entre cada ano de treino e o teste.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
N_BINS = 10
TOP_FEATURES = 20


def psi(expected, actual, n_bins=N_BINS):
    breakpoints = np.linspace(0, 100, n_bins + 1)
    exp_pct = np.percentile(expected.dropna(), breakpoints)
    exp_pct = np.unique(exp_pct)
    if len(exp_pct) < 3:
        return 0.0

    exp_counts = np.histogram(expected.dropna(), bins=exp_pct)[0]
    act_counts = np.histogram(actual.dropna(), bins=exp_pct)[0]

    exp_pct_arr = (exp_counts + 1) / (exp_counts.sum() + len(exp_counts))
    act_pct_arr = (act_counts + 1) / (act_counts.sum() + len(act_counts))

    return float(np.sum((act_pct_arr - exp_pct_arr) * np.log(act_pct_arr / exp_pct_arr)))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  v4-50: Data Drift Analysis (PSI + KS)")
    print("=" * 70)

    if Path(MODEL_DIR / "shap_importance_v4.csv").exists():
        shap_df = pd.read_csv(MODEL_DIR / "shap_importance_v4.csv")
        top_feats = shap_df.sort_values("shap_mean_abs_B", ascending=False).head(TOP_FEATURES)["feature"].tolist()
        print(f"  Top-{TOP_FEATURES} features por SHAP (variante B)")
    else:
        schema = pd.read_csv(DATA_DIR / "feature_schema_v4.csv")
        top_feats = schema.head(TOP_FEATURES)["feature"].tolist()
        print(f"  SHAP não disponível, usando primeiras {TOP_FEATURES} features do schema")

    print("\nCarregando splits variante B...")
    train_b = pd.read_parquet(DATA_DIR / "splits_b" / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "splits_a" / "test.parquet")
    print(f"  Train B: {len(train_b):,} linhas, anos {sorted(train_b['ano'].unique())}")
    print(f"  Test: {len(test):,} linhas, anos {sorted(test['ano'].unique())}")

    train_years = sorted(train_b["ano"].unique())
    test_years = sorted(test["ano"].unique())
    ref = test

    psi_rows = []
    ks_rows = []

    print(f"\nComputando PSI e KS para {len(top_feats)} features × {len(train_years)} anos de treino...")
    for feat in top_feats:
        ref_vals = ref[feat].dropna()
        if len(ref_vals) < 50:
            continue
        for year in train_years:
            yr_vals = train_b[train_b["ano"] == year][feat].dropna()
            if len(yr_vals) < 50:
                continue

            p = psi(ref_vals, yr_vals)
            ks_stat, ks_p = ks_2samp(ref_vals, yr_vals)

            psi_rows.append({"feature": feat, "ano": int(year), "PSI": round(p, 4)})
            ks_rows.append({"feature": feat, "ano": int(year), "KS_stat": round(float(ks_stat), 4),
                           "KS_p": float(ks_p)})

    psi_df = pd.DataFrame(psi_rows)
    ks_df = pd.DataFrame(ks_rows)

    psi_df.to_csv(MODEL_DIR / "data_drift_psi_v4.csv", index=False)
    ks_df.to_csv(MODEL_DIR / "data_drift_ks_v4.csv", index=False)

    psi_pivot = psi_df.pivot(index="ano", columns="feature", values="PSI")
    mean_psi_by_year = psi_pivot.mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(mean_psi_by_year.index.astype(int), mean_psi_by_year.values, color="#1976D2", alpha=0.8)
    ax.axhline(y=0.1, color="#FF9800", linestyle="--", label="PSI = 0.1 (moderado)")
    ax.axhline(y=0.25, color="#E53935", linestyle="--", label="PSI = 0.25 (severo)")
    ax.set_xlabel("Ano de treino")
    ax.set_ylabel("PSI médio (top-20 features)")
    ax.set_title("Data Drift: PSI Médio por Ano vs Teste (2024-2026)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v4_data_drift_psi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_v4_data_drift_psi.png")

    n_high = (psi_df["PSI"] > 0.25).sum()
    n_moderate = ((psi_df["PSI"] > 0.1) & (psi_df["PSI"] <= 0.25)).sum()
    n_low = (psi_df["PSI"] <= 0.1).sum()

    report = {
        "n_features": len(top_feats),
        "n_train_years": len(train_years),
        "psi_summary": {
            "high_drift_gt025": int(n_high),
            "moderate_drift_01_025": int(n_moderate),
            "low_drift_le01": int(n_low),
        },
        "mean_psi_by_year": {int(k): round(v, 4) for k, v in mean_psi_by_year.items()},
        "features_analyzed": top_feats,
        "tempo_s": round(time.time() - t0, 1),
    }
    with open(DATA_DIR / "50_drift_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-50: PSI alta (>0.25): {n_high}, moderada: {n_moderate}, baixa: {n_low}")
    print(f"  Ano com maior drift: {mean_psi_by_year.idxmax()} (PSI={mean_psi_by_year.max():.4f})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

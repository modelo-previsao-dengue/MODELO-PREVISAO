#!/usr/bin/env python3
"""US-117: Walk-forward v2 com features enriquecidas.

5 folds temporais, comparando SINAN-only vs INMET enriquecido.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

FOLDS = [
    {"train": [2019], "test": 2021},
    {"train": [2019, 2021], "test": 2023},
    {"train": [2019, 2021, 2023], "test": 2024},
    {"train": [2019, 2021, 2023, 2024], "test": 2025},
    {"train": [2019, 2021, 2023, 2024, 2025], "test": 2026},
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def train_eval_fold(df, features, train_years, test_year):
    """Train and evaluate one fold."""
    train_data = df[df["ano"].isin(train_years)]
    test_data = df[df["ano"] == test_year]
    if len(train_data) == 0 or len(test_data) == 0:
        return None

    y_train = np.log1p(train_data[TARGET])
    y_test_log = np.log1p(test_data[TARGET])
    y_test_orig = test_data[TARGET].values

    n_train = len(train_data)
    split_idx = int(n_train * 0.8)
    train_sorted = train_data.sort_values(["ano", "semana_epidemiologica"])
    X_tr = train_sorted[features].iloc[:split_idx]
    y_tr = np.log1p(train_sorted[TARGET].iloc[:split_idx])
    X_ev = train_sorted[features].iloc[split_idx:]
    y_ev = np.log1p(train_sorted[TARGET].iloc[split_idx:])

    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=0)

    pred_log = model.predict(test_data[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    return {
        "train_years": train_years,
        "test_year": test_year,
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test_orig, pred_orig))), 2),
        "MAE": round(float(mean_absolute_error(y_test_orig, pred_orig)), 4),
        "R2": round(float(r2_score(y_test_orig, pred_orig)), 4),
        "R2_log": round(float(r2_score(y_test_log.values, pred_log)), 4),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
    }


def plot_walk_forward(res_sinan, res_full, phase1_sinan, phase1_full, out_path):
    """R² per fold, comparing SINAN-only vs enriched INMET, with Phase 1 overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    years = [r["test_year"] for r in res_sinan]

    ax = axes[0]
    ax.plot(years, [r["R2_log"] for r in res_sinan], "o-", label="v2 SINAN-only",
            color="#1976D2", linewidth=2, markersize=8)
    ax.plot(years, [r["R2_log"] for r in res_full], "s-", label="v2 SINAN+INMET enriq.",
            color="#E53935", linewidth=2, markersize=8)
    if phase1_sinan:
        ax.plot(years[:len(phase1_sinan)], [r["R2_log"] for r in phase1_sinan], "^--",
                label="v1 SINAN-only", color="#1976D2", alpha=0.4, markersize=6)
    if phase1_full:
        ax.plot(years[:len(phase1_full)], [r["R2_log"] for r in phase1_full], "v--",
                label="v1 SINAN+INMET bruto", color="#E53935", alpha=0.4, markersize=6)
    ax.set_xlabel("Ano de Teste", fontsize=11)
    ax.set_ylabel("R² (log)", fontsize=11)
    ax.set_title("R² por Fold — Walk-Forward v2", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    deltas = [f["R2_log"] - s["R2_log"] for s, f in zip(res_sinan, res_full)]
    colors = ["#4CAF50" if d > 0 else "#E53935" for d in deltas]
    ax.bar(years, deltas, color=colors)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.set_xlabel("Ano de Teste", fontsize=11)
    ax.set_ylabel("ΔR²_log", fontsize=11)
    ax.set_title("Delta INMET enriquecido por Fold", fontsize=12)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    for y, d in zip(years, deltas):
        ax.text(y, d + (0.001 if d > 0 else -0.003), f"{d:+.4f}", ha="center", fontsize=9)

    fig.suptitle("Validação Walk-Forward — Fase 2 (Features Enriquecidas)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-117: Walk-Forward v2 (Features Enriquecidas)")
    print("=" * 60)

    print("\nCarregando dataset completo v2...")
    df = pd.read_parquet(DATA_DIR / "integrated_filtered_v2.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df["notificacoes_t4"] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)
    df = df.dropna(subset=["notificacoes_t4"])

    non_numeric = [c for c in df.columns
                   if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")]
    exclude = ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"] + non_numeric
    all_features = [c for c in df.columns if c not in exclude]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]
    print(f"  {len(df):,} linhas, {len(all_features)} features ({len(sinan_features)} SINAN)")

    results_sinan = []
    results_full = []

    for i, fold in enumerate(FOLDS, 1):
        print(f"\n--- Fold {i}: train={fold['train']} → test={fold['test']} ---")

        r_s = train_eval_fold(df, sinan_features, fold["train"], fold["test"])
        if r_s:
            results_sinan.append(r_s)
            print(f"  SINAN-only:     R2_log={r_s['R2_log']:.4f}")

        r_f = train_eval_fold(df, all_features, fold["train"], fold["test"])
        if r_f:
            results_full.append(r_f)
            print(f"  INMET enriq.:   R2_log={r_f['R2_log']:.4f}")

        if r_s and r_f:
            delta = r_f["R2_log"] - r_s["R2_log"]
            print(f"  Delta: {delta:+.4f} {'✓' if delta > 0 else ''}")

    mean_r2_sinan = np.mean([r["R2_log"] for r in results_sinan])
    mean_r2_full = np.mean([r["R2_log"] for r in results_full])
    deltas = [f["R2_log"] - s["R2_log"] for s, f in zip(results_sinan, results_full)]
    folds_inmet_wins = sum(1 for d in deltas if d > 0)

    phase1_path = DATA_DIR / "08_walkforward_report.json"
    phase1_sinan_folds = []
    phase1_full_folds = []
    if phase1_path.exists():
        p1 = json.load(open(phase1_path))
        phase1_sinan_folds = p1.get("sinan_only", {}).get("folds", [])
        phase1_full_folds = p1.get("sinan_inmet", {}).get("folds", [])

    print("\nGerando figuras...")
    plot_walk_forward(results_sinan, results_full, phase1_sinan_folds, phase1_full_folds,
                      FIG_DIR / "fig_5yr_v2_walkforward.png")

    report = {
        "n_folds": len(FOLDS),
        "sinan_only": {
            "mean_R2_log": round(float(mean_r2_sinan), 4),
            "std_R2_log": round(float(np.std([r["R2_log"] for r in results_sinan])), 4),
            "folds": results_sinan,
        },
        "inmet_enriquecido": {
            "mean_R2_log": round(float(mean_r2_full), 4),
            "std_R2_log": round(float(np.std([r["R2_log"] for r in results_full])), 4),
            "folds": results_full,
        },
        "delta_mean": round(float(np.mean(deltas)), 4),
        "folds_inmet_better": int(folds_inmet_wins),
        "comparison_phase1": {
            "phase1_sinan_mean": round(float(np.mean([r["R2_log"] for r in phase1_sinan_folds])), 4) if phase1_sinan_folds else None,
            "phase1_inmet_mean": round(float(np.mean([r["R2_log"] for r in phase1_full_folds])), 4) if phase1_full_folds else None,
        },
    }
    with open(DATA_DIR / "17_walkforward_v2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO WALK-FORWARD v2 ({len(FOLDS)} folds):")
    print(f"  SINAN-only:     R²_log médio = {mean_r2_sinan:.4f}")
    print(f"  INMET enriq.:   R²_log médio = {mean_r2_full:.4f}")
    print(f"  Delta médio:    {np.mean(deltas):+.4f}")
    print(f"  Folds INMET ganha: {folds_inmet_wins}/{len(FOLDS)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-108: Walk-forward temporal validation — dataset filtrado 5yr.

Expanding window: treina com anos anteriores, testa no ano seguinte.
Folds: train→test: [2019]→2021, [2019,2021]→2023, [2019,2021,2023]→2024,
       [2019,2021,2023,2024]→2025, [2019,2021,2023,2024,2025]→2026.
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
MODEL_DIR = BASE_DIR / "models" / "walkforward_5yr"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

VALID_YEARS = [2019, 2021, 2023, 2024, 2025, 2026]

FOLDS = [
    {"train": [2019], "test": 2021},
    {"train": [2019, 2021], "test": 2023},
    {"train": [2019, 2021, 2023], "test": 2024},
    {"train": [2019, 2021, 2023, 2024], "test": 2025},
    {"train": [2019, 2021, 2023, 2024, 2025], "test": 2026},
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def train_eval_fold(df, all_features, train_years, test_year, label):
    train_data = df[df["ano"].isin(train_years)]
    test_data = df[df["ano"] == test_year]
    if len(train_data) == 0 or len(test_data) == 0:
        return None

    y_train = np.log1p(train_data[TARGET])
    y_test_log = np.log1p(test_data[TARGET])
    y_test_orig = test_data[TARGET].values

    # Use last 20% of train as eval set for early stopping
    n_train = len(train_data)
    split_idx = int(n_train * 0.8)
    train_sorted = train_data.sort_values(["ano", "semana_epidemiologica"])
    X_tr = train_sorted[all_features].iloc[:split_idx]
    y_tr = np.log1p(train_sorted[TARGET].iloc[:split_idx])
    X_ev = train_sorted[all_features].iloc[split_idx:]
    y_ev = np.log1p(train_sorted[TARGET].iloc[split_idx:])

    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=0)

    pred_log = model.predict(test_data[all_features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)
    r2_orig = r2_score(y_test_orig, pred_orig)
    r2_log = r2_score(y_test_log.values, pred_log)

    return {
        "train_years": train_years, "test_year": test_year,
        "label": label, "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 4), "R2": round(float(r2_orig), 4),
        "R2_log": round(float(r2_log), 4),
        "train_rows": len(train_data), "test_rows": len(test_data),
        "best_iteration": int(model.best_iteration),
    }


def plot_walk_forward(results_sinan, results_full, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    years = [r["test_year"] for r in results_sinan]
    r2_sinan = [r["R2_log"] for r in results_sinan]
    r2_full = [r["R2_log"] for r in results_full]
    rmse_sinan = [r["RMSE"] for r in results_sinan]
    rmse_full = [r["RMSE"] for r in results_full]

    ax = axes[0]
    ax.plot(years, r2_sinan, "o-", label="SINAN-only", color="#1976D2", linewidth=2, markersize=8)
    ax.plot(years, r2_full, "s-", label="SINAN+INMET", color="#E53935", linewidth=2, markersize=8)
    for i, (y, s, f) in enumerate(zip(years, r2_sinan, r2_full)):
        ax.annotate(f"{s:.3f}", (y, s), textcoords="offset points", xytext=(-15, 10), fontsize=8, color="#1976D2")
        ax.annotate(f"{f:.3f}", (y, f), textcoords="offset points", xytext=(5, -15), fontsize=8, color="#E53935")
    ax.set_xlabel("Ano de Teste", fontsize=11)
    ax.set_ylabel("R² (log)", fontsize=11)
    ax.set_title("R² por Fold (Walk-Forward)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(years, rmse_sinan, "o-", label="SINAN-only", color="#1976D2", linewidth=2, markersize=8)
    ax.plot(years, rmse_full, "s-", label="SINAN+INMET", color="#E53935", linewidth=2, markersize=8)
    ax.set_xlabel("Ano de Teste", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title("RMSE por Fold (Walk-Forward)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Validação Walk-Forward — Dataset Filtrado 5yr", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-108: Walk-Forward Validation (5yr)")
    print("=" * 60)

    print("\nCarregando dataset filtrado completo...")
    df = pd.read_parquet(DATA_DIR / "integrated_filtered.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df["notificacoes_t4"] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)
    df = df.dropna(subset=["notificacoes_t4"])
    print(f"  {len(df):,} linhas com target")

    exclude_cols = ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
    non_numeric = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")]
    all_features = [c for c in df.columns if c not in exclude_cols + non_numeric]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]
    print(f"  Features: {len(all_features)} total, {len(sinan_features)} SINAN-only")

    results_sinan = []
    results_full = []

    for i, fold in enumerate(FOLDS, 1):
        print(f"\n--- Fold {i}: train={fold['train']} → test={fold['test']} ---")

        r_sinan = train_eval_fold(df, sinan_features, fold["train"], fold["test"], "SINAN-only")
        if r_sinan:
            results_sinan.append(r_sinan)
            print(f"  SINAN-only:  R2_log={r_sinan['R2_log']:.4f}, RMSE={r_sinan['RMSE']:.2f}")

        r_full = train_eval_fold(df, all_features, fold["train"], fold["test"], "SINAN+INMET")
        if r_full:
            results_full.append(r_full)
            print(f"  SINAN+INMET: R2_log={r_full['R2_log']:.4f}, RMSE={r_full['RMSE']:.2f}")

        if r_sinan and r_full:
            delta = r_full["R2_log"] - r_sinan["R2_log"]
            print(f"  Delta R2_log: {delta:+.4f} {'(INMET ajuda)' if delta > 0 else ''}")

    # Summary
    mean_r2_sinan = np.mean([r["R2_log"] for r in results_sinan])
    mean_r2_full = np.mean([r["R2_log"] for r in results_full])
    std_r2_sinan = np.std([r["R2_log"] for r in results_sinan])
    std_r2_full = np.std([r["R2_log"] for r in results_full])
    mean_rmse_sinan = np.mean([r["RMSE"] for r in results_sinan])
    mean_rmse_full = np.mean([r["RMSE"] for r in results_full])

    deltas = [f["R2_log"] - s["R2_log"] for s, f in zip(results_sinan, results_full)]
    folds_inmet_wins = sum(1 for d in deltas if d > 0)

    cv_rmse = np.std([r["RMSE"] for r in results_full]) / mean_rmse_full * 100

    print(f"\n{'=' * 60}")
    print(f"  RESUMO WALK-FORWARD ({len(FOLDS)} folds):")
    print(f"  SINAN-only:  R2_log medio={mean_r2_sinan:.4f} ± {std_r2_sinan:.4f}")
    print(f"  SINAN+INMET: R2_log medio={mean_r2_full:.4f} ± {std_r2_full:.4f}")
    print(f"  Delta medio: {np.mean(deltas):+.4f}")
    print(f"  Folds onde INMET ganha: {folds_inmet_wins}/{len(FOLDS)}")
    print(f"  CV(RMSE): {cv_rmse:.1f}%")
    print(f"{'=' * 60}")

    # Figure
    print("\nGerando figuras...")
    plot_walk_forward(results_sinan, results_full,
                      FIG_DIR / "fig_5yr_walk_forward.png")

    report = {
        "n_folds": len(FOLDS),
        "sinan_only": {
            "mean_R2_log": round(float(mean_r2_sinan), 4),
            "std_R2_log": round(float(std_r2_sinan), 4),
            "mean_RMSE": round(float(mean_rmse_sinan), 2),
            "folds": results_sinan,
        },
        "sinan_inmet": {
            "mean_R2_log": round(float(mean_r2_full), 4),
            "std_R2_log": round(float(std_r2_full), 4),
            "mean_RMSE": round(float(mean_rmse_full), 2),
            "folds": results_full,
        },
        "delta_mean": round(float(np.mean(deltas)), 4),
        "folds_inmet_better": int(folds_inmet_wins),
        "cv_rmse_pct": round(float(cv_rmse), 1),
    }
    with open(DATA_DIR / "08_walkforward_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Salvo: 08_walkforward_report.json")


if __name__ == "__main__":
    main()

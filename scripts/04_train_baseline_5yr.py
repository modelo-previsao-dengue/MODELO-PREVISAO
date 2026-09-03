#!/usr/bin/env python3
"""US-104: Baseline SINAN-only vs SINAN+INMET no dataset filtrado 5yr."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "baseline_5yr"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, label):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    y_pred = np.maximum(model.predict(X_test), 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape_v = mape(y_test.values, y_pred)

    print(f"  [{label}] RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, MAPE={mape_v:.1f}%")
    return {
        "model": label, "RMSE": round(float(rmse), 4), "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4), "MAPE": round(float(mape_v), 2),
        "n_features": int(X_train.shape[1]),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 500,
    }, y_pred, model


def plot_comparison(sinan_metrics, full_metrics, out_path):
    metrics = ["RMSE", "MAE", "R2"]
    sinan_vals = [sinan_metrics[m] for m in metrics]
    full_vals = [full_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width / 2, sinan_vals, width, label="SINAN-only", color="#1976D2")
    bars2 = ax.bar(x + width / 2, full_vals, width, label="SINAN + INMET", color="#E53935")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title("Comparação Baseline: SINAN-only vs SINAN+INMET\n(Dataset Filtrado 2019-2026, 4.400 Municípios)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_residuals(y_test, y_pred_sinan, y_pred_full, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample = np.random.RandomState(42).choice(len(y_test), min(5000, len(y_test)), replace=False)
    for ax, preds, title in zip(axes, [y_pred_sinan, y_pred_full], ["SINAN-only", "SINAN+INMET"]):
        residuals = y_test.values[sample] - preds[sample]
        ax.scatter(preds[sample], residuals, alpha=0.15, s=3, color="#1976D2")
        ax.axhline(y=0, color="red", linewidth=1)
        ax.set_xlabel("Predição", fontsize=10)
        ax.set_ylabel("Resíduo", fontsize=10)
        ax.set_title(f"Resíduos — {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-104: Baseline SINAN-only vs SINAN+INMET (5yr)")
    print("=" * 60)

    print("\nCarregando splits...")
    train = pd.read_parquet(DATA_DIR / "train_5yr.parquet")
    val = pd.read_parquet(DATA_DIR / "val_5yr.parquet")
    test = pd.read_parquet(DATA_DIR / "test_5yr.parquet")
    train["ibge_municipio"] = train["ibge_municipio"].astype(str)
    val["ibge_municipio"] = val["ibge_municipio"].astype(str)
    test["ibge_municipio"] = test["ibge_municipio"].astype(str)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    all_features = [c for c in train.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]
    inmet_features = [c for c in all_features if is_inmet_feature(c)]
    print(f"  Total features: {len(all_features)}")
    print(f"  SINAN features: {len(sinan_features)}")
    print(f"  INMET features: {len(inmet_features)}")

    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    # SINAN-only
    print("\nTreinando SINAN-only...")
    sinan_metrics, y_pred_sinan, model_sinan = train_and_eval(
        train[sinan_features], y_train, val[sinan_features], y_val,
        test[sinan_features], y_test, "SINAN-only"
    )

    # SINAN+INMET
    print("Treinando SINAN+INMET...")
    full_metrics, y_pred_full, model_full = train_and_eval(
        train[all_features], y_train, val[all_features], y_val,
        test[all_features], y_test, "SINAN+INMET"
    )

    # Paired t-test
    errors_sinan = (y_test.values - y_pred_sinan) ** 2
    errors_full = (y_test.values - y_pred_full) ** 2
    t_stat, p_value = stats.ttest_rel(errors_sinan, errors_full)
    inmet_helps = p_value < 0.05 and t_stat > 0

    delta_rmse = (full_metrics["RMSE"] - sinan_metrics["RMSE"]) / sinan_metrics["RMSE"] * 100
    delta_r2 = full_metrics["R2"] - sinan_metrics["R2"]

    print(f"\n{'=' * 60}")
    print(f"  COMPARACAO:")
    print(f"  SINAN-only:  R2={sinan_metrics['R2']:.4f}  RMSE={sinan_metrics['RMSE']:.4f}")
    print(f"  SINAN+INMET: R2={full_metrics['R2']:.4f}  RMSE={full_metrics['RMSE']:.4f}")
    print(f"  Delta R2: {delta_r2:+.4f}")
    print(f"  Delta RMSE: {delta_rmse:+.2f}%")
    print(f"  t-test (INMET ajuda?): t={t_stat:.4f}, p={p_value:.6f}")
    print(f"  INMET melhora significativamente: {'SIM' if inmet_helps else 'NAO'}")
    print(f"{'=' * 60}")

    # Figures
    print("\nGerando figuras...")
    plot_comparison(sinan_metrics, full_metrics, FIG_DIR / "fig_5yr_baseline_comparison.png")
    plot_residuals(y_test, y_pred_sinan, y_pred_full, FIG_DIR / "fig_5yr_baseline_residuals.png")

    # Save models
    model_sinan.save_model(str(MODEL_DIR / "model_sinan_only.json"))
    model_full.save_model(str(MODEL_DIR / "model_sinan_inmet.json"))

    comparison = pd.DataFrame([sinan_metrics, full_metrics])
    comparison.to_csv(MODEL_DIR / "comparison.csv", index=False)

    report = {
        "sinan_only": sinan_metrics,
        "sinan_inmet": full_metrics,
        "delta_R2": round(float(delta_r2), 4),
        "delta_RMSE_pct": round(float(delta_rmse), 2),
        "paired_ttest": {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_value), 8)},
        "inmet_significant_improvement": bool(inmet_helps),
        "n_sinan_features": len(sinan_features),
        "n_inmet_features": len(inmet_features),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    with open(DATA_DIR / "04_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Salvo: 04_baseline_report.json")


if __name__ == "__main__":
    main()

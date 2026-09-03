#!/usr/bin/env python3
"""US-105: XGBoost regressao completa no dataset filtrado 5yr.

Treina com log1p(target), avalia na escala original e log.
Compara SINAN-only vs SINAN+INMET com metricas por UF e regiao.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "regression_5yr"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}

REGIAO_MAP = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte",
    "PA": "Norte", "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste",
    "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste",
    "DF": "Centro-Oeste",
}


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def train_model(X_train, y_train, X_val, y_val, label):
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    print(f"  [{label}] best_iteration={model.best_iteration}, n_features={X_train.shape[1]}")
    return model


def evaluate(y_true_orig, y_pred_orig, y_true_log, y_pred_log):
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)
    return {
        "RMSE": round(float(rmse_orig), 4),
        "MAE": round(float(mae_orig), 4),
        "R2": round(float(r2_orig), 4),
        "RMSE_log": round(float(rmse_log), 4),
        "R2_log": round(float(r2_log), 4),
    }


def plot_pred_vs_actual(y_true, y_pred_sinan, y_pred_full, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sample = np.random.RandomState(42).choice(len(y_true), min(5000, len(y_true)), replace=False)
    for ax, preds, title in zip(axes, [y_pred_sinan, y_pred_full], ["SINAN-only", "SINAN+INMET"]):
        ax.scatter(y_true[sample], preds[sample], alpha=0.15, s=3, color="#1976D2")
        lim = max(y_true[sample].max(), preds[sample].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlabel("Real (notificações t+4)", fontsize=10)
        ax.set_ylabel("Predição", fontsize=10)
        ax.set_title(f"Predição vs Real — {title}", fontsize=11)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle("XGBoost Regressão — Dataset Filtrado 5yr (log1p target)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_metrics_by_uf(uf_metrics_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    df = uf_metrics_df.sort_values("R2_log_full", ascending=True)

    ax = axes[0]
    y_pos = np.arange(len(df))
    ax.barh(y_pos - 0.2, df["R2_log_sinan"], 0.4, label="SINAN-only", color="#1976D2", alpha=0.8)
    ax.barh(y_pos + 0.2, df["R2_log_full"], 0.4, label="SINAN+INMET", color="#E53935", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["uf"], fontsize=8)
    ax.set_xlabel("R² (log)", fontsize=10)
    ax.set_title("R² por UF", fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    ax = axes[1]
    df_sorted = df.sort_values("delta_R2_log", ascending=True)
    colors = ["#E53935" if v < 0 else "#4CAF50" for v in df_sorted["delta_R2_log"]]
    ax.barh(np.arange(len(df_sorted)), df_sorted["delta_R2_log"], color=colors)
    ax.set_yticks(np.arange(len(df_sorted)))
    ax.set_yticklabels(df_sorted["uf"], fontsize=8)
    ax.set_xlabel("Delta R² (INMET - SINAN-only)", fontsize=10)
    ax.set_title("Contribuição INMET por UF", fontsize=11)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    fig.suptitle("Desempenho por UF — XGBoost Regressão 5yr", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-105: XGBoost Regressao (5yr)")
    print("=" * 60)

    print("\nCarregando splits...")
    train = pd.read_parquet(DATA_DIR / "train_5yr.parquet")
    val = pd.read_parquet(DATA_DIR / "val_5yr.parquet")
    test = pd.read_parquet(DATA_DIR / "test_5yr.parquet")
    for d in [train, val, test]:
        d["ibge_municipio"] = d["ibge_municipio"].astype(str)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    all_features = [c for c in train.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]

    # Log-transform target
    y_train_log = np.log1p(train[TARGET])
    y_val_log = np.log1p(val[TARGET])
    y_test_log = np.log1p(test[TARGET])
    y_test_orig = test[TARGET].values

    # Train SINAN-only
    print("\nTreinando SINAN-only (log1p target)...")
    model_sinan = train_model(
        train[sinan_features], y_train_log,
        val[sinan_features], y_val_log, "SINAN-only"
    )
    pred_log_sinan = model_sinan.predict(test[sinan_features])
    pred_orig_sinan = np.maximum(np.expm1(pred_log_sinan), 0)
    metrics_sinan = evaluate(y_test_orig, pred_orig_sinan, y_test_log.values, pred_log_sinan)
    print(f"  R2_log={metrics_sinan['R2_log']:.4f}, R2_orig={metrics_sinan['R2']:.4f}, RMSE={metrics_sinan['RMSE']:.2f}")

    # Train SINAN+INMET
    print("\nTreinando SINAN+INMET (log1p target)...")
    model_full = train_model(
        train[all_features], y_train_log,
        val[all_features], y_val_log, "SINAN+INMET"
    )
    pred_log_full = model_full.predict(test[all_features])
    pred_orig_full = np.maximum(np.expm1(pred_log_full), 0)
    metrics_full = evaluate(y_test_orig, pred_orig_full, y_test_log.values, pred_log_full)
    print(f"  R2_log={metrics_full['R2_log']:.4f}, R2_orig={metrics_full['R2']:.4f}, RMSE={metrics_full['RMSE']:.2f}")

    # Paired t-test on log-scale squared errors
    err_sinan = (y_test_log.values - pred_log_sinan) ** 2
    err_full = (y_test_log.values - pred_log_full) ** 2
    t_stat, p_value = stats.ttest_rel(err_sinan, err_full)
    inmet_helps = p_value < 0.05 and t_stat > 0

    delta_r2 = metrics_full["R2_log"] - metrics_sinan["R2_log"]

    print(f"\n{'=' * 60}")
    print(f"  SINAN-only:  R2_log={metrics_sinan['R2_log']:.4f}")
    print(f"  SINAN+INMET: R2_log={metrics_full['R2_log']:.4f}")
    print(f"  Delta R2_log: {delta_r2:+.4f}")
    print(f"  t-test: t={t_stat:.4f}, p={p_value:.8f}")
    print(f"  INMET melhora? {'SIM' if inmet_helps else 'NAO'}")
    print(f"{'=' * 60}")

    # Per-UF metrics
    print("\nMetricas por UF:")
    test_eval = test[ID_COLS + [TARGET]].copy()
    test_eval["pred_sinan"] = pred_orig_sinan
    test_eval["pred_full"] = pred_orig_full
    test_eval["pred_log_sinan"] = pred_log_sinan
    test_eval["pred_log_full"] = pred_log_full
    test_eval["y_log"] = y_test_log.values
    test_eval["uf"] = test_eval["ibge_municipio"].str[:2].map(UF_MAP)
    test_eval["regiao"] = test_eval["uf"].map(REGIAO_MAP)

    uf_rows = []
    for uf in sorted(test_eval["uf"].dropna().unique()):
        sub = test_eval[test_eval["uf"] == uf]
        if len(sub) < 50:
            continue
        r2_s = r2_score(sub["y_log"], sub["pred_log_sinan"])
        r2_f = r2_score(sub["y_log"], sub["pred_log_full"])
        rmse_s = np.sqrt(mean_squared_error(sub[TARGET], sub["pred_sinan"]))
        rmse_f = np.sqrt(mean_squared_error(sub[TARGET], sub["pred_full"]))
        uf_rows.append({
            "uf": uf, "regiao": REGIAO_MAP.get(uf, ""),
            "R2_log_sinan": round(r2_s, 4), "R2_log_full": round(r2_f, 4),
            "delta_R2_log": round(r2_f - r2_s, 4),
            "RMSE_sinan": round(rmse_s, 2), "RMSE_full": round(rmse_f, 2),
            "n": len(sub),
        })

    uf_df = pd.DataFrame(uf_rows)
    uf_df.to_csv(MODEL_DIR / "metrics_por_uf.csv", index=False)

    ufs_inmet_better = uf_df[uf_df["delta_R2_log"] > 0]
    print(f"  UFs onde INMET melhora: {len(ufs_inmet_better)}/{len(uf_df)}")
    for _, row in ufs_inmet_better.sort_values("delta_R2_log", ascending=False).head(5).iterrows():
        print(f"    {row['uf']}: delta={row['delta_R2_log']:+.4f}")

    # Per-region
    region_rows = []
    for reg in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        sub = test_eval[test_eval["regiao"] == reg]
        if len(sub) < 50:
            continue
        r2_s = r2_score(sub["y_log"], sub["pred_log_sinan"])
        r2_f = r2_score(sub["y_log"], sub["pred_log_full"])
        region_rows.append({
            "regiao": reg, "R2_log_sinan": round(r2_s, 4),
            "R2_log_full": round(r2_f, 4), "delta": round(r2_f - r2_s, 4), "n": len(sub),
        })
        print(f"  {reg:15s} SINAN={r2_s:.4f} INMET={r2_f:.4f} delta={r2_f - r2_s:+.4f}")

    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(MODEL_DIR / "metrics_por_regiao.csv", index=False)

    # Figures
    print("\nGerando figuras...")
    plot_pred_vs_actual(y_test_orig, pred_orig_sinan, pred_orig_full,
                        FIG_DIR / "fig_5yr_regression_pred_vs_real.png")
    plot_metrics_by_uf(uf_df, FIG_DIR / "fig_5yr_regression_uf_metrics.png")

    # Save models
    model_sinan.save_model(str(MODEL_DIR / "model_sinan_only.json"))
    model_full.save_model(str(MODEL_DIR / "model_sinan_inmet.json"))

    report = {
        "sinan_only": metrics_sinan,
        "sinan_inmet": metrics_full,
        "delta_R2_log": round(float(delta_r2), 4),
        "paired_ttest": {"t_stat": round(float(t_stat), 4), "p_value": float(p_value)},
        "inmet_significant_improvement": bool(inmet_helps),
        "ufs_inmet_better": int(len(ufs_inmet_better)),
        "ufs_total": int(len(uf_df)),
        "region_metrics": region_rows,
    }
    with open(DATA_DIR / "05_regression_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Salvo: 05_regression_report.json")


if __name__ == "__main__":
    main()

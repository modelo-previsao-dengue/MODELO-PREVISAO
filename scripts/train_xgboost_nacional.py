"""
Train XGBoost national model on ALL municipalities (not just DF).
Uses the full model-ready dataset (5.8M rows train, 947K test).

Usage:
    python scripts/train_xgboost_nacional.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "nacional"
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET_REG = "notificacoes_t4"
TARGET_CLS = "risco_surto_t4"
CLASS_NAMES = ["Low", "Medium", "High", "Outbreak"]


def split_xy(df, target):
    drop_cols = [c for c in META_COLS + [TARGET_REG, TARGET_CLS] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].values
    return X, y


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def main():
    print("=" * 60)
    print("  XGBoost Nacional — All Municipalities")
    print("=" * 60)

    t0 = time.time()
    print("\nLoading data...")
    df_train = pd.read_parquet(DATA_DIR / "model_ready" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "model_ready" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "model_ready" / "test.parquet")

    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    print(f"  Train+val: {len(df_trainval):,} rows")
    print(f"  Test:      {len(df_test):,} rows")
    print(f"  Municipios train: {df_trainval['ibge_municipio'].nunique():,}")
    print(f"  Municipios test:  {df_test['ibge_municipio'].nunique():,}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    X_trainval, y_reg_trainval = split_xy(df_trainval, TARGET_REG)
    X_test, y_reg_test = split_xy(df_test, TARGET_REG)
    _, y_cls_trainval = split_xy(df_trainval, TARGET_CLS)
    _, y_cls_test = split_xy(df_test, TARGET_CLS)

    y_reg_trainval_log = np.log1p(y_reg_trainval)
    y_reg_test_log = np.log1p(y_reg_test)

    feature_names = list(X_trainval.columns)
    print(f"  Features: {len(feature_names)}")

    # ── Regression ──
    print("\n── Training Nacional Regression ──")
    t1 = time.time()
    params_reg = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 500,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": -1,
    }
    model_reg = xgb.XGBRegressor(**params_reg)
    model_reg.fit(X_trainval, y_reg_trainval_log)
    elapsed_reg = time.time() - t1
    print(f"  Training time: {elapsed_reg:.1f}s")

    pred_reg_log = model_reg.predict(X_test)
    pred_reg_orig = np.expm1(np.maximum(pred_reg_log, 0))

    r2_log = r2_score(y_reg_test_log, pred_reg_log)
    r2_orig = r2_score(y_reg_test, pred_reg_orig)
    mae_log = mean_absolute_error(y_reg_test_log, pred_reg_log)
    mae_orig = mean_absolute_error(y_reg_test, pred_reg_orig)
    rmse_log = np.sqrt(mean_squared_error(y_reg_test_log, pred_reg_log))

    print(f"  R²_log:   {r2_log:.4f}")
    print(f"  R²_orig:  {r2_orig:.4f}")
    print(f"  MAE_log:  {mae_log:.4f}")
    print(f"  MAE_orig: {mae_orig:.1f}")
    print(f"  RMSE_log: {rmse_log:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "xgb_reg_nacional.ubj"
    model_reg.save_model(str(model_path))
    print(f"  Model saved: {model_path} ({model_path.stat().st_size/1024/1024:.1f} MB)")

    # ── Classification ──
    print("\n── Training Nacional Classification ──")
    t2 = time.time()
    params_cls = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 500,
        "objective": "multi:softprob",
        "num_class": 4,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "n_jobs": -1,
    }
    model_cls = xgb.XGBClassifier(**params_cls)
    model_cls.fit(X_trainval, y_cls_trainval)
    elapsed_cls = time.time() - t2
    print(f"  Training time: {elapsed_cls:.1f}s")

    pred_cls = model_cls.predict(X_test)
    f1_macro = f1_score(y_cls_test, pred_cls, average="macro", zero_division=0)
    print(f"  F1_macro: {f1_macro:.4f}")

    model_path_cls = OUTPUT_DIR / "xgb_cls_nacional.ubj"
    model_cls.save_model(str(model_path_cls))
    print(f"  Model saved: {model_path_cls}")

    # ── Evaluate per-state (UF) ──
    print("\n── Per-State Analysis ──")
    df_test_eval = df_test.copy()
    df_test_eval["pred_log"] = pred_reg_log
    df_test_eval["pred_orig"] = pred_reg_orig
    df_test_eval["pred_cls"] = pred_cls

    uf_map = {
        "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
        "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
        "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
        "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
        "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
        "52": "GO", "53": "DF",
    }
    df_test_eval["uf"] = df_test_eval["ibge_municipio"].str[:2].map(uf_map)

    uf_results = []
    for uf, grp in df_test_eval.groupby("uf"):
        if len(grp) < 10:
            continue
        r2_uf = r2_score(np.log1p(grp[TARGET_REG].values), grp["pred_log"].values)
        mae_uf = mean_absolute_error(grp[TARGET_REG].values, grp["pred_orig"].values)
        f1_uf = f1_score(grp[TARGET_CLS].values, grp["pred_cls"].values, average="macro", zero_division=0)
        uf_results.append({"UF": uf, "n_rows": len(grp), "R2_log": r2_uf, "MAE_orig": mae_uf, "F1_macro": f1_uf})

    uf_df = pd.DataFrame(uf_results).sort_values("R2_log", ascending=False)
    print(uf_df.to_string(index=False))

    # ── Fig: Nacional vs DF comparison ──
    # DF metrics
    df_test_df = df_test[df_test["ibge_municipio"] == "5300108"]
    if len(df_test_df) > 0:
        X_df, y_df = split_xy(df_test_df, TARGET_REG)
        pred_df_log = model_reg.predict(X_df)
        r2_df_log = r2_score(np.log1p(y_df), pred_df_log)

        fig, ax = plt.subplots(figsize=(12, 6))
        models = ["Nacional\n(all munic.)", "DF only\n(nacional model)"]
        vals = [r2_log, r2_df_log]
        colors = ["#1976D2", "#E53935"]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
        ax.set_ylabel("R² (log scale)")
        ax.set_title("XGBoost Nacional: Performance on All Municipalities vs DF")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        fig.tight_layout()
        savefig(fig, "fig_nacional_vs_df")
        print(f"\n  Nacional R²_log={r2_log:.4f}, DF (via nacional model) R²_log={r2_df_log:.4f}")

    # ── Fig: Per-UF heatmap ──
    fig, ax = plt.subplots(figsize=(14, 6))
    uf_sorted = uf_df.sort_values("R2_log", ascending=True)
    colors = ["#E53935" if v < 0 else "#4CAF50" for v in uf_sorted["R2_log"]]
    ax.barh(uf_sorted["UF"], uf_sorted["R2_log"], color=colors, edgecolor="white")
    ax.set_xlabel("R² (log scale)")
    ax.set_title("XGBoost Nacional — R² por UF (teste 2023–2026)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.tight_layout()
    savefig(fig, "fig_nacional_r2_por_uf")

    # ── Save metrics ──
    metrics = {
        "model": "XGBoost Nacional",
        "train_rows": len(df_trainval),
        "test_rows": len(df_test),
        "features": len(feature_names),
        "R2_log": r2_log,
        "R2_orig": r2_orig,
        "MAE_log": mae_log,
        "MAE_orig": mae_orig,
        "RMSE_log": rmse_log,
        "F1_macro": f1_macro,
        "train_time_reg_s": elapsed_reg,
        "train_time_cls_s": elapsed_cls,
    }
    with open(OUTPUT_DIR / "metrics_nacional.json", "w") as f:
        json.dump(metrics, f, indent=2)

    uf_df.to_csv(OUTPUT_DIR / "metrics_por_uf.csv", index=False)

    print("\n" + "=" * 60)
    print("  NACIONAL SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(df_trainval):,} rows ({df_trainval['ibge_municipio'].nunique():,} municipios)")
    print(f"  Test:  {len(df_test):,} rows ({df_test['ibge_municipio'].nunique():,} municipios)")
    print(f"  Regression:      R²_log={r2_log:.4f}  MAE_orig={mae_orig:.1f}")
    print(f"  Classification:  F1_macro={f1_macro:.4f}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

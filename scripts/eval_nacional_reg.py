"""
Evaluate the saved nacional regression model and generate figures.
No classification training needed - Overleaf only uses regression R² per UF.
"""
import json, time, warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "nacional"
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

META_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET_REG = "notificacoes_t4"
TARGET_CLS = "risco_surto_t4"


def split_xy(df, target):
    drop_cols = [c for c in META_COLS + [TARGET_REG, TARGET_CLS] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].values
    return X, y


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    t0 = time.time()
    print("Loading data...")
    df_test = pd.read_parquet(DATA_DIR / "model_ready" / "test.parquet")
    print(f"Test: {len(df_test):,} rows, {df_test['ibge_municipio'].nunique()} municipios")

    X_test, y_reg_test = split_xy(df_test, TARGET_REG)
    y_reg_test_log = np.log1p(y_reg_test)

    print("Loading saved regression model...")
    model_reg = xgb.XGBRegressor()
    model_reg.load_model(str(OUTPUT_DIR / "xgb_reg_nacional.ubj"))

    print("Predicting...")
    pred_reg_log = model_reg.predict(X_test)
    pred_reg_orig = np.expm1(np.maximum(pred_reg_log, 0))

    r2_log = r2_score(y_reg_test_log, pred_reg_log)
    r2_orig = r2_score(y_reg_test, pred_reg_orig)
    mae_log = mean_absolute_error(y_reg_test_log, pred_reg_log)
    mae_orig = mean_absolute_error(y_reg_test, pred_reg_orig)
    rmse_log = np.sqrt(mean_squared_error(y_reg_test_log, pred_reg_log))

    print(f"R2_log={r2_log:.4f}  R2_orig={r2_orig:.4f}  MAE_log={mae_log:.4f}  MAE_orig={mae_orig:.1f}")

    # Per-UF
    print("Per-UF analysis...")
    df_test_eval = df_test.copy()
    df_test_eval["pred_log"] = pred_reg_log
    df_test_eval["pred_orig"] = pred_reg_orig

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
        uf_results.append({"UF": uf, "n_rows": len(grp), "R2_log": round(r2_uf, 4), "MAE_orig": round(mae_uf, 1)})

    uf_df = pd.DataFrame(uf_results).sort_values("R2_log", ascending=False)
    print(uf_df.to_string(index=False))

    # Fig 1: Nacional vs DF
    df_test_df = df_test[df_test["ibge_municipio"] == "5300108"]
    r2_df_log = None
    if len(df_test_df) > 0:
        X_df, y_df = split_xy(df_test_df, TARGET_REG)
        pred_df_log = model_reg.predict(X_df)
        r2_df_log = r2_score(np.log1p(y_df), pred_df_log)

        fig, ax = plt.subplots(figsize=(10, 5))
        models = ["Nacional\n(todos municípios)", "DF\n(modelo nacional)"]
        vals = [r2_log, r2_df_log]
        colors = ["#1976D2", "#E53935"]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=14)
        ax.set_ylabel("R² (escala logarítmica)", fontsize=12)
        ax.set_title("XGBoost Nacional: Desempenho Geral vs Distrito Federal", fontsize=13)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        fig.tight_layout()
        savefig(fig, "fig_nacional_vs_df")
        print(f"Nacional R2_log={r2_log:.4f}, DF R2_log={r2_df_log:.4f}")

    # Fig 2: Per-UF bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    uf_sorted = uf_df.sort_values("R2_log", ascending=True)
    colors = ["#E53935" if v < 0 else "#4CAF50" for v in uf_sorted["R2_log"]]
    bars = ax.barh(uf_sorted["UF"], uf_sorted["R2_log"], color=colors, edgecolor="white")
    for bar, val in zip(bars, uf_sorted["R2_log"]):
        x = max(val, 0) + 0.01
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=8)
    ax.set_xlabel("R² (escala logarítmica)", fontsize=12)
    ax.set_title("XGBoost Nacional — R² por UF (teste 2023–2026)", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.tight_layout()
    savefig(fig, "fig_nacional_r2_por_uf")

    # Save metrics
    metrics = {
        "model": "XGBoost Nacional",
        "test_rows": int(len(df_test)),
        "n_municipios": int(df_test["ibge_municipio"].nunique()),
        "features": int(len(list(X_test.columns))),
        "R2_log": float(r2_log),
        "R2_orig": float(r2_orig),
        "MAE_log": float(mae_log),
        "MAE_orig": float(mae_orig),
        "RMSE_log": float(rmse_log),
        "R2_df_log": float(r2_df_log) if r2_df_log is not None else None,
    }
    with open(OUTPUT_DIR / "metrics_nacional.json", "w") as f:
        json.dump(metrics, f, indent=2)
    uf_df.to_csv(OUTPUT_DIR / "metrics_por_uf.csv", index=False)

    print(f"\nDONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

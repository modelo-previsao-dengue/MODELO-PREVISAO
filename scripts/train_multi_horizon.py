#!/usr/bin/env python3
"""US-013: Multi-horizonte (t+1, t+2, t+4, t+8)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
MULTI_DIR = BASE_DIR / "models" / "multi_horizon"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
HORIZONS = [1, 2, 4, 8]

EXCLUDE_COLS = [
    "ano_semana", "week_start", "municipio", "uf", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field",
    "ibge_municipio", "notificacoes_t4", "risco_surto_t4",
]


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    MULTI_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dataset integrado...")
    df = pd.read_parquet(INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet")
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])

    # Load best hyperparameters
    tuned_path = BASE_DIR / "models" / "xgb_regression_tuned" / "metrics.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            best_params = json.load(f).get("best_params", {})
        best_params.pop("early_stopping_rounds", None)
        print("  Usando hiperparâmetros tuned")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        print("  Usando hiperparâmetros MVP")

    best_params["tree_method"] = "hist"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["early_stopping_rounds"] = 50

    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.99]
    exclude = set(EXCLUDE_COLS + high_missing)
    feature_cols = [c for c in df.columns if c not in exclude and c != "notificacoes"]

    results = []
    for h in HORIZONS:
        print(f"\n=== Horizonte t+{h} ===")

        data = df.copy()
        data[f"target_t{h}"] = data.groupby("ibge_municipio")["notificacoes"].shift(-h)
        data = data.dropna(subset=[f"target_t{h}"])

        target_col = f"target_t{h}"
        feat = [c for c in feature_cols if c != target_col]

        train = data[data["ano"] <= 2019]
        val = data[(data["ano"] >= 2020) & (data["ano"] <= 2022)]
        test = data[data["ano"] >= 2023]

        # Subsample training to avoid OOM on 16GB machine
        TRAIN_MAX = 2_000_000
        if len(train) > TRAIN_MAX:
            train = train.sample(TRAIN_MAX, random_state=42)

        print(f"  Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            train[feat], train[target_col],
            eval_set=[(val[feat], val[target_col])],
            verbose=0,
        )

        y_pred = np.maximum(model.predict(test[feat]), 0)
        y_test = test[target_col].values

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape_v = mape(y_test, y_pred)

        results.append({
            "horizon": h,
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape_v), 2),
        })
        print(f"  RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        model.save_model(str(MULTI_DIR / f"model_t{h}.json"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(MULTI_DIR / "comparison.csv", index=False)

    print(f"\n=== Comparação Multi-Horizonte ===")
    print(results_df.to_string(index=False))

    # Degradation check: RMSE(t+8) <= 2x RMSE(t+1)
    rmse_t1 = results_df.loc[results_df["horizon"] == 1, "RMSE"].values[0]
    rmse_t8 = results_df.loc[results_df["horizon"] == 8, "RMSE"].values[0]
    ratio = rmse_t8 / rmse_t1 if rmse_t1 > 0 else np.inf

    summary = {
        "horizons": HORIZONS,
        "results": results,
        "rmse_t8_over_t1": round(float(ratio), 2),
        "degradation_acceptable": bool(ratio <= 2.0),
    }
    with open(MULTI_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRMSE(t+8)/RMSE(t+1) = {ratio:.2f} ({'OK ≤ 2x' if ratio <= 2 else 'ALTO > 2x'})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([r["horizon"] for r in results], [r["RMSE"] for r in results], "o-", linewidth=2)
    axes[0].set_xlabel("Horizonte (semanas)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Degradação RMSE vs Horizonte")
    axes[0].set_xticks(HORIZONS)

    axes[1].plot([r["horizon"] for r in results], [r["R2"] for r in results], "o-", color="green", linewidth=2)
    axes[1].set_xlabel("Horizonte (semanas)")
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² vs Horizonte")
    axes[1].set_xticks(HORIZONS)

    plt.tight_layout()
    plt.savefig(MULTI_DIR / "degradation_plot.png", dpi=150)
    plt.close()
    print(f"\nResultados salvos em {MULTI_DIR}")


if __name__ == "__main__":
    main()

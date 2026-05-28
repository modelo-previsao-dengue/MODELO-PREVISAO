#!/usr/bin/env python3
"""US-012: Walk-forward validation temporal."""

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
MODEL_READY = BASE_DIR / "data" / "model_ready"
WF_DIR = BASE_DIR / "models" / "walk_forward_results"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

MIN_TRAIN_YEARS = 5


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    WF_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados (amostra por ano para 16GB RAM)...")
    import pyarrow.parquet as pq
    import gc
    frames = []
    ROWS_PER_YEAR = 30_000
    for f in ["train.parquet", "val.parquet", "test.parquet"]:
        tbl = pq.read_table(MODEL_READY / f, columns=None)
        chunk = tbl.to_pandas()
        del tbl; gc.collect()
        sampled = chunk.groupby("ano", group_keys=False).apply(
            lambda g: g.sample(min(len(g), ROWS_PER_YEAR), random_state=42)
        )
        frames.append(sampled)
        del chunk, sampled; gc.collect()
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} rows (~{ROWS_PER_YEAR}/year sample)")

    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]
    years = sorted(df["ano"].unique())
    print(f"  Anos disponíveis: {years[0]}-{years[-1]} ({len(years)} anos)")

    # Load best hyperparameters
    tuned_path = BASE_DIR / "models" / "xgb_regression_tuned" / "metrics.json"
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned = json.load(f)
        best_params = tuned.get("best_params", {})
        best_params.pop("early_stopping_rounds", None)
        print(f"  Usando hiperparâmetros tuned")
    else:
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        print(f"  Usando hiperparâmetros MVP")

    best_params["tree_method"] = "hist"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["early_stopping_rounds"] = 50

    # Walk-forward: expanding window, test every 3 years for tractability
    start_test_year = years[0] + MIN_TRAIN_YEARS
    all_test_years = [y for y in years if y >= start_test_year]
    test_years = all_test_years[::3]
    if all_test_years[-1] not in test_years:
        test_years.append(all_test_years[-1])
    print(f"\nWalk-forward: {len(test_years)} folds (test years: {test_years})")

    results = []
    for test_year in test_years:
        train_data = df[df["ano"] < test_year]
        test_data = df[df["ano"] == test_year]

        if len(test_data) == 0:
            continue

        # Use year before test as validation for early stopping
        val_year = test_year - 1
        val_data = train_data[train_data["ano"] == val_year]
        train_only = train_data[train_data["ano"] < val_year]

        if len(train_only) == 0 or len(val_data) == 0:
            continue

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            train_only[feature_cols], train_only[TARGET],
            eval_set=[(val_data[feature_cols], val_data[TARGET])],
            verbose=0,
        )

        y_pred = np.maximum(model.predict(test_data[feature_cols]), 0)
        y_test = test_data[TARGET].values

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape_v = mape(y_test, y_pred)

        results.append({
            "test_year": test_year,
            "train_years": f"{years[0]}-{val_year - 1}",
            "n_train": len(train_only),
            "n_test": len(test_data),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape_v, 2),
        })
        print(f"  [{test_year}] RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(WF_DIR / "walk_forward_results.csv", index=False)

    # Summary statistics
    mean_rmse = results_df["RMSE"].mean()
    std_rmse = results_df["RMSE"].std()
    cv_rmse = std_rmse / mean_rmse * 100

    summary = {
        "n_folds": len(results),
        "mean_RMSE": round(float(mean_rmse), 4),
        "std_RMSE": round(float(std_rmse), 4),
        "cv_RMSE_pct": round(float(cv_rmse), 2),
        "mean_R2": round(float(results_df["R2"].mean()), 4),
        "stable": bool(cv_rmse < 30),
    }
    with open(WF_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResumo Walk-Forward:")
    print(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} (CV={cv_rmse:.1f}%)")
    print(f"  Mean R²: {results_df['R2'].mean():.4f}")
    print(f"  Estável (CV < 30%): {'SIM' if cv_rmse < 30 else 'NÃO'}")

    # Plot degradation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(results_df["test_year"], results_df["RMSE"], "o-")
    axes[0].set_xlabel("Ano de Teste")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE ao Longo dos Folds")
    axes[0].axhline(mean_rmse, color="r", linestyle="--", alpha=0.5, label=f"Média={mean_rmse:.1f}")
    axes[0].legend()

    axes[1].plot(results_df["test_year"], results_df["R2"], "o-", color="green")
    axes[1].set_xlabel("Ano de Teste")
    axes[1].set_ylabel("R²")
    axes[1].set_title("R² ao Longo dos Folds")
    axes[1].axhline(results_df["R2"].mean(), color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(WF_DIR / "walk_forward_degradation.png", dpi=150)
    plt.close()
    print(f"\nResultados salvos em {WF_DIR}")


if __name__ == "__main__":
    main()

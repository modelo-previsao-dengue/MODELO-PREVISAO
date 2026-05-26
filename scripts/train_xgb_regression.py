#!/usr/bin/env python3
"""US-007: XGBoost Regressão MVP — previsão de notificações t+4."""

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
MODEL_DIR = BASE_DIR / "models" / "xgb_regression_mvp"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"


def load_split(name):
    df = pd.read_parquet(MODEL_READY / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, df


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando splits...")
    X_train, y_train, df_train = load_split("train")
    X_val, y_val, df_val = load_split("val")
    X_test, y_test, df_test = load_split("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\nTreinando XGBRegressor MVP...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    # Predictions on test
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape_val = mape(y_test.values, y_pred)

    metrics = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "MAPE": round(mape_val, 2)}
    print(f"\n=== Métricas no Teste (2023-2026) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False)
    importance.head(20).to_csv(MODEL_DIR / "feature_importance_top20.csv", index=False)
    print(f"\nTop-10 features por gain:")
    print(importance.head(10).to_string(index=False))

    # Save model
    model.save_model(str(MODEL_DIR / "model.json"))
    print(f"\nModelo salvo: {MODEL_DIR / 'model.json'}")

    # Training log
    results = model.evals_result()
    log_df = pd.DataFrame({
        "epoch": range(len(results["validation_0"]["rmse"])),
        "train_rmse": results["validation_0"]["rmse"],
        "val_rmse": results["validation_1"]["rmse"],
    })
    log_df.to_csv(MODEL_DIR / "training_log.csv", index=False)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Real vs Predicted scatter
    sample_idx = np.random.RandomState(42).choice(len(y_test), min(50000, len(y_test)), replace=False)
    axes[0].scatter(y_test.values[sample_idx], y_pred[sample_idx], alpha=0.1, s=1)
    axes[0].plot([0, y_test.max()], [0, y_test.max()], "r--", lw=1)
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Previsto")
    axes[0].set_title(f"Real vs Previsto (R²={r2:.3f})")

    # Training curves
    axes[1].plot(log_df["epoch"], log_df["train_rmse"], label="Train")
    axes[1].plot(log_df["epoch"], log_df["val_rmse"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Curva de Treinamento")
    axes[1].legend()

    # Feature importance bar
    top20 = importance.head(20)
    axes[2].barh(range(len(top20)), top20["gain"].values)
    axes[2].set_yticks(range(len(top20)))
    axes[2].set_yticklabels(top20["feature"].values, fontsize=7)
    axes[2].set_title("Top-20 Feature Importance")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "plots.png", dpi=150)
    plt.close()
    print(f"Gráficos salvos: {MODEL_DIR / 'plots.png'}")


if __name__ == "__main__":
    main()

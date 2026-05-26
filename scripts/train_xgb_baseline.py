#!/usr/bin/env python3
"""US-009: Baseline SINAN-only vs SINAN+INMET comparativo."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY = BASE_DIR / "data" / "model_ready"
MODEL_DIR = BASE_DIR / "models" / "xgb_baseline_sinan_only"

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

    print(f"\n  [{label}] RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, MAPE={mape_v:.2f}%")
    return {
        "model": label, "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "R2": round(r2, 4), "MAPE": round(mape_v, 2),
    }, y_pred, model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando splits...")
    train = pd.read_parquet(MODEL_READY / "train.parquet")
    val = pd.read_parquet(MODEL_READY / "val.parquet")
    test = pd.read_parquet(MODEL_READY / "test.parquet")

    all_features = [c for c in train.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]
    print(f"  Todas features: {len(all_features)}")
    print(f"  Features SINAN-only: {len(sinan_features)}")
    print(f"  Features INMET: {len(all_features) - len(sinan_features)}")

    y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]

    # SINAN+INMET (full)
    print("\nTreinando SINAN+INMET (full)...")
    full_metrics, y_pred_full, _ = train_and_eval(
        train[all_features], y_train, val[all_features], y_val,
        test[all_features], y_test, "SINAN+INMET"
    )

    # SINAN-only
    print("Treinando SINAN-only...")
    sinan_metrics, y_pred_sinan, model_sinan = train_and_eval(
        train[sinan_features], y_train, val[sinan_features], y_val,
        test[sinan_features], y_test, "SINAN-only"
    )

    # Statistical test (paired t-test on squared errors)
    errors_full = (y_test.values - y_pred_full) ** 2
    errors_sinan = (y_test.values - y_pred_sinan) ** 2
    t_stat, p_value = stats.ttest_rel(errors_sinan, errors_full)

    comparison = pd.DataFrame([sinan_metrics, full_metrics])
    comparison["delta_RMSE"] = ""
    comparison.loc[comparison["model"] == "SINAN+INMET", "delta_RMSE"] = (
        f"{((full_metrics['RMSE'] - sinan_metrics['RMSE']) / sinan_metrics['RMSE'] * 100):+.2f}%"
    )

    print(f"\n=== Comparação ===")
    print(comparison.to_string(index=False))
    print(f"\nTeste pareado (H1: SINAN+INMET < SINAN-only):")
    print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.6f}")
    significant = p_value < 0.05 and t_stat > 0
    print(f"  Significativo (p<0.05): {'SIM ✓' if significant else 'NÃO'}")

    comparison.to_csv(MODEL_DIR / "comparison.csv", index=False)
    comparison.to_markdown(MODEL_DIR / "comparison.md") if hasattr(comparison, "to_markdown") else None

    result = {
        "sinan_only": sinan_metrics,
        "sinan_inmet": full_metrics,
        "paired_ttest": {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_value), 6)},
        "significant_improvement": bool(significant),
    }
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    model_sinan.save_model(str(MODEL_DIR / "model_sinan_only.json"))
    print(f"\nResultados salvos em {MODEL_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-011: Explicabilidade com SHAP."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY = BASE_DIR / "data" / "model_ready"
SHAP_DIR = BASE_DIR / "models" / "shap_analysis"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

SAMPLE_SIZE = 50000

CLIMATE_FEATURES = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Load best model (tuned if available, else MVP)
    tuned_path = BASE_DIR / "models" / "xgb_regression_tuned" / "model.json"
    mvp_path = BASE_DIR / "models" / "xgb_regression_mvp" / "model.json"
    model_path = tuned_path if tuned_path.exists() else mvp_path
    print(f"Carregando modelo: {model_path.parent.name}")

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    print("Carregando dados de teste...")
    test = pd.read_parquet(MODEL_READY / "test.parquet")
    feature_cols = [c for c in test.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]

    # Sample for SHAP
    if len(test) > SAMPLE_SIZE:
        sample = test.sample(SAMPLE_SIZE, random_state=42)
    else:
        sample = test
    X_sample = sample[feature_cols]
    y_sample = sample[TARGET]
    print(f"  Sample: {len(X_sample)} linhas")

    print("Calculando SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (beeswarm)
    print("Gerando summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top-5 climate features dependence plots
    climate_cols_present = [c for c in feature_cols if any(c.startswith(cf) for cf in CLIMATE_FEATURES)]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    climate_importance = [(c, mean_abs_shap[feature_cols.index(c)]) for c in climate_cols_present]
    climate_importance.sort(key=lambda x: x[1], reverse=True)
    top5_climate = [c for c, _ in climate_importance[:5]]

    print(f"Top-5 climate features: {top5_climate}")
    fig, axes = plt.subplots(1, min(5, len(top5_climate)), figsize=(20, 4))
    if len(top5_climate) == 1:
        axes = [axes]
    for i, feat in enumerate(top5_climate[:5]):
        idx = feature_cols.index(feat)
        ax = axes[i] if len(top5_climate) > 1 else axes[0]
        ax.scatter(X_sample[feat].values, shap_values[:, idx], alpha=0.1, s=1)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_dependence_climate_top5.png", dpi=150)
    plt.close()

    # Feature importance ranking from SHAP
    shap_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance["is_climate"] = shap_importance["feature"].apply(
        lambda c: any(c.startswith(cf) for cf in CLIMATE_FEATURES)
    )
    shap_importance.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)

    # Climate features in top-20
    top20 = shap_importance.head(20)
    climate_in_top20 = top20[top20["is_climate"]]["feature"].tolist()

    print(f"\nTop-20 SHAP features:")
    print(top20[["feature", "mean_abs_shap", "is_climate"]].to_string(index=False))

    # Validation: climate features with lag 2-8 should have significant importance
    lag_climate = [c for c in climate_cols_present if any(f"lag_{l}" in c for l in [2, 4, 8])]
    lag_climate_ranks = {c: int(shap_importance[shap_importance["feature"] == c].index[0] + 1)
                         for c in lag_climate if c in shap_importance["feature"].values}

    report = {
        "model_source": model_path.parent.name,
        "sample_size": len(X_sample),
        "top5_climate_features": top5_climate,
        "climate_in_top20": climate_in_top20,
        "n_climate_in_top20": len(climate_in_top20),
        "lag_climate_ranks": lag_climate_ranks,
    }
    with open(SHAP_DIR / "shap_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nClimate features no top-20: {len(climate_in_top20)} → {climate_in_top20}")
    print(f"Lag climate feature ranks: {lag_climate_ranks}")
    print(f"\nResultados salvos em {SHAP_DIR}")


if __name__ == "__main__":
    main()

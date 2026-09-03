#!/usr/bin/env python3
"""US-107: SHAP explainability analysis — dataset filtrado 5yr."""

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
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "regression_5yr"
SHAP_DIR = BASE_DIR / "models" / "shap_5yr"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

SAMPLE_SIZE = 50000


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-107: SHAP Explainability Analysis (5yr)")
    print("=" * 60)

    print("\nCarregando modelo e dados de teste...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_DIR / "model_sinan_inmet.json"))

    test = pd.read_parquet(DATA_DIR / "test_5yr.parquet")
    test["ibge_municipio"] = test["ibge_municipio"].astype(str)
    all_features = [c for c in test.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    X_test = test[all_features]
    print(f"  Test: {len(X_test):,} linhas, {len(all_features)} features")

    # Sample for SHAP
    np.random.seed(42)
    idx = np.random.choice(len(X_test), min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx]
    print(f"  Amostra SHAP: {len(X_sample):,} linhas")

    # TreeExplainer
    print("\nCalculando SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print(f"  Shape: {shap_values.shape}")

    # Feature importance
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": all_features,
        "mean_abs_shap": shap_importance,
        "is_inmet": [is_inmet_feature(f) for f in all_features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)

    # Print top-30
    print("\nTop-30 features por importancia SHAP:")
    for _, row in importance_df.head(30).iterrows():
        tag = "[INMET]" if row["is_inmet"] else "[SINAN]"
        print(f"  {int(row['rank']):3d}. {tag:8s} {row['feature']:40s} SHAP={row['mean_abs_shap']:.4f}")

    # Climate features ranking
    climate_feats = importance_df[importance_df["is_inmet"]]
    sinan_feats = importance_df[~importance_df["is_inmet"]]
    n_climate_top20 = len(climate_feats[climate_feats["rank"] <= 20])
    n_climate_top50 = len(climate_feats[climate_feats["rank"] <= 50])

    print(f"\n  Features climaticas no top-20: {n_climate_top20}/20")
    print(f"  Features climaticas no top-50: {n_climate_top50}/50")
    print(f"  Melhor feature climatica: rank {climate_feats['rank'].min()} ({climate_feats.iloc[0]['feature']})")
    print(f"  SHAP medio (INMET): {climate_feats['mean_abs_shap'].mean():.4f}")
    print(f"  SHAP medio (SINAN): {sinan_feats['mean_abs_shap'].mean():.4f}")

    # Summary plot (top-20)
    print("\nGerando figuras SHAP...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title("SHAP Summary — Top-20 Features (Dataset Filtrado 5yr)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_5yr_shap_summary_top20.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Salvo: fig_5yr_shap_summary_top20.png")

    # Bar plot top-30
    fig, ax = plt.subplots(figsize=(10, 10))
    top30 = importance_df.head(30)
    colors = ["#E53935" if is_inmet else "#1976D2" for is_inmet in top30["is_inmet"]]
    y_pos = np.arange(len(top30))
    ax.barh(y_pos, top30["mean_abs_shap"].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importância SHAP média (|SHAP|)", fontsize=11)
    ax.set_title("Top-30 Features — Importância SHAP\n(Azul=SINAN, Vermelho=INMET)", fontsize=12)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#1976D2", label="SINAN"),
                       Patch(facecolor="#E53935", label="INMET")]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5yr_shap_bar_top30.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_5yr_shap_bar_top30.png")

    # SHAP dependence plots for top-3 climate features
    top3_climate = climate_feats.head(3)["feature"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat in enumerate(top3_climate):
        feat_idx = all_features.index(feat)
        ax = axes[i]
        ax.scatter(
            X_sample[feat].values, shap_values[:, feat_idx],
            alpha=0.1, s=2, color="#E53935",
        )
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP value" if i == 0 else "", fontsize=10)
        ax.set_title(f"{feat}\n(rank {int(climate_feats[climate_feats['feature'] == feat]['rank'].values[0])})", fontsize=11)
        ax.axhline(y=0, color="gray", linewidth=0.5)
    fig.suptitle("Dependência SHAP — Top-3 Variáveis Climáticas", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5yr_shap_dependence_climate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_5yr_shap_dependence_climate.png")

    # Report
    report = {
        "sample_size": int(len(X_sample)),
        "n_features": len(all_features),
        "n_inmet_features": int(len(climate_feats)),
        "n_sinan_features": int(len(sinan_feats)),
        "climate_in_top_20": int(n_climate_top20),
        "climate_in_top_50": int(n_climate_top50),
        "best_climate_feature": climate_feats.iloc[0]["feature"],
        "best_climate_rank": int(climate_feats.iloc[0]["rank"]),
        "best_climate_shap": round(float(climate_feats.iloc[0]["mean_abs_shap"]), 4),
        "mean_shap_inmet": round(float(climate_feats["mean_abs_shap"].mean()), 4),
        "mean_shap_sinan": round(float(sinan_feats["mean_abs_shap"].mean()), 4),
        "top_10": importance_df.head(10)[["rank", "feature", "mean_abs_shap", "is_inmet"]].to_dict("records"),
    }
    with open(DATA_DIR / "07_shap_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO SHAP:")
    print(f"  Features climaticas no top-20: {n_climate_top20}")
    print(f"  Melhor INMET: {climate_feats.iloc[0]['feature']} (rank {int(climate_feats.iloc[0]['rank'])})")
    print(f"  SHAP medio INMET: {climate_feats['mean_abs_shap'].mean():.4f}")
    print(f"  SHAP medio SINAN: {sinan_feats['mean_abs_shap'].mean():.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-116: SHAP v2 aprofundado com limiares climáticos.

Analisa o MELHOR modelo da Fase 2, extrai thresholds exatos para features
climáticas, e compara ranking SHAP entre Fase 1 e Fase 2.
"""

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
SHAP_DIR = BASE_DIR / "models" / "shap_v2"
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


def find_threshold(feature_vals, shap_vals):
    """Find the feature value where SHAP crosses zero (inflection point)."""
    sorted_idx = np.argsort(feature_vals)
    f_sorted = feature_vals[sorted_idx]
    s_sorted = shap_vals[sorted_idx]

    n_bins = 50
    bins = np.linspace(f_sorted.min(), f_sorted.max(), n_bins + 1)
    bin_means = []
    bin_shap_means = []
    for i in range(n_bins):
        mask = (f_sorted >= bins[i]) & (f_sorted < bins[i + 1])
        if mask.sum() > 10:
            bin_means.append((bins[i] + bins[i + 1]) / 2)
            bin_shap_means.append(s_sorted[mask].mean())

    if len(bin_means) < 3:
        return None, None

    bin_means = np.array(bin_means)
    bin_shap_means = np.array(bin_shap_means)

    for i in range(len(bin_shap_means) - 1):
        if bin_shap_means[i] * bin_shap_means[i + 1] < 0:
            frac = abs(bin_shap_means[i]) / (abs(bin_shap_means[i]) + abs(bin_shap_means[i + 1]))
            threshold = bin_means[i] + frac * (bin_means[i + 1] - bin_means[i])
            direction = "positivo" if bin_shap_means[i + 1] > bin_shap_means[i] else "negativo"
            return round(float(threshold), 2), direction

    return None, None


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-116: SHAP v2 — Limiares Climáticos")
    print("=" * 60)

    print("\nDeterminando melhor modelo da Fase 2...")
    reg_report = json.load(open(DATA_DIR / "13_regression_v2_report.json"))
    models_r2 = {k: v["R2_log"] for k, v in reg_report["modelos"].items()}
    best_key = max(models_r2, key=models_r2.get)
    print(f"  Melhor: {best_key} (R²_log = {models_r2[best_key]})")

    model_map = {
        "A_sinan_only": "model_a_sinan_only.json",
        "B_inmet_bruto": "model_b_inmet_bruto.json",
        "C_inmet_enriquecido": "model_c_inmet_enriquecido.json",
    }
    model_dir = BASE_DIR / "models" / "regression_v2"
    model = xgb.XGBRegressor()
    model.load_model(str(model_dir / model_map[best_key]))

    print("\nCarregando dados de teste...")
    test = pd.read_parquet(DATA_DIR / "test_v2.parquet")
    test["ibge_municipio"] = test["ibge_municipio"].astype(str)

    all_features = [c for c in test.columns
                    if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                    and test[c].dtype != "object" and not str(test[c].dtype).startswith("datetime")]

    if best_key == "A_sinan_only":
        features = [c for c in all_features if not is_inmet_feature(c)]
    elif best_key == "B_inmet_bruto":
        features = [c for c in all_features
                    if not is_inmet_feature(c)
                    or (is_inmet_feature(c) and not (("_lag_" in c and c.endswith("w")) or "_anomalia" in c or "_mm_2_4w" in c or "_mm_4_8w" in c))]
    else:
        features = all_features

    X_test = test[features]
    print(f"  Test: {len(X_test):,} linhas, {len(features)} features")

    np.random.seed(42)
    idx = np.random.choice(len(X_test), min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx]
    print(f"  Amostra SHAP: {len(X_sample):,}")

    print("\nCalculando SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print(f"  Shape: {shap_values.shape}")

    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": shap_importance,
        "is_inmet": [is_inmet_feature(f) for f in features],
        "is_anomaly": ["_anomalia" in f for f in features],
        "is_lag_bio": ["_lag_" in f and f.endswith("w") for f in features],
        "is_mm_bio": ["_mm_2_4w" in f or "_mm_4_8w" in f for f in features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(SHAP_DIR / "shap_feature_importance_v2.csv", index=False)

    climate_feats = importance_df[importance_df["is_inmet"]]
    sinan_feats = importance_df[~importance_df["is_inmet"]]

    n_climate_top10 = len(climate_feats[climate_feats["rank"] <= 10])
    n_climate_top20 = len(climate_feats[climate_feats["rank"] <= 20])
    n_climate_top30 = len(climate_feats[climate_feats["rank"] <= 30])

    print(f"\n  Features climáticas no top-10: {n_climate_top10}")
    print(f"  Features climáticas no top-20: {n_climate_top20}")
    print(f"  Features climáticas no top-30: {n_climate_top30}")

    anom_in_top30 = importance_df[(importance_df["rank"] <= 30) & importance_df["is_anomaly"]]
    lag_in_top30 = importance_df[(importance_df["rank"] <= 30) & importance_df["is_lag_bio"]]
    mm_in_top30 = importance_df[(importance_df["rank"] <= 30) & importance_df["is_mm_bio"]]
    print(f"  No top-30: {len(anom_in_top30)} anomalias, {len(lag_in_top30)} lags bio, {len(mm_in_top30)} mm bio")

    print("\nTop-30 features:")
    for _, row in importance_df.head(30).iterrows():
        tag = "[INMET]" if row["is_inmet"] else "[SINAN]"
        extra = ""
        if row["is_anomaly"]:
            extra = " (anomalia)"
        elif row["is_lag_bio"]:
            extra = " (lag bio)"
        elif row["is_mm_bio"]:
            extra = " (mm bio)"
        print(f"  {int(row['rank']):3d}. {tag:8s} {row['feature']:45s} SHAP={row['mean_abs_shap']:.4f}{extra}")

    print("\nExtraindo limiares climáticos...")
    climate_in_top30 = climate_feats[climate_feats["rank"] <= 30]
    thresholds = []
    for _, row in climate_in_top30.iterrows():
        feat = row["feature"]
        feat_idx = features.index(feat)
        fvals = X_sample[feat].values
        svals = shap_values[:, feat_idx]
        valid = ~np.isnan(fvals)
        if valid.sum() < 100:
            continue
        thresh, direction = find_threshold(fvals[valid], svals[valid])
        if thresh is not None:
            thresholds.append({
                "feature": feat,
                "rank": int(row["rank"]),
                "threshold": thresh,
                "direction": direction,
                "mean_shap_above": round(float(svals[valid][fvals[valid] > thresh].mean()), 4),
                "mean_shap_below": round(float(svals[valid][fvals[valid] <= thresh].mean()), 4),
            })
            print(f"  {feat}: limiar = {thresh}, direção = {direction}")

    print("\nGerando figuras SHAP...")

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, max_display=30, show=False)
    plt.title(f"SHAP Summary — Top-30 Features ({best_key})", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_5yr_v2_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  Salvo: fig_5yr_v2_shap_beeswarm.png")

    top5_climate = climate_feats.head(5)["feature"].tolist()
    n_plots = min(5, len(top5_climate))
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        for i, feat in enumerate(top5_climate[:n_plots]):
            feat_idx = features.index(feat)
            ax = axes[i]
            fvals = X_sample[feat].values
            svals = shap_values[:, feat_idx]
            valid = ~np.isnan(fvals)
            ax.scatter(fvals[valid], svals[valid], alpha=0.1, s=2, color="#E53935")
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.set_xlabel(feat.split("_lag_")[0] if "_lag_" in feat else feat, fontsize=9)
            ax.set_ylabel("SHAP" if i == 0 else "", fontsize=9)
            rank = int(climate_feats[climate_feats["feature"] == feat]["rank"].values[0])
            ax.set_title(f"{feat}\n(rank {rank})", fontsize=9)

            matching = [t for t in thresholds if t["feature"] == feat]
            if matching:
                t = matching[0]
                ax.axvline(x=t["threshold"], color="#4CAF50", linewidth=1.5, linestyle="--")
                ax.text(t["threshold"], ax.get_ylim()[1] * 0.9,
                        f" ≈{t['threshold']}", fontsize=8, color="#4CAF50")

        fig.suptitle("SHAP Dependência — Top-5 Variáveis Climáticas (com limiares)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_5yr_v2_shap_dependence_climate.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: fig_5yr_v2_shap_dependence_climate.png")

    if thresholds:
        fig, ax = plt.subplots(figsize=(12, max(4, len(thresholds) * 0.5)))
        feats_t = [t["feature"] for t in thresholds]
        vals_t = [t["threshold"] for t in thresholds]
        colors_t = ["#E53935" if t["direction"] == "positivo" else "#1976D2" for t in thresholds]
        y_pos = range(len(feats_t))
        ax.barh(y_pos, vals_t, color=colors_t)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats_t, fontsize=8)
        ax.set_xlabel("Valor Limiar", fontsize=10)
        ax.set_title("Limiares Climáticos Extraídos do SHAP\n(Vermelho = acima aumenta risco, Azul = acima diminui)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_5yr_v2_shap_limiares.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvo: fig_5yr_v2_shap_limiares.png")

    print("\nComparando SHAP Fase 1 vs Fase 2...")
    phase1_report = {}
    phase1_path = DATA_DIR / "07_shap_report.json"
    if phase1_path.exists():
        phase1_report = json.load(open(phase1_path))

    comparison = {
        "fase_1": {
            "climate_in_top_10": phase1_report.get("climate_in_top_20", "?"),
            "climate_in_top_20": phase1_report.get("climate_in_top_20", "?"),
            "best_climate_feature": phase1_report.get("best_climate_feature", "?"),
            "best_climate_rank": phase1_report.get("best_climate_rank", "?"),
        },
        "fase_2": {
            "climate_in_top_10": n_climate_top10,
            "climate_in_top_20": n_climate_top20,
            "climate_in_top_30": n_climate_top30,
            "best_climate_feature": climate_feats.iloc[0]["feature"] if len(climate_feats) > 0 else "N/A",
            "best_climate_rank": int(climate_feats.iloc[0]["rank"]) if len(climate_feats) > 0 else "N/A",
        },
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["Top-10", "Top-20", "Top-30"]
    p1 = [phase1_report.get("climate_in_top_20", 0),
          phase1_report.get("climate_in_top_20", 0),
          phase1_report.get("climate_in_top_50", 0)]
    p2 = [n_climate_top10, n_climate_top20, n_climate_top30]
    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width / 2, p1, width, label="Fase 1", color="#1976D2")
    ax.bar(x + width / 2, p2, width, label="Fase 2", color="#E53935")
    for i in range(len(cats)):
        ax.text(x[i] - width / 2, p1[i] + 0.1, str(p1[i]), ha="center", fontsize=10)
        ax.text(x[i] + width / 2, p2[i] + 0.1, str(p2[i]), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Qtd. Features Climáticas", fontsize=11)
    ax.set_title("Features Climáticas no Ranking SHAP — Fase 1 vs Fase 2", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5yr_v2_shap_comparacao_fases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_5yr_v2_shap_comparacao_fases.png")

    report = {
        "modelo_analisado": best_key,
        "sample_size": len(X_sample),
        "n_features": len(features),
        "climate_in_top_10": n_climate_top10,
        "climate_in_top_20": n_climate_top20,
        "climate_in_top_30": n_climate_top30,
        "best_climate_feature": climate_feats.iloc[0]["feature"] if len(climate_feats) > 0 else None,
        "best_climate_rank": int(climate_feats.iloc[0]["rank"]) if len(climate_feats) > 0 else None,
        "best_climate_shap": round(float(climate_feats.iloc[0]["mean_abs_shap"]), 4) if len(climate_feats) > 0 else None,
        "mean_shap_inmet": round(float(climate_feats["mean_abs_shap"].mean()), 4),
        "mean_shap_sinan": round(float(sinan_feats["mean_abs_shap"].mean()), 4),
        "thresholds": thresholds,
        "comparison_phases": comparison,
        "enriched_features_analysis": {
            "anomalias_in_top_30": len(anom_in_top30),
            "lags_bio_in_top_30": len(lag_in_top30),
            "mm_bio_in_top_30": len(mm_in_top30),
        },
        "top_10": importance_df.head(10)[["rank", "feature", "mean_abs_shap", "is_inmet"]].to_dict("records"),
    }
    with open(DATA_DIR / "16_shap_v2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-116:")
    print(f"  Modelo: {best_key}")
    print(f"  Clima no top-10: {n_climate_top10} | top-20: {n_climate_top20} | top-30: {n_climate_top30}")
    if len(climate_feats) > 0:
        print(f"  Melhor clima: {climate_feats.iloc[0]['feature']} (rank {int(climate_feats.iloc[0]['rank'])})")
    print(f"  Limiares encontrados: {len(thresholds)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

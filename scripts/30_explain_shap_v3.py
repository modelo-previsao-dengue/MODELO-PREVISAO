#!/usr/bin/env python3
"""US-210: SHAP v3 (mesorregional) — explicabilidade e limiares climáticos.

Analisa o modelo C (SINAN + INMET enriquecido) da regressão v3 com
shap.TreeExplainer sobre o conjunto de teste, gera importância top-30,
beeswarm top-20, dependence plots para 3 variáveis climáticas-chave e
extrai limiares climáticos por cruzamento de zero do SHAP suavizado.
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
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
SHAP_DIR = BASE_DIR / "models" / "shap_v3"
MODEL_PATH = BASE_DIR / "models" / "regression_v3" / "model_c_inmet_enriquecido.ubj"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v3.csv"

SAMPLE_SIZE = 50000

# Features de dependência solicitadas — se ausentes, usa o nome disponível
# mais próximo (mesmo prefixo de variável climática).
DEPENDENCE_FEATURES = [
    "temp_mean_c_lag_8w",
    "rain_sum_mm_lag_4w",
    "humidity_mean_pct_lag_4w",
]


def categorize_features(schema_path):
    """Read feature_schema_v3.csv and split features into SINAN, INMET bruto, INMET enriquecido."""
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return sinan, inmet_bruto, inmet_enriched


def find_closest_feature(wanted, available):
    """Return wanted if present; otherwise the closest match by variable prefix."""
    if wanted in available:
        return wanted
    prefix = wanted.split("_lag_")[0].split("_mm_")[0].split("_anomalia")[0]
    candidates = [f for f in available if f.startswith(prefix)]
    if candidates:
        # Prefer same suffix family (lag), else first candidate.
        return sorted(candidates, key=len)[0]
    return None


def find_threshold(feature_vals, shap_vals):
    """Find the feature value where SHAP crosses zero (inflection point), via binned smoothing."""
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


def plot_top30_importance(importance_df, out_path):
    """Horizontal barplot of top-30 features by mean |SHAP|, colored by category."""
    top30 = importance_df.head(30).iloc[::-1]
    color_map = {
        "sinan": "#1976D2",
        "inmet_bruto": "#FF9800",
        "inmet_lag_bio": "#E53935",
        "inmet_mm_bio": "#8E24AA",
        "inmet_anomalia": "#4CAF50",
    }
    colors = [color_map.get(c, "#9E9E9E") for c in top30["category"]]

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(range(len(top30)), top30["mean_abs_shap"], color=colors)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Top-30 Features por Importância SHAP — Modelo C v3", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-210: SHAP v3 — Explicabilidade e Limiares Climáticos")
    print("=" * 60)

    print("\nCarregando modelo C (SINAN + INMET enriquecido)...")
    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(SCHEMA_PATH)
    features = sinan_feats + inmet_bruto + inmet_enriched
    print(f"  Modelo: {MODEL_PATH.name} ({len(features)} features)")

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    print("\nCarregando dados de teste v3...")
    test = pd.read_parquet(DATA_DIR / "test_v3.parquet")
    X_test = test[features]
    print(f"  Test: {len(X_test):,} linhas, {len(features)} features")

    np.random.seed(42)
    idx = np.random.choice(len(X_test), min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx]
    print(f"  Amostra SHAP: {len(X_sample):,}")

    print("\nCalculando SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print(f"  Shape: {shap_values.shape}")

    schema = pd.read_csv(SCHEMA_PATH).set_index("feature")["category"].to_dict()
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": shap_importance,
        "category": [schema.get(f, "sinan") for f in features],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["is_inmet"] = importance_df["category"] != "sinan"
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(SHAP_DIR / "shap_feature_importance_v3.csv", index=False)

    climate_feats = importance_df[importance_df["is_inmet"]]
    sinan_feats_imp = importance_df[~importance_df["is_inmet"]]

    n_climate_top10 = len(climate_feats[climate_feats["rank"] <= 10])
    n_climate_top20 = len(climate_feats[climate_feats["rank"] <= 20])
    n_climate_top30 = len(climate_feats[climate_feats["rank"] <= 30])

    print(f"\n  Features climáticas no top-10: {n_climate_top10}")
    print(f"  Features climáticas no top-20: {n_climate_top20}")
    print(f"  Features climáticas no top-30: {n_climate_top30}")

    print("\nTop-30 features:")
    for _, row in importance_df.head(30).iterrows():
        tag = "[SINAN]" if row["category"] == "sinan" else f"[{row['category'].upper()}]"
        print(f"  {int(row['rank']):3d}. {tag:20s} {row['feature']:45s} SHAP={row['mean_abs_shap']:.4f}")

    print("\nExtraindo limiares climáticos (top-30 climáticas)...")
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

    plot_top30_importance(importance_df, FIG_DIR / "fig_v3_shap_top30_importance.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Top-20 Features (Modelo C v3)", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_v3_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close("all")
    print("  Salvo: fig_v3_shap_beeswarm.png")

    dep_resolved = []
    for wanted in DEPENDENCE_FEATURES:
        resolved = find_closest_feature(wanted, features)
        dep_resolved.append((wanted, resolved))
        if resolved is None:
            print(f"  AVISO: nenhuma feature disponível para {wanted}")
        elif resolved != wanted:
            print(f"  AVISO: {wanted} ausente — usando feature mais próxima: {resolved}")

    dep_valid = [(w, r) for w, r in dep_resolved if r is not None]
    if dep_valid:
        fig, axes = plt.subplots(1, len(dep_valid), figsize=(5 * len(dep_valid), 5))
        if len(dep_valid) == 1:
            axes = [axes]
        for i, (wanted, feat) in enumerate(dep_valid):
            feat_idx = features.index(feat)
            ax = axes[i]
            fvals = X_sample[feat].values
            svals = shap_values[:, feat_idx]
            valid = ~np.isnan(fvals)
            ax.scatter(fvals[valid], svals[valid], alpha=0.1, s=2, color="#E53935")
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel("SHAP" if i == 0 else "", fontsize=9)
            rank = int(importance_df[importance_df["feature"] == feat]["rank"].values[0])
            ax.set_title(f"{feat}\n(rank {rank})", fontsize=9)

            matching = [t for t in thresholds if t["feature"] == feat]
            if matching:
                t = matching[0]
                ax.axvline(x=t["threshold"], color="#4CAF50", linewidth=1.5, linestyle="--")
                ax.text(t["threshold"], ax.get_ylim()[1] * 0.9,
                        f" ≈{t['threshold']}", fontsize=8, color="#4CAF50")

        fig.suptitle("SHAP Dependência — Variáveis Climáticas Solicitadas (v3)", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_v3_shap_dependence_climate.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Salvo: fig_v3_shap_dependence_climate.png")

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
        ax.set_title("Limiares Climáticos Extraídos do SHAP (v3)\n(Vermelho = acima aumenta risco, Azul = acima diminui)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_v3_shap_limiares.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Salvo: fig_v3_shap_limiares.png")

    report = {
        "modelo_analisado": "C_inmet_enriquecido",
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "sample_size": len(X_sample),
        "n_features": len(features),
        "climate_in_top_10": n_climate_top10,
        "climate_in_top_20": n_climate_top20,
        "climate_in_top_30": n_climate_top30,
        "best_climate_feature": climate_feats.iloc[0]["feature"] if len(climate_feats) > 0 else None,
        "best_climate_rank": int(climate_feats.iloc[0]["rank"]) if len(climate_feats) > 0 else None,
        "best_climate_shap": round(float(climate_feats.iloc[0]["mean_abs_shap"]), 4) if len(climate_feats) > 0 else None,
        "mean_shap_inmet": round(float(climate_feats["mean_abs_shap"].mean()), 4) if len(climate_feats) > 0 else None,
        "mean_shap_sinan": round(float(sinan_feats_imp["mean_abs_shap"].mean()), 4) if len(sinan_feats_imp) > 0 else None,
        "dependence_features_requested": {w: r for w, r in dep_resolved},
        "thresholds": thresholds,
        "top_10": importance_df.head(10)[["rank", "feature", "mean_abs_shap", "category"]].to_dict("records"),
        "top_30": importance_df.head(30)[["rank", "feature", "mean_abs_shap", "category"]].to_dict("records"),
    }
    with open(DATA_DIR / "30_shap_v3_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-210:")
    print(f"  Modelo: C_inmet_enriquecido")
    print(f"  Clima no top-10: {n_climate_top10} | top-20: {n_climate_top20} | top-30: {n_climate_top30}")
    if len(climate_feats) > 0:
        print(f"  Melhor clima: {climate_feats.iloc[0]['feature']} (rank {int(climate_feats.iloc[0]['rank'])})")
    print(f"  Limiares encontrados: {len(thresholds)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-110: Feature Engineering — Lags biológicos do ciclo do Aedes (1-8 semanas).

Cria 96 features de lag (12 features × 8 lags) + 24 médias móveis biológicas
(12 features × 2 janelas: mm_2_4w e mm_4_8w).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

CLIMATE_BASE = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]

LAGS = list(range(1, 9))  # 1-8 weeks


def compute_lags(df):
    """Compute lag features for all 12 climate variables, grouped by municipality."""
    print("\nCalculando lags 1-8 semanas para 12 variáveis climáticas...")
    grouped = df.groupby("ibge_municipio")
    new_cols = {}
    for feat in CLIMATE_BASE:
        if feat not in df.columns:
            print(f"  AVISO: {feat} não encontrada, pulando")
            continue
        for lag in LAGS:
            col_name = f"{feat}_lag_{lag}w"
            new_cols[col_name] = grouped[feat].shift(lag)

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de lag")
    return pd.concat([df, new_df], axis=1)


def compute_bio_moving_averages(df):
    """Compute biologically motivated moving averages.

    mm_2_4w: mean of lags 2,3,4 — hatching window
    mm_4_8w: mean of lags 4,5,6,7,8 — full Aedes lifecycle window
    """
    print("\nCalculando médias móveis biológicas...")
    new_cols = {}
    for feat in CLIMATE_BASE:
        lag_2_4 = [f"{feat}_lag_{i}w" for i in [2, 3, 4]]
        lag_4_8 = [f"{feat}_lag_{i}w" for i in [4, 5, 6, 7, 8]]

        available_2_4 = [c for c in lag_2_4 if c in df.columns]
        available_4_8 = [c for c in lag_4_8 if c in df.columns]

        if available_2_4:
            new_cols[f"{feat}_mm_2_4w"] = df[available_2_4].mean(axis=1)
        if available_4_8:
            new_cols[f"{feat}_mm_4_8w"] = df[available_4_8].mean(axis=1)

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de média móvel biológica")
    return pd.concat([df, new_df], axis=1)


def compute_correlations(df):
    """Compute Spearman correlations of all new features with notificacoes."""
    print("\nCalculando correlações Spearman com notificações...")
    target = df["notificacoes"].dropna()

    results = []
    for feat in CLIMATE_BASE:
        for lag in LAGS:
            col = f"{feat}_lag_{lag}w"
            if col not in df.columns:
                continue
            valid = df[[col, "notificacoes"]].dropna()
            if len(valid) < 100:
                continue
            rho, pval = spearmanr(valid[col], valid["notificacoes"])
            results.append({
                "feature_base": feat,
                "lag": lag,
                "col_name": col,
                "type": "lag",
                "spearman_r": round(float(rho), 6),
                "p_value": float(pval),
                "n_valid": len(valid),
            })

        for suffix, label in [("mm_2_4w", "mm_2_4w"), ("mm_4_8w", "mm_4_8w")]:
            col = f"{feat}_{suffix}"
            if col not in df.columns:
                continue
            valid = df[[col, "notificacoes"]].dropna()
            if len(valid) < 100:
                continue
            rho, pval = spearmanr(valid[col], valid["notificacoes"])
            results.append({
                "feature_base": feat,
                "lag": label,
                "col_name": col,
                "type": "moving_avg",
                "spearman_r": round(float(rho), 6),
                "p_value": float(pval),
                "n_valid": len(valid),
            })

    corr_df = pd.DataFrame(results)
    print(f"  {len(corr_df)} correlações calculadas")
    return corr_df


def find_optimal_lags(corr_df):
    """Find the lag with highest |r| for each base feature."""
    lag_only = corr_df[corr_df["type"] == "lag"].copy()
    lag_only["abs_r"] = lag_only["spearman_r"].abs()
    best = lag_only.loc[lag_only.groupby("feature_base")["abs_r"].idxmax()]
    return best.sort_values("abs_r", ascending=False)


def plot_heatmap(corr_df, out_path):
    """Heatmap: 12 features × 8 lags with Spearman correlation."""
    lag_only = corr_df[corr_df["type"] == "lag"].copy()
    pivot = lag_only.pivot(index="feature_base", columns="lag", values="spearman_r")
    pivot = pivot.reindex(index=CLIMATE_BASE, columns=LAGS)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(len(LAGS)))
    ax.set_xticklabels([f"{l}w" for l in LAGS])
    ax.set_yticks(range(len(CLIMATE_BASE)))
    ax.set_yticklabels(CLIMATE_BASE, fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.12 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlação Spearman (ρ)", fontsize=10)
    ax.set_xlabel("Defasagem (semanas)", fontsize=11)
    ax.set_ylabel("Variável Climática", fontsize=11)
    ax.set_title("Correlação Spearman × Lag Biológico — Variáveis Climáticas vs Notificações",
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_optimal_lag(best_lags, out_path):
    """Barplot: optimal lag per climate variable."""
    best = best_lags.sort_values("abs_r", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ["#E53935" if r > 0 else "#1976D2" for r in best["spearman_r"]]
    ax.barh(range(len(best)), best["spearman_r"].values, color=colors)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best["feature_base"].values, fontsize=9)
    ax.set_xlabel("Correlação Spearman (ρ)", fontsize=10)
    ax.set_title("Correlação no Lag Ótimo", fontsize=11)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    ax.barh(range(len(best)), best["lag"].values, color="#4CAF50")
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best["feature_base"].values, fontsize=9)
    ax.set_xlabel("Lag Ótimo (semanas)", fontsize=10)
    ax.set_title("Lag com Maior |ρ|", fontsize=11)
    ax.set_xticks(LAGS)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Lag Biológico Ótimo por Variável Climática", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-110: Feature Engineering — Lags Biológicos")
    print("=" * 60)

    print("\nCarregando dataset filtrado...")
    df = pd.read_parquet(DATA_DIR / "integrated_filtered.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    print(f"  {len(df):,} linhas, {df['ibge_municipio'].nunique():,} municípios")
    print(f"  Colunas iniciais: {len(df.columns)}")

    n_before = len(df.columns)
    df = compute_lags(df)
    df = compute_bio_moving_averages(df)
    n_after = len(df.columns)
    n_new = n_after - n_before
    print(f"\n  Colunas: {n_before} → {n_after} (+{n_new} novas)")

    corr_df = compute_correlations(df)
    corr_df.to_csv(DATA_DIR / "eda_lags_biologicos.csv", index=False)

    best_lags = find_optimal_lags(corr_df)
    print("\nLag ótimo por variável climática:")
    for _, row in best_lags.iterrows():
        print(f"  {row['feature_base']:25s}  lag={int(row['lag'])}w  ρ={row['spearman_r']:+.4f}")

    print("\nGerando figuras...")
    plot_heatmap(corr_df, FIG_DIR / "fig_5yr_v2_lag_heatmap.png")
    plot_optimal_lag(best_lags, FIG_DIR / "fig_5yr_v2_lag_otimo.png")

    print("\nSalvando dataset enriquecido...")
    df.to_parquet(DATA_DIR / "integrated_filtered_v2.parquet", index=False)
    print(f"  Salvo: integrated_filtered_v2.parquet ({len(df):,} linhas, {len(df.columns)} colunas)")

    lag_corrs = corr_df[corr_df["type"] == "lag"]
    mm_corrs = corr_df[corr_df["type"] == "moving_avg"]

    report = {
        "n_base_features": len(CLIMATE_BASE),
        "n_lags": len(LAGS),
        "n_new_lag_features": len(lag_corrs),
        "n_new_mm_features": len(mm_corrs),
        "n_total_new": n_new,
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
        "lag_otimo": {
            row["feature_base"]: {
                "lag": int(row["lag"]),
                "spearman_r": row["spearman_r"],
            }
            for _, row in best_lags.iterrows()
        },
        "correlacoes_resumo": {
            "media_abs_r_lags": round(float(lag_corrs["spearman_r"].abs().mean()), 4),
            "max_abs_r_lags": round(float(lag_corrs["spearman_r"].abs().max()), 4),
            "media_abs_r_mm": round(float(mm_corrs["spearman_r"].abs().mean()), 4),
            "features_r_above_010": int((lag_corrs["spearman_r"].abs() > 0.10).sum()),
            "features_r_above_015": int((lag_corrs["spearman_r"].abs() > 0.15).sum()),
        },
    }
    with open(DATA_DIR / "10_lags_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-110:")
    print(f"  Novas features de lag: {len(lag_corrs)}")
    print(f"  Novas features de média móvel: {len(mm_corrs)}")
    print(f"  Total novas: {n_new}")
    print(f"  Features com |ρ| > 0.10: {report['correlacoes_resumo']['features_r_above_010']}")
    print(f"  Features com |ρ| > 0.15: {report['correlacoes_resumo']['features_r_above_015']}")
    print(f"  Dataset salvo: {len(df):,} × {len(df.columns)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

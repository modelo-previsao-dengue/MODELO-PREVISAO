#!/usr/bin/env python3
"""US-111: Feature Engineering — Anomalias climáticas (desvio da norma histórica).

Para cada variável climática base, calcula:
  - anomalia = valor_observado - média_histórica(municipio, semana_do_ano)
  - anomalia_std = anomalia / desvio_padrão (z-score)
Total: 12 × 2 = 24 novas features.
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


def compute_anomalies(df):
    """Compute anomalies: deviation from historical mean per (municipality, epi_week)."""
    print("\nCalculando anomalias climáticas...")
    group_keys = ["ibge_municipio", "semana_epidemiologica"]

    new_cols = {}
    for feat in CLIMATE_BASE:
        if feat not in df.columns:
            print(f"  AVISO: {feat} não encontrada, pulando")
            continue

        stats = df.groupby(group_keys)[feat].agg(["mean", "std"])
        stats.columns = ["hist_mean", "hist_std"]
        stats["hist_std"] = stats["hist_std"].replace(0, np.nan)

        merged = df[group_keys].merge(stats, left_on=group_keys, right_index=True, how="left")

        anomalia = df[feat] - merged["hist_mean"]
        anomalia_std = anomalia / merged["hist_std"]

        new_cols[f"{feat}_anomalia"] = anomalia
        new_cols[f"{feat}_anomalia_std"] = anomalia_std

    new_df = pd.DataFrame(new_cols, index=df.index)
    print(f"  Criadas {len(new_cols)} features de anomalia")
    return pd.concat([df, new_df], axis=1)


def compute_correlations(df):
    """Compute Spearman correlations comparing raw vs anomaly features."""
    print("\nCalculando correlações Spearman...")
    results = []
    for feat in CLIMATE_BASE:
        for suffix, label in [("", "bruto"), ("_anomalia", "anomalia"), ("_anomalia_std", "z-score")]:
            col = f"{feat}{suffix}" if suffix else feat
            if col not in df.columns:
                continue
            valid = df[[col, "notificacoes"]].dropna()
            if len(valid) < 100:
                continue
            rho, pval = spearmanr(valid[col], valid["notificacoes"])
            results.append({
                "feature_base": feat,
                "type": label,
                "col_name": col,
                "spearman_r": round(float(rho), 6),
                "abs_r": round(abs(float(rho)), 6),
                "p_value": float(pval),
                "n_valid": len(valid),
            })

    return pd.DataFrame(results)


def compute_surto_correlations(df):
    """Compute correlations with risco_surto (not just notificacoes)."""
    print("\nCalculando correlações com risco de surto...")
    df_tmp = df.copy()
    df_tmp = df_tmp.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df_tmp["notificacoes_t4"] = df_tmp.groupby("ibge_municipio")["notificacoes"].shift(-4)
    df_tmp = df_tmp.dropna(subset=["notificacoes_t4"])

    results = []
    for feat in CLIMATE_BASE:
        for suffix, label in [("_anomalia", "anomalia"), ("_anomalia_std", "z-score")]:
            col = f"{feat}{suffix}"
            if col not in df_tmp.columns:
                continue
            valid = df_tmp[[col, "notificacoes_t4"]].dropna()
            if len(valid) < 100:
                continue
            rho, pval = spearmanr(valid[col], valid["notificacoes_t4"])
            results.append({
                "feature_base": feat,
                "type": label,
                "spearman_r_t4": round(float(rho), 6),
                "p_value": float(pval),
            })

    return pd.DataFrame(results)


def plot_anomaly_vs_raw(corr_df, out_path):
    """Barplot comparing correlation of raw vs anomaly features."""
    raw = corr_df[corr_df["type"] == "bruto"].set_index("feature_base")["abs_r"]
    anom = corr_df[corr_df["type"] == "anomalia"].set_index("feature_base")["abs_r"]
    zscore = corr_df[corr_df["type"] == "z-score"].set_index("feature_base")["abs_r"]

    feats = CLIMATE_BASE
    x = np.arange(len(feats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, [raw.get(f, 0) for f in feats], width, label="Bruto", color="#1976D2")
    ax.bar(x, [anom.get(f, 0) for f in feats], width, label="Anomalia", color="#E53935")
    ax.bar(x + width, [zscore.get(f, 0) for f in feats], width, label="Z-score", color="#4CAF50")

    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|ρ Spearman|", fontsize=11)
    ax.set_title("Correlação com Notificações: Valor Bruto vs Anomalia vs Z-score", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(y=0.10, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_anomaly_heatmap_regiao(df, out_path):
    """Heatmap: anomaly correlations by region."""
    regioes = sorted(df["regiao"].dropna().unique())
    results = []
    for regiao in regioes:
        sub = df[df["regiao"] == regiao]
        for feat in CLIMATE_BASE:
            col = f"{feat}_anomalia"
            if col not in sub.columns:
                continue
            valid = sub[[col, "notificacoes"]].dropna()
            if len(valid) < 100:
                continue
            rho, _ = spearmanr(valid[col], valid["notificacoes"])
            results.append({"regiao": regiao, "feature": feat, "spearman_r": rho})

    if not results:
        print("  Sem dados para heatmap por região")
        return

    pivot = pd.DataFrame(results).pivot(index="feature", columns="regiao", values="spearman_r")
    pivot = pivot.reindex(index=CLIMATE_BASE)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.25, vmax=0.25)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.15 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ρ Spearman", fontsize=10)
    ax.set_title("Correlação Anomalias Climáticas × Notificações por Região", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-111: Feature Engineering — Anomalias Climáticas")
    print("=" * 60)

    print("\nCarregando dataset enriquecido (US-110)...")
    df = pd.read_parquet(DATA_DIR / "integrated_filtered_v2.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas")

    n_before = len(df.columns)
    df = compute_anomalies(df)
    n_after = len(df.columns)
    n_new = n_after - n_before
    print(f"\n  Colunas: {n_before} → {n_after} (+{n_new} novas)")

    corr_df = compute_correlations(df)
    surto_df = compute_surto_correlations(df)

    raw_corrs = corr_df[corr_df["type"] == "bruto"]
    anom_corrs = corr_df[corr_df["type"] == "anomalia"]
    z_corrs = corr_df[corr_df["type"] == "z-score"]

    print("\nComparação: Bruto vs Anomalia vs Z-score:")
    for feat in CLIMATE_BASE:
        r_raw = raw_corrs[raw_corrs["feature_base"] == feat]["abs_r"].values
        r_anom = anom_corrs[anom_corrs["feature_base"] == feat]["abs_r"].values
        r_z = z_corrs[z_corrs["feature_base"] == feat]["abs_r"].values
        r_raw = r_raw[0] if len(r_raw) else 0
        r_anom = r_anom[0] if len(r_anom) else 0
        r_z = r_z[0] if len(r_z) else 0
        winner = "ANOMALIA" if r_anom > r_raw else "BRUTO"
        print(f"  {feat:25s}  bruto={r_raw:.4f}  anom={r_anom:.4f}  z={r_z:.4f}  → {winner}")

    print("\nGerando figuras...")
    plot_anomaly_vs_raw(corr_df, FIG_DIR / "fig_5yr_v2_anomalia_vs_bruto.png")
    plot_anomaly_heatmap_regiao(df, FIG_DIR / "fig_5yr_v2_anomalia_heatmap.png")

    print("\nSalvando dataset atualizado...")
    df.to_parquet(DATA_DIR / "integrated_filtered_v2.parquet", index=False)
    print(f"  Salvo: integrated_filtered_v2.parquet ({len(df):,} linhas, {len(df.columns)} colunas)")

    anom_better = sum(
        1 for feat in CLIMATE_BASE
        if (anom_corrs[anom_corrs["feature_base"] == feat]["abs_r"].values[0]
            if len(anom_corrs[anom_corrs["feature_base"] == feat]) else 0) >
           (raw_corrs[raw_corrs["feature_base"] == feat]["abs_r"].values[0]
            if len(raw_corrs[raw_corrs["feature_base"] == feat]) else 0)
    )

    report = {
        "n_new_features": n_new,
        "dataset_rows": len(df),
        "dataset_cols": len(df.columns),
        "correlacoes_bruto_vs_anomalia": corr_df.to_dict("records"),
        "correlacoes_surto_t4": surto_df.to_dict("records") if len(surto_df) else [],
        "anomalia_melhor_que_bruto": anom_better,
        "total_features_comparadas": len(CLIMATE_BASE),
        "resumo": f"Anomalia melhor que bruto em {anom_better}/{len(CLIMATE_BASE)} features",
    }
    with open(DATA_DIR / "11_anomalias_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-111:")
    print(f"  Novas features: {n_new}")
    print(f"  Anomalia > bruto em: {anom_better}/{len(CLIMATE_BASE)} features")
    print(f"  Dataset final: {len(df):,} × {len(df.columns)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-102: Analise de correlacao exploratoria — Clima vs Dengue."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "model_ready_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

CLIMATE_BASE = [
    "rain_sum_mm", "rain_mean_mm", "rain_days", "rain_heavy_days",
    "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj",
]
LAGS = [0, 1, 2, 4, 8]
TARGET = "notificacoes"

UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}

REGIAO_MAP = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte",
    "PA": "Norte", "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste",
    "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste",
    "DF": "Centro-Oeste",
}


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_correlations(df, method="pearson"):
    """Compute correlation matrix: features × lags."""
    rows = []
    df_sorted = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    for feat in CLIMATE_BASE:
        if feat not in df_sorted.columns:
            continue
        for lag in LAGS:
            if lag == 0:
                climate_col = df_sorted[feat]
            else:
                climate_col = df_sorted.groupby("ibge_municipio")[feat].shift(lag)
            mask = climate_col.notna() & df_sorted[TARGET].notna()
            x = climate_col[mask].values
            y = df_sorted[TARGET][mask].values
            if len(x) < 100:
                continue
            if method == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)
            rows.append({
                "feature": feat, "lag": lag, "r": round(r, 6),
                "p_value": float(p), "n": len(x), "sig": sig_stars(p),
            })
    return pd.DataFrame(rows)


def plot_heatmap(corr_df, method_name, out_path):
    """Plot heatmap of correlation matrix."""
    pivot_r = corr_df.pivot(index="feature", columns="lag", values="r")
    pivot_sig = corr_df.pivot(index="feature", columns="lag", values="sig")
    annot = pivot_r.round(3).astype(str) + "\n" + pivot_sig.fillna("")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot_r, annot=annot, fmt="", cmap="RdBu_r", center=0,
        vmin=-0.15, vmax=0.15, linewidths=0.5, ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(f"Correlação {method_name} — Variáveis Climáticas × Dengue\n(por lag semanal, dataset filtrado 2019–2026)", fontsize=12)
    ax.set_xlabel("Lag (semanas)", fontsize=11)
    ax.set_ylabel("Variável Climática", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_scatter_top3(df, top3, out_path):
    """Scatter plots for top-3 correlations."""
    df_sorted = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (feat, lag, r_val) in enumerate(top3):
        if lag == 0:
            x = df_sorted[feat]
        else:
            x = df_sorted.groupby("ibge_municipio")[feat].shift(lag)
        y = df_sorted[TARGET]
        mask = x.notna() & y.notna()
        xs, ys = x[mask].values, y[mask].values
        sample_idx = np.random.RandomState(42).choice(len(xs), min(5000, len(xs)), replace=False)
        ax = axes[i]
        ax.scatter(xs[sample_idx], ys[sample_idx], alpha=0.15, s=3, color="#1976D2")
        z = np.polyfit(xs[sample_idx], ys[sample_idx], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_range, p_line(x_range), color="#E53935", linewidth=2)
        ax.set_xlabel(f"{feat} (lag {lag}s)", fontsize=10)
        ax.set_ylabel("Notificações" if i == 0 else "", fontsize=10)
        ax.set_title(f"{feat} (lag={lag}s)\nr={r_val:.4f}", fontsize=11)
    fig.suptitle("Top-3 Correlações Clima–Dengue (Spearman)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_region_bars(region_df, out_path):
    """Barplot of correlation by region for top features."""
    top_feats = region_df.groupby("feature")["abs_r"].mean().nlargest(5).index.tolist()
    sub = region_df[region_df["feature"].isin(top_feats)]

    fig, ax = plt.subplots(figsize=(12, 6))
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    width = 0.15
    x = np.arange(len(regioes))
    for i, feat in enumerate(top_feats):
        vals = []
        for reg in regioes:
            row = sub[(sub["feature"] == feat) & (sub["regiao"] == reg)]
            vals.append(row["r"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=feat)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(regioes, fontsize=10)
    ax.set_ylabel("Correlação (Spearman r)", fontsize=11)
    ax.set_title("Correlação Clima–Dengue por Região Brasileira (top-5 variáveis)", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-102: EDA Correlacao Clima vs Dengue")
    print("=" * 60)

    print("\nCarregando dataset filtrado...")
    df = pd.read_parquet(OUTPUT_DIR / "integrated_filtered.parquet")
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  {len(df):,} linhas, {df['ibge_municipio'].nunique()} municipios")

    # Pearson
    print("\nCalculando Pearson (features x lags)...")
    pearson_df = compute_correlations(df, "pearson")
    pearson_df.to_csv(OUTPUT_DIR / "eda_pearson_matrix.csv", index=False)
    print(f"  {len(pearson_df)} combinacoes calculadas")

    # Spearman
    print("Calculando Spearman (features x lags)...")
    spearman_df = compute_correlations(df, "spearman")
    spearman_df.to_csv(OUTPUT_DIR / "eda_spearman_matrix.csv", index=False)
    print(f"  {len(spearman_df)} combinacoes calculadas")

    # Optimal lag per feature
    print("\nLag otimo por feature (Spearman):")
    lag_otimo = []
    for feat in CLIMATE_BASE:
        sub = spearman_df[spearman_df["feature"] == feat]
        if sub.empty:
            continue
        best = sub.loc[sub["r"].abs().idxmax()]
        lag_otimo.append({
            "feature": feat, "best_lag": int(best["lag"]),
            "r": best["r"], "p_value": best["p_value"], "method": "spearman",
        })
        star = sig_stars(best["p_value"])
        print(f"  {feat:30s} lag={int(best['lag'])} r={best['r']:+.4f} {star}")

    lag_otimo_df = pd.DataFrame(lag_otimo)
    lag_otimo_df.to_csv(OUTPUT_DIR / "eda_lag_otimo.csv", index=False)

    # Correlation by region
    print("\nCorrelacao por regiao (Spearman, lag otimo):")
    df["uf"] = df["ibge_municipio"].str[:2].map(UF_MAP)
    df["regiao"] = df["uf"].map(REGIAO_MAP)

    region_rows = []
    df_sorted = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    for feat_info in lag_otimo:
        feat = feat_info["feature"]
        lag = feat_info["best_lag"]
        if lag == 0:
            climate_col = df_sorted[feat]
        else:
            climate_col = df_sorted.groupby("ibge_municipio")[feat].shift(lag)
        for regiao in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
            mask_reg = df_sorted["regiao"] == regiao
            mask_valid = climate_col.notna() & df_sorted[TARGET].notna() & mask_reg
            x = climate_col[mask_valid].values
            y = df_sorted[TARGET][mask_valid].values
            if len(x) < 50:
                continue
            rho, p = stats.spearmanr(x, y)
            region_rows.append({
                "feature": feat, "regiao": regiao, "lag": lag,
                "r": round(rho, 6), "p_value": float(p), "abs_r": abs(rho), "n": len(x),
            })

    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(OUTPUT_DIR / "eda_por_regiao.csv", index=False)

    print("  Top correlacoes por regiao:")
    for reg in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        sub = region_df[region_df["regiao"] == reg].nlargest(2, "abs_r")
        for _, row in sub.iterrows():
            print(f"    {reg:15s} {row['feature']:25s} r={row['r']:+.4f}")

    # Heatmaps
    print("\nGerando figuras...")
    plot_heatmap(pearson_df, "Pearson", FIG_DIR / "fig_5yr_eda_heatmap_pearson.png")
    plot_heatmap(spearman_df, "Spearman", FIG_DIR / "fig_5yr_eda_heatmap_spearman.png")

    # Scatter top-3
    top3_spearman = spearman_df.loc[spearman_df["r"].abs().nlargest(3).index]
    top3_list = [(r["feature"], int(r["lag"]), r["r"]) for _, r in top3_spearman.iterrows()]
    plot_scatter_top3(df, top3_list, FIG_DIR / "fig_5yr_eda_scatter_top3.png")

    # Region barplot
    plot_region_bars(region_df, FIG_DIR / "fig_5yr_eda_correlacao_regiao.png")

    # GO/NO-GO assessment
    sig_features = spearman_df[(spearman_df["r"].abs() > 0.10) & (spearman_df["p_value"] < 0.05)]
    n_sig = sig_features["feature"].nunique()
    if n_sig >= 3:
        decision = "GO"
    elif n_sig >= 1:
        decision = "INVESTIGATE"
    else:
        decision = "NO-GO"

    print(f"\n{'=' * 60}")
    print(f"  GO/NO-GO: {decision}")
    print(f"  Features com |r|>0.10 e p<0.05: {n_sig}")
    sig_list = sig_features.groupby("feature")["r"].apply(lambda x: x.abs().max()).nlargest(10)
    for feat, val in sig_list.items():
        print(f"    {feat}: |r|={val:.4f}")
    print(f"{'=' * 60}")

    report = {
        "decision": decision,
        "n_features_significant": int(n_sig),
        "significant_features": sig_list.to_dict(),
        "pearson_computed": len(pearson_df),
        "spearman_computed": len(spearman_df),
        "lag_otimo": {r["feature"]: {"lag": r["best_lag"], "r": r["r"]} for r in lag_otimo},
    }
    with open(OUTPUT_DIR / "02_eda_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Salvo: {OUTPUT_DIR / '02_eda_report.json'}")


if __name__ == "__main__":
    main()

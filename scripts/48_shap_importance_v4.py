#!/usr/bin/env python3
"""v4-48: SHAP feature importance para variantes A e B (INMET enriquecido).

RF-13: Importância de features (gain + SHAP) para variantes A e B,
comparadas lado a lado. 10.000 amostras do teste para SHAP.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"

SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"

N_SHAP_SAMPLES = 10_000
TOP_N = 30


def get_enriquecido_features(schema_path):
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return sinan + inmet_bruto + inmet_enriched


def plot_shap_comparison(shap_a, shap_b, features, out_path):
    mean_abs_a = np.abs(shap_a).mean(axis=0)
    mean_abs_b = np.abs(shap_b).mean(axis=0)

    imp_a = pd.Series(mean_abs_a, index=features).sort_values(ascending=False)
    imp_b = pd.Series(mean_abs_b, index=features).sort_values(ascending=False)

    top_union = list(dict.fromkeys(list(imp_a.head(TOP_N).index) + list(imp_b.head(TOP_N).index)))[:TOP_N]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, imp, title, color in [
        (axes[0], imp_a, "Variante A (Baseline 4yr)", "#1976D2"),
        (axes[1], imp_b, "Variante B (Série completa 22yr)", "#FF9800"),
    ]:
        top = imp.head(TOP_N)
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top.values, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"SHAP Top-{TOP_N} Features — A vs B (INMET enriquecido)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_gain_comparison(model_a, model_b, features, out_path):
    gain_a = model_a.get_booster().get_score(importance_type="gain")
    gain_b = model_b.get_booster().get_score(importance_type="gain")

    ga = pd.Series({f: gain_a.get(f"f{i}", 0) for i, f in enumerate(features)
                     if f"f{i}" in gain_a}, dtype=float)
    gb = pd.Series({f: gain_b.get(f"f{i}", 0) for i, f in enumerate(features)
                     if f"f{i}" in gain_b}, dtype=float)

    if ga.empty and gb.empty:
        ga = pd.Series({f: gain_a.get(f, 0) for f in features}, dtype=float)
        gb = pd.Series({f: gain_b.get(f, 0) for f in features}, dtype=float)

    ga = ga.sort_values(ascending=False).head(TOP_N)
    gb = gb.sort_values(ascending=False).head(TOP_N)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, imp, title, color in [
        (axes[0], ga, "Var A — Gain", "#1976D2"),
        (axes[1], gb, "Var B — Gain", "#FF9800"),
    ]:
        y_pos = np.arange(len(imp))
        ax.barh(y_pos, imp.values, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(imp.index, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Gain")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"XGBoost Gain Top-{TOP_N} — A vs B", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  v4-48: SHAP Feature Importance — Variantes A e B")
    print("=" * 70)

    features = get_enriquecido_features(SCHEMA_PATH)
    print(f"  Features (enriquecido): {len(features)}")

    print("\nCarregando modelos...")
    model_a = xgb.XGBRegressor()
    model_a.load_model(str(MODEL_DIR / "model_A_inmet_enriquecido.ubj"))
    model_b = xgb.XGBRegressor()
    model_b.load_model(str(MODEL_DIR / "model_B_inmet_enriquecido.ubj"))

    print("\nCarregando test set...")
    test = pd.read_parquet(DATA_DIR / "splits_a" / "test.parquet")
    print(f"  Test: {len(test):,} linhas")

    rng = np.random.RandomState(42)
    if len(test) > N_SHAP_SAMPLES:
        idx = rng.choice(len(test), size=N_SHAP_SAMPLES, replace=False)
        test_sample = test.iloc[idx]
    else:
        test_sample = test
    print(f"  Amostra SHAP: {len(test_sample):,} linhas")

    X_sample = test_sample[features]

    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
        print("  AVISO: shap não instalado, usando somente gain")

    report = {"n_shap_samples": len(test_sample), "n_features": len(features)}

    if has_shap:
        print("\nComputando SHAP para Variante A...")
        t0 = time.time()
        explainer_a = shap.TreeExplainer(model_a)
        shap_a = explainer_a.shap_values(X_sample)
        print(f"  SHAP A: {time.time() - t0:.1f}s")

        print("Computando SHAP para Variante B...")
        t0 = time.time()
        explainer_b = shap.TreeExplainer(model_b)
        shap_b = explainer_b.shap_values(X_sample)
        print(f"  SHAP B: {time.time() - t0:.1f}s")

        print("\nGerando figuras SHAP...")
        plot_shap_comparison(shap_a, shap_b, features, FIG_DIR / "fig_v4_shap_comparison.png")

        mean_abs_a = np.abs(shap_a).mean(axis=0)
        mean_abs_b = np.abs(shap_b).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": features,
            "shap_mean_abs_A": np.round(mean_abs_a, 6),
            "shap_mean_abs_B": np.round(mean_abs_b, 6),
        })
        shap_df = shap_df.sort_values("shap_mean_abs_B", ascending=False)
        shap_df.to_csv(MODEL_DIR / "shap_importance_v4.csv", index=False)
        print(f"  Salvo: shap_importance_v4.csv")

        report["shap_top10_A"] = shap_df.sort_values("shap_mean_abs_A", ascending=False).head(10)[
            ["feature", "shap_mean_abs_A"]].to_dict("records")
        report["shap_top10_B"] = shap_df.sort_values("shap_mean_abs_B", ascending=False).head(10)[
            ["feature", "shap_mean_abs_B"]].to_dict("records")
    else:
        report["shap_available"] = False

    print("\nGerando figuras Gain...")
    plot_gain_comparison(model_a, model_b, features, FIG_DIR / "fig_v4_gain_comparison.png")

    gain_a = model_a.get_booster().get_score(importance_type="gain")
    gain_b = model_b.get_booster().get_score(importance_type="gain")
    report["gain_top10_A"] = sorted(gain_a.items(), key=lambda x: -x[1])[:10]
    report["gain_top10_B"] = sorted(gain_b.items(), key=lambda x: -x[1])[:10]
    report["tempo_total_s"] = round(time.time() - t_total, 1)

    with open(DATA_DIR / "48_shap_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-48: SHAP + Gain ({time.time() - t_total:.0f}s)")
    print(f"  Amostras SHAP: {len(test_sample):,}")
    if has_shap:
        top_a = shap_df.sort_values("shap_mean_abs_A", ascending=False).head(5)["feature"].tolist()
        top_b = shap_df.sort_values("shap_mean_abs_B", ascending=False).head(5)["feature"].tolist()
        print(f"  Top-5 SHAP A: {top_a}")
        print(f"  Top-5 SHAP B: {top_b}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

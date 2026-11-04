#!/usr/bin/env python3
"""v4-47: Walk-forward v4 — 3 variantes × 3 sub-modelos × 4 folds = 36 modelos.

Baseado em scripts/31_validate_walkforward_v3.py.
Diferenças:
  - 3 variantes com janelas de treino diferentes.
  - Variante C usa sample_weight com decaimento exponencial.
  - Folds idênticos entre variantes (test=2023,2024,2025,2026).
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"

GROUP_KEY = "cod_mesorregiao"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

N_VAL_WEEKS = 10
VARIANTES = ["a", "b", "c"]
SUB_MODELS = ["sinan_only", "inmet_bruto", "inmet_enriquecido"]
TEST_YEARS = [2023, 2024, 2025, 2026]


def categorize_features(schema_path):
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return {
        "sinan_only": sinan,
        "inmet_bruto": sinan + inmet_bruto,
        "inmet_enriquecido": sinan + inmet_bruto + inmet_enriched,
    }


def build_folds(var_cfg):
    train_years = list(var_cfg["train"])
    folds = []
    for test_year in TEST_YEARS:
        folds.append({"train": list(train_years), "test": [test_year]})
        train_years.append(test_year)
    return folds


def split_train_val(train_data, n_val_weeks=N_VAL_WEEKS):
    week_keys = (
        train_data[["ano", "semana_epidemiologica"]]
        .drop_duplicates()
        .sort_values(["ano", "semana_epidemiologica"])
    )
    val_weeks = week_keys.tail(n_val_weeks)
    val_index = pd.MultiIndex.from_frame(val_weeks)
    train_week_index = pd.MultiIndex.from_frame(train_data[["ano", "semana_epidemiologica"]])
    val_mask = train_week_index.isin(val_index)
    return train_data[~val_mask], train_data[val_mask]


def compute_sample_weights(df, var_cfg):
    lam = var_cfg.get("sample_weight_lambda")
    ref = var_cfg.get("sample_weight_ref_year")
    if lam is None or ref is None:
        return None
    return np.exp(-lam * (ref - df["ano"].values.astype(float)))


def train_eval_fold(df, features, train_years, test_years, label, var_cfg=None):
    train_data = df[df["ano"].isin(train_years)]
    test_data = df[df["ano"].isin(test_years)]
    if len(train_data) == 0 or len(test_data) == 0:
        return None

    tr, val = split_train_val(train_data)
    if len(val) == 0:
        return None

    y_tr = np.log1p(tr[TARGET])
    y_val = np.log1p(val[TARGET])
    y_test_log = np.log1p(test_data[TARGET])
    y_test_orig = test_data[TARGET].values

    fit_kwargs = dict(eval_set=[(val[features], y_val)], verbose=0)
    if var_cfg is not None:
        sw = compute_sample_weights(tr, var_cfg)
        if sw is not None:
            fit_kwargs["sample_weight"] = sw

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(tr[features], y_tr, **fit_kwargs)

    pred_log = model.predict(test_data[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    r2_log = r2_score(y_test_log, pred_log)
    r2 = r2_score(y_test_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)

    print(f"    {label:40s}  R2_log={r2_log:.4f}  RMSE={rmse:.2f}  train={len(tr):,}  test={len(test_data):,}")

    return {
        "label": label,
        "train_years": train_years,
        "test_years": test_years,
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "R2_log": round(float(r2_log), 4),
        "train_rows": len(tr),
        "val_rows": len(val),
        "test_rows": len(test_data),
        "best_iteration": int(model.best_iteration),
    }


def plot_walk_forward(all_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_var = {"A": "#1976D2", "B": "#FF9800", "C": "#E53935"}
    markers_var = {"A": "o", "B": "s", "C": "^"}

    for ax, sub_model in zip(axes, SUB_MODELS):
        sub_label = {"sinan_only": "SINAN-only", "inmet_bruto": "INMET bruto",
                     "inmet_enriquecido": "INMET enriquecido"}[sub_model]

        for var in ["A", "B", "C"]:
            key = f"{var}_{sub_model}"
            if key not in all_results or not all_results[key]:
                continue
            r2s = [r["R2_log"] for r in all_results[key]]
            xs = range(1, len(r2s) + 1)
            ax.plot(xs, r2s, marker=markers_var[var], label=f"Var {var}",
                    color=colors_var[var], linewidth=2, markersize=8)

        ax.set_xticks(range(1, len(TEST_YEARS) + 1))
        ax.set_xticklabels([f"Fold {i+1}\n(test={y})" for i, y in enumerate(TEST_YEARS)], fontsize=8)
        ax.set_ylabel("R²_log")
        ax.set_title(sub_label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Walk-Forward v4 — R²_log por Fold e Variante", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print(f"  v4-47: Walk-Forward — 3 Variantes × 3 Sub-modelos × {len(TEST_YEARS)} Folds")
    print("=" * 70)

    print("\nCarregando dataset enriquecido v4...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_v4.parquet")
    df = df.sort_values(ID_COLS).reset_index(drop=True)
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas")

    print("\nComputando target (notificacoes_t4 = shift -4)...")
    df[TARGET] = df.groupby(GROUP_KEY)["notificacoes"].shift(-4)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Removidas {before - len(df):,} linhas sem target → {len(df):,} restantes")

    drop_cols = ["nome_mesorregiao", "uf_sigla", "nome_grande_regiao", "n_municipios",
                 "n_estacoes", "flag_fonte_inmet"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    non_numeric = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")]
    if non_numeric:
        df = df.drop(columns=non_numeric)

    feature_sets = categorize_features(SCHEMA_PATH)
    for name, feats in feature_sets.items():
        print(f"  {name}: {len(feats)} features")

    all_results = {}
    n_models = 0

    for var in VARIANTES:
        var_upper = var.upper()
        var_cfg = recorte_config.load_v4(f"variante_{var}")
        folds = build_folds(var_cfg)

        print(f"\n{'=' * 70}")
        print(f"  VARIANTE {var_upper}: {len(folds)} folds (train starts {var_cfg['train'][0]})")
        print(f"{'=' * 70}")

        for sub_name in SUB_MODELS:
            feats = feature_sets[sub_name]
            key = f"{var_upper}_{sub_name}"
            all_results[key] = []

            for i, fold in enumerate(folds, 1):
                sub_label_short = {"sinan_only": "SINAN", "inmet_bruto": "INMET-b",
                                   "inmet_enriquecido": "INMET-e"}[sub_name]
                label = f"Var{var_upper} {sub_label_short} F{i}(test={fold['test'][0]})"

                r = train_eval_fold(df, feats, fold["train"], fold["test"], label, var_cfg)
                if r:
                    all_results[key].append(r)
                    n_models += 1

    print(f"\n  Total modelos treinados: {n_models}")

    print("\nGerando figuras...")
    plot_walk_forward(all_results, FIG_DIR / "fig_v4_walkforward.png")

    summary = {}
    for key, fold_results in all_results.items():
        r2s = [r["R2_log"] for r in fold_results]
        summary[key] = {
            "mean_R2_log": round(float(np.mean(r2s)), 4) if r2s else None,
            "std_R2_log": round(float(np.std(r2s)), 4) if r2s else None,
            "folds": fold_results,
        }

    report = {
        "n_folds": len(TEST_YEARS),
        "n_models_total": n_models,
        "n_val_weeks": N_VAL_WEEKS,
        "modelos": summary,
        "tempo_total_s": round(time.time() - t_total, 1),
    }

    enriq_means = {}
    for var in ["A", "B", "C"]:
        k = f"{var}_inmet_enriquecido"
        enriq_means[var] = summary[k]["mean_R2_log"] if summary[k]["mean_R2_log"] is not None else 0

    report["resumo_enriquecido"] = {
        "mean_R2_log_A": enriq_means["A"],
        "mean_R2_log_B": enriq_means["B"],
        "mean_R2_log_C": enriq_means["C"],
        "delta_B_vs_A": round(enriq_means["B"] - enriq_means["A"], 4),
        "delta_C_vs_A": round(enriq_means["C"] - enriq_means["A"], 4),
    }

    with open(DATA_DIR / "47_walkforward_v4_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  RESUMO WALK-FORWARD v4 ({n_models} modelos, {elapsed:.0f}s):")
    for var in ["A", "B", "C"]:
        k = f"{var}_inmet_enriquecido"
        m = summary[k]
        print(f"  Var {var} (INMET enriquecido): R²_log médio = {m['mean_R2_log']:.4f} ± {m['std_R2_log']:.4f}")
    print(f"  Δ B vs A: {enriq_means['B'] - enriq_means['A']:+.4f}")
    print(f"  Δ C vs A: {enriq_means['C'] - enriq_means['A']:+.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

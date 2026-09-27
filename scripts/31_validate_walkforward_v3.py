#!/usr/bin/env python3
"""US-211: Walk-forward v3 (mesorregional) — 5 folds temporais, 3 modelos.

Usa o dataset completo `integrated_mesorregiao_v3.parquet` (não os splits
pré-definidos) para permitir janelas expansivas por ano. Para cada fold,
treina A (SINAN-only), B (INMET bruto) e C (INMET enriquecido) e avalia
R²_log no ano(s) de teste do fold. A validação (para early stopping) usa as
últimas 10 semanas epidemiológicas do conjunto de treino de cada fold.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

GROUP_KEY = "cod_mesorregiao"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v3.csv"

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

N_VAL_WEEKS = 10

FOLDS = [
    {"train": [2019], "test": [2021]},
    {"train": [2019, 2021], "test": [2023]},
    {"train": [2019, 2021, 2023], "test": [2024]},
    {"train": [2019, 2021, 2023, 2024], "test": [2025]},
    {"train": [2019, 2021, 2023, 2024, 2025], "test": [2026]},
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


def split_train_val(train_data, n_val_weeks=N_VAL_WEEKS):
    """Split training data: last N distinct (ano, semana_epidemiologica) weeks -> val."""
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


def train_eval_fold(df, features, train_years, test_years, label):
    """Train and evaluate one fold for one feature set."""
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

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(tr[features], y_tr, eval_set=[(val[features], y_val)], verbose=0)

    pred_log = model.predict(test_data[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    r2_log = r2_score(y_test_log, pred_log)
    r2 = r2_score(y_test_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)

    print(f"    {label:25s}  R2_log={r2_log:.4f}  RMSE={rmse:.2f}  train={len(tr):,}  val={len(val):,}  test={len(test_data):,}")

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


def plot_walk_forward(results_by_model, out_path):
    """Line plot of R²_log per fold, one line per model (A/B/C)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A: SINAN-only": "#1976D2", "B: INMET bruto": "#FF9800", "C: INMET enriquecido": "#E53935"}
    markers = {"A: SINAN-only": "o", "B: INMET bruto": "s", "C: INMET enriquecido": "^"}

    fold_labels = None
    for model_label, fold_results in results_by_model.items():
        if not fold_results:
            continue
        test_labels = ["/".join(str(y) for y in r["test_years"]) for r in fold_results]
        fold_labels = test_labels
        r2s = [r["R2_log"] for r in fold_results]
        ax.plot(range(1, len(r2s) + 1), r2s, marker=markers.get(model_label, "o"),
                label=model_label, color=colors.get(model_label), linewidth=2, markersize=8)

    if fold_labels:
        ax.set_xticks(range(1, len(fold_labels) + 1))
        ax.set_xticklabels([f"Fold {i+1}\n(test={t})" for i, t in enumerate(fold_labels)], fontsize=9)
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("R² (log)", fontsize=11)
    ax.set_title("Validação Walk-Forward v3 — R²_log por Fold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-211: Walk-Forward v3 Mesorregional — 5 Folds, 3 Modelos")
    print("=" * 60)

    print("\nCarregando dataset completo v3...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_v3.parquet")
    df = df.sort_values(ID_COLS).reset_index(drop=True)
    print(f"  {len(df):,} linhas, {len(df.columns)} colunas antes do target")

    print("\nComputando target (notificacoes_t4 = shift -4 por mesorregião)...")
    df[TARGET] = df.groupby(GROUP_KEY)["notificacoes"].shift(-4)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Removidas {before - len(df):,} linhas sem target -> {len(df):,} restantes")

    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(SCHEMA_PATH)
    feats_a = sinan_feats
    feats_b = sinan_feats + inmet_bruto
    feats_c = sinan_feats + inmet_bruto + inmet_enriched
    print(f"  Features: A={len(feats_a)}  B={len(feats_b)}  C={len(feats_c)}")

    model_specs = [
        ("A: SINAN-only", feats_a),
        ("B: INMET bruto", feats_b),
        ("C: INMET enriquecido", feats_c),
    ]

    results_by_model = {label: [] for label, _ in model_specs}

    for i, fold in enumerate(FOLDS, 1):
        print(f"\n--- Fold {i}: train={fold['train']} -> test={fold['test']} ---")
        for label, feats in model_specs:
            r = train_eval_fold(df, feats, fold["train"], fold["test"], label)
            if r:
                results_by_model[label].append(r)

        fold_r2 = {label: results_by_model[label][-1]["R2_log"] for label in results_by_model if results_by_model[label]}
        if len(fold_r2) == 3:
            delta_ac = fold_r2["C: INMET enriquecido"] - fold_r2["A: SINAN-only"]
            print(f"  Delta C vs A: {delta_ac:+.4f}")

    print("\nGerando figuras...")
    plot_walk_forward(results_by_model, FIG_DIR / "fig_v3_walkforward.png")

    summary = {}
    for label, fold_results in results_by_model.items():
        r2s = [r["R2_log"] for r in fold_results]
        summary[label] = {
            "mean_R2_log": round(float(np.mean(r2s)), 4) if r2s else None,
            "std_R2_log": round(float(np.std(r2s)), 4) if r2s else None,
            "folds": fold_results,
        }

    mean_a = summary["A: SINAN-only"]["mean_R2_log"]
    mean_b = summary["B: INMET bruto"]["mean_R2_log"]
    mean_c = summary["C: INMET enriquecido"]["mean_R2_log"]

    deltas_ac = [
        c["R2_log"] - a["R2_log"]
        for a, c in zip(results_by_model["A: SINAN-only"], results_by_model["C: INMET enriquecido"])
    ]
    folds_c_better = sum(1 for d in deltas_ac if d > 0)

    report = {
        "n_folds": len(FOLDS),
        "n_val_weeks": N_VAL_WEEKS,
        "modelos": {
            "A_sinan_only": summary["A: SINAN-only"],
            "B_inmet_bruto": summary["B: INMET bruto"],
            "C_inmet_enriquecido": summary["C: INMET enriquecido"],
        },
        "deltas_mean": {
            "B_vs_A": round(float(mean_b - mean_a), 4) if mean_a is not None and mean_b is not None else None,
            "C_vs_A": round(float(mean_c - mean_a), 4) if mean_a is not None and mean_c is not None else None,
            "C_vs_B": round(float(mean_c - mean_b), 4) if mean_b is not None and mean_c is not None else None,
        },
        "folds_c_better_than_a": int(folds_c_better),
        "conclusao": {
            "estavel_ao_longo_do_tempo": bool(np.std([r2 for r2 in [
                r["R2_log"] for r in results_by_model["C: INMET enriquecido"]
            ]]) < 0.1) if results_by_model["C: INMET enriquecido"] else None,
            "inmet_enriquecido_ajuda_na_maioria_dos_folds": bool(folds_c_better > len(FOLDS) / 2),
        },
    }
    with open(DATA_DIR / "31_walkforward_v3_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO WALK-FORWARD v3 ({len(FOLDS)} folds):")
    print(f"  A (SINAN-only):        R²_log médio = {mean_a:.4f}")
    print(f"  B (INMET bruto):       R²_log médio = {mean_b:.4f}")
    print(f"  C (INMET enriquecido): R²_log médio = {mean_c:.4f}")
    print(f"  Folds C > A: {folds_c_better}/{len(FOLDS)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

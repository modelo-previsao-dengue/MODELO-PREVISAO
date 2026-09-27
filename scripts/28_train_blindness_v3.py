#!/usr/bin/env python3
"""US-208: Data Blindness v3 (mesorregional) — simulação de atraso real do SINAN.

Cenários:
  - Full: modelo completo (referência)
  - Blind-4w: lags SINAN 1-4 semanas + diff/pct_change/média-móvel curta → NaN
    (simula atraso de notificação de 4 semanas)
  - Blind-8w: Blind-4w + lags 5-8 semanas + média-móvel longa → NaN
    (simula atraso de notificação de 8 semanas)
  - INMET-only: TODAS as features SINAN (categoria "sinan" do schema) → NaN
    (só restam clima + sazonalidade)

Para cada cenário, treina com e sem INMET para medir o valor do INMET
quando dados SINAN recentes não estão disponíveis (ΔR²_log = com − sem).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
MODEL_DIR = BASE_DIR / "models" / "blindness_v3"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v3.csv"

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

# Nomes de features v3 (schema categoria "sinan") — diferem do v2 apenas no
# sufixo "w" e no agrupamento das médias móveis (janelas combinadas em vez de
# uma feature por tamanho de janela), a semântica temporal é a mesma.
BLIND_4W_MASK = [
    "notificacoes_lag_1w", "notificacoes_lag_2w", "notificacoes_lag_3w", "notificacoes_lag_4w",
    "notificacoes_diff_1w",
    "notificacoes_pct_change_1w",
    "notificacoes_media_movel_2_4w",
]

BLIND_8W_MASK = BLIND_4W_MASK + [
    "notificacoes_lag_5w", "notificacoes_lag_6w", "notificacoes_lag_7w", "notificacoes_lag_8w",
    "notificacoes_media_movel_4_8w",
]


def categorize_features(schema_path):
    """Read feature_schema_v3.csv and split features into SINAN vs INMET (todas as sub-categorias)."""
    schema = pd.read_csv(schema_path)
    sinan_feats = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_feats = schema.loc[
        schema["category"].isin(["inmet_bruto", "inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]),
        "feature",
    ].tolist()
    return sinan_feats, inmet_feats


def apply_blindness(df, scenario, sinan_feats):
    """Apply data blindness by masking SINAN features to NaN."""
    df_blind = df.copy()
    if scenario == "blind_4w":
        cols_to_mask = [c for c in BLIND_4W_MASK if c in df_blind.columns]
    elif scenario == "blind_8w":
        cols_to_mask = [c for c in BLIND_8W_MASK if c in df_blind.columns]
    elif scenario == "inmet_only":
        cols_to_mask = [c for c in sinan_feats if c in df_blind.columns]
    else:
        return df_blind, []

    df_blind[cols_to_mask] = np.nan
    return df_blind, cols_to_mask


def get_features(sinan_feats, inmet_feats, include_inmet=True):
    """Get feature list, optionally excluding INMET."""
    if include_inmet:
        return sinan_feats + inmet_feats
    return sinan_feats


def train_eval(train, val, test, features, label):
    """Train and evaluate one model."""
    y_train = np.log1p(train[TARGET])
    y_val = np.log1p(val[TARGET])
    y_test_log = np.log1p(test[TARGET])
    y_test_orig = test[TARGET].values

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[features], y_train, eval_set=[(val[features], y_val)], verbose=0)

    pred_log = model.predict(test[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    r2_log = r2_score(y_test_log, pred_log)
    r2_orig = r2_score(y_test_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)

    errors_sq = (y_test_log.values - pred_log) ** 2

    print(f"    {label:40s}  R²_log={r2_log:.4f}  RMSE={rmse:.2f}  feats={len(features)}")

    metrics = {
        "label": label,
        "n_features": len(features),
        "R2_log": round(float(r2_log), 4),
        "R2": round(float(r2_orig), 4),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 4),
        "best_iteration": int(model.best_iteration),
    }

    return model, metrics, errors_sq


def plot_blindness_comparison(results, out_path):
    """Barplot: R² by blindness scenario."""
    scenarios = []
    r2_with = []
    r2_without = []
    for scenario, data in results.items():
        scenarios.append(scenario)
        r2_with.append(data["com_inmet"]["R2_log"])
        r2_without.append(data["sem_inmet"]["R2_log"])

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, r2_without, width, label="Sem INMET", color="#1976D2")
    bars2 = ax.bar(x + width / 2, r2_with, width, label="Com INMET", color="#E53935")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("R² (log)", fontsize=11)
    ax.set_title("Data Blindness v3: Impacto do INMET em Cenários de Atraso SINAN", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_delta(results, out_path):
    """Barplot: how much INMET adds in each scenario."""
    scenarios = list(results.keys())
    deltas = [results[s]["delta_R2_log"] for s in scenarios]
    colors = ["#4CAF50" if d > 0 else "#E53935" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(scenarios)), deltas, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_ylabel("ΔR²_log (com INMET − sem INMET)", fontsize=11)
    ax.set_title("Contribuição do INMET por Cenário de Data Blindness (v3)", fontsize=12)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.001 if d > 0 else -0.003),
                f"{d:+.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-208: Data Blindness v3 — Simulação de Atraso SINAN (mesorregional)")
    print("=" * 60)

    print("\nCarregando splits v3...")
    train = pd.read_parquet(DATA_DIR / "train_v3.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v3.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v3.parquet")
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    sinan_feats, inmet_feats = categorize_features(SCHEMA_PATH)
    print(f"  Features: {len(sinan_feats)} SINAN + {len(inmet_feats)} INMET")

    scenarios = {
        "Full (referência)": "full",
        "Blind-4w": "blind_4w",
        "Blind-8w": "blind_8w",
        "INMET-only": "inmet_only",
    }

    results = {}

    for scenario_name, scenario_key in scenarios.items():
        print(f"\n{'─' * 50}")
        print(f"  Cenário: {scenario_name}")
        print(f"{'─' * 50}")

        if scenario_key == "full":
            train_b, val_b, test_b = train, val, test
            masked_cols = []
        else:
            train_b, _ = apply_blindness(train, scenario_key, sinan_feats)
            val_b, _ = apply_blindness(val, scenario_key, sinan_feats)
            test_b, masked_cols = apply_blindness(test, scenario_key, sinan_feats)
            print(f"  Mascaradas: {len(masked_cols)} features SINAN")

        feats_with = get_features(sinan_feats, inmet_feats, include_inmet=True)
        feats_without = get_features(sinan_feats, inmet_feats, include_inmet=False)

        model_with, met_with, err_with = train_eval(train_b, val_b, test_b, feats_with, f"{scenario_name} + INMET")
        model_without, met_without, err_without = train_eval(train_b, val_b, test_b, feats_without, f"{scenario_name} sem INMET")

        t_stat, p_val = ttest_rel(err_without, err_with)
        delta = met_with["R2_log"] - met_without["R2_log"]

        results[scenario_name] = {
            "com_inmet": met_with,
            "sem_inmet": met_without,
            "delta_R2_log": round(float(delta), 4),
            "ttest": {"t_stat": round(float(t_stat), 4), "p_value": float(p_val)},
            "n_masked": len(masked_cols),
            "inmet_ajuda": delta > 0,
        }

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  ΔR²_log = {delta:+.4f} (p={p_val:.2e}, {sig})")
        print(f"  INMET ajuda? {'SIM' if delta > 0 else 'NÃO'}")

        print("  Salvando modelos...")
        model_with.save_model(str(MODEL_DIR / f"model_{scenario_key}_com_inmet.ubj"))
        model_without.save_model(str(MODEL_DIR / f"model_{scenario_key}_sem_inmet.ubj"))

    print("\nGerando figuras...")
    plot_blindness_comparison(results, FIG_DIR / "fig_v3_blindness_comparison.png")
    plot_delta(results, FIG_DIR / "fig_v3_blindness_delta.png")

    report = {
        "cenarios": results,
        "conclusao": {
            "cenario_inmet_mais_ajuda": max(results, key=lambda k: results[k]["delta_R2_log"]),
            "delta_maximo": max(r["delta_R2_log"] for r in results.values()),
            "inmet_compensa_atraso_4w": results.get("Blind-4w", {}).get("inmet_ajuda", False),
            "inmet_compensa_atraso_8w": results.get("Blind-8w", {}).get("inmet_ajuda", False),
        },
    }
    with open(DATA_DIR / "28_blindness_v3_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-208 — DATA BLINDNESS v3:")
    for name, data in results.items():
        print(f"  {name:20s}  com={data['com_inmet']['R2_log']:.4f}  sem={data['sem_inmet']['R2_log']:.4f}  Δ={data['delta_R2_log']:+.4f}")
    best = report["conclusao"]["cenario_inmet_mais_ajuda"]
    print(f"\n  INMET mais ajuda no cenário: {best} (Δ={report['conclusao']['delta_maximo']:+.4f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

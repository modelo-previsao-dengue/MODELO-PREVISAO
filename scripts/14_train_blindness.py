#!/usr/bin/env python3
"""US-114: Data Blindness — simulação de atraso real do SINAN.

Cenários:
  - Full: modelo completo (referência)
  - Blind-4w: lags SINAN 1-4 → NaN (atraso de 4 semanas)
  - Blind-8w: lags SINAN 1-8 → NaN + médias móveis curtas → NaN
  - INMET-only: TODAS features SINAN de lag/média → NaN (só clima + sazonalidade)

Para cada cenário, treina com e sem INMET para medir o valor do INMET
quando dados SINAN recentes não estão disponíveis.
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
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "blindness_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

BLIND_4W_MASK = [
    "notificacoes_lag_1", "notificacoes_lag_2", "notificacoes_lag_3", "notificacoes_lag_4",
    "notificacoes_media_movel_3", "notificacoes_media_movel_4",
    "notificacoes_min_movel_3", "notificacoes_min_movel_4",
    "notificacoes_max_movel_3", "notificacoes_max_movel_4",
    "notificacoes_diff_1", "notificacoes_diff_4",
    "notificacoes_pct_change_1", "notificacoes_pct_change_4",
    "notificacoes_aceleracao_1",
    "notificacoes_razao_media_4",
]

BLIND_8W_MASK = BLIND_4W_MASK + [
    "notificacoes_lag_8",
    "notificacoes_media_movel_8",
    "notificacoes_min_movel_8",
    "notificacoes_max_movel_8",
    "notificacoes_razao_media_8",
]

INMET_ONLY_MASK_PATTERNS = [
    "notificacoes_lag_", "notificacoes_media_movel_", "notificacoes_min_movel_",
    "notificacoes_max_movel_", "notificacoes_diff_", "notificacoes_pct_change_",
    "notificacoes_aceleracao_", "notificacoes_razao_media_",
    "qt_hospitalizados", "qt_obitos", "qt_confirmados", "qt_descartados",
    "qt_inconclusivos", "qt_dengue_alarme", "qt_dengue_grave", "qt_chikungunya",
    "idade_media_anos", "atraso_notificacao",
    "prop_", "indice_", "label_", "is_zero_notification_week",
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def apply_blindness(df, scenario):
    """Apply data blindness by masking SINAN features to NaN."""
    df_blind = df.copy()
    if scenario == "blind_4w":
        cols_to_mask = [c for c in BLIND_4W_MASK if c in df_blind.columns]
    elif scenario == "blind_8w":
        cols_to_mask = [c for c in BLIND_8W_MASK if c in df_blind.columns]
    elif scenario == "inmet_only":
        cols_to_mask = [c for c in df_blind.columns
                        if any(c.startswith(p) for p in INMET_ONLY_MASK_PATTERNS)
                        and c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    else:
        return df_blind, []

    df_blind[cols_to_mask] = np.nan
    return df_blind, cols_to_mask


def get_features(df, include_inmet=True):
    """Get feature list, optionally excluding INMET."""
    all_feats = [c for c in df.columns
                 if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                 and df[c].dtype != "object" and not str(df[c].dtype).startswith("datetime")]
    if include_inmet:
        return all_feats
    return [c for c in all_feats if not is_inmet_feature(c)]


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

    return {
        "label": label,
        "n_features": len(features),
        "R2_log": round(float(r2_log), 4),
        "R2": round(float(r2_orig), 4),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 4),
    }, errors_sq


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
    ax.set_title("Data Blindness: Impacto do INMET em Cenários de Atraso SINAN", fontsize=12)
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
    ax.set_title("Contribuição do INMET por Cenário de Data Blindness", fontsize=12)
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
    print("  US-114: Data Blindness — Simulação de Atraso SINAN")
    print("=" * 60)

    print("\nCarregando splits v2...")
    train = pd.read_parquet(DATA_DIR / "train_v2.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v2.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v2.parquet")
    for d in [train, val, test]:
        d["ibge_municipio"] = d["ibge_municipio"].astype(str)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

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
            train_b, masked_train = apply_blindness(train, scenario_key)
            val_b, _ = apply_blindness(val, scenario_key)
            test_b, masked_cols = apply_blindness(test, scenario_key)
            print(f"  Mascaradas: {len(masked_cols)} features SINAN")

        feats_with = get_features(test_b, include_inmet=True)
        feats_without = get_features(test_b, include_inmet=False)

        met_with, err_with = train_eval(train_b, val_b, test_b, feats_with, f"{scenario_name} + INMET")
        met_without, err_without = train_eval(train_b, val_b, test_b, feats_without, f"{scenario_name} sem INMET")

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

    print("\nGerando figuras...")
    plot_blindness_comparison(results, FIG_DIR / "fig_5yr_v2_blindness_comparison.png")
    plot_delta(results, FIG_DIR / "fig_5yr_v2_blindness_delta.png")

    report = {
        "cenarios": results,
        "conclusao": {
            "cenario_inmet_mais_ajuda": max(results, key=lambda k: results[k]["delta_R2_log"]),
            "delta_maximo": max(r["delta_R2_log"] for r in results.values()),
            "inmet_compensa_atraso_4w": results.get("Blind-4w", {}).get("inmet_ajuda", False),
            "inmet_compensa_atraso_8w": results.get("Blind-8w", {}).get("inmet_ajuda", False),
        },
    }
    with open(DATA_DIR / "14_blindness_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-114 — DATA BLINDNESS:")
    for name, data in results.items():
        print(f"  {name:20s}  com={data['com_inmet']['R2_log']:.4f}  sem={data['sem_inmet']['R2_log']:.4f}  Δ={data['delta_R2_log']:+.4f}")
    best = report["conclusao"]["cenario_inmet_mais_ajuda"]
    print(f"\n  INMET mais ajuda no cenário: {best} (Δ={report['conclusao']['delta_maximo']:+.4f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

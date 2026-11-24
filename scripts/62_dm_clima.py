#!/usr/bin/env python3
"""v4-62: Teste estatístico formal para contribuição do clima.

Compara modelo realista com_clima vs sem_clima usando:
1. Diebold-Mariano ponto-a-ponto (escala log, h=4)
2. Diebold-Mariano sobre perda média semanal entre regiões (agregação temporal)
3. Bootstrap em blocos (mesorregião×ano) para IC do Δ R²_log
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"
AUDIT_PATH = DATA_DIR / "feature_availability_audit.json"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"

CLIMA_PREFIXES = ("rain_", "temp_", "humidity_", "pressure_", "wind_", "radiation_")

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)


def load_splits():
    train = pd.read_parquet(DATA_DIR / "splits_b" / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "splits_b" / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "splits_b" / "test.parquet")
    return train, val, test


def add_notificacoes(train, val, test):
    if "notificacoes" in train.columns:
        return
    full = pd.read_parquet(
        DATA_DIR / "integrated_mesorregiao_v4.parquet",
        columns=ID_COLS + ["notificacoes"]
    )
    for df in [train, val, test]:
        merged = df.merge(full, on=ID_COLS, how="left", suffixes=("", "_new"))
        col = "notificacoes_new" if "notificacoes_new" in merged.columns else "notificacoes"
        df["notificacoes"] = merged[col].values


def get_realista_features():
    schema = pd.read_csv(SCHEMA_PATH)
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    delayed = set(audit["delayed_consolidation"]["features"].keys())
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet = schema.loc[schema["category"] != "sinan", "feature"].tolist()
    realtime_sinan = [f for f in sinan if f not in delayed]
    return realtime_sinan + ["notificacoes"] + inmet


def split_clima(features):
    com = [f for f in features]
    sem = [f for f in features if not any(f.startswith(p) for p in CLIMA_PREFIXES)]
    return com, sem


def train_and_predict(train, val, test, features, label):
    available = [f for f in features if f in train.columns]
    print(f"  [{label}] {len(available)} features")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[available].values, np.log1p(train[TARGET].values),
        eval_set=[(val[available].values, np.log1p(val[TARGET].values))],
        verbose=False
    )
    pred_log = model.predict(test[available].values)
    pred_raw = np.expm1(np.maximum(pred_log, 0))

    y_log = np.log1p(test[TARGET].values)
    p_log = np.log1p(np.maximum(pred_raw, 0))
    r2 = round(r2_score(y_log, p_log), 4)
    print(f"    R²_log={r2} | iter={model.best_iteration}")
    return pred_raw, r2


def dm_pointwise(y_true, pred_a, pred_b, h=4):
    """DM ponto-a-ponto em escala log. Positivo = A melhor que B."""
    e_a = np.log1p(y_true) - np.log1p(np.maximum(pred_a, 0))
    e_b = np.log1p(y_true) - np.log1p(np.maximum(pred_b, 0))

    d = e_b**2 - e_a**2
    n = len(d)
    d_bar = np.mean(d)
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n

    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": round(float(dm_stat), 4),
        "p_value": float(p_value),
        "p_value_str": f"{p_value:.4f}",
        "d_bar": round(float(d_bar), 6),
        "h": h,
        "n": n,
        "A_melhor": dm_stat > 0,
    }


def dm_weekly_aggregate(test, pred_a, pred_b, h=4):
    """DM sobre perda média semanal entre regiões (elimina correlação espacial)."""
    test_c = test[["ano", "semana_epidemiologica", TARGET]].copy()
    test_c["loss_a"] = (np.log1p(test_c[TARGET].values) - np.log1p(np.maximum(pred_a, 0)))**2
    test_c["loss_b"] = (np.log1p(test_c[TARGET].values) - np.log1p(np.maximum(pred_b, 0)))**2

    weekly = test_c.groupby(["ano", "semana_epidemiologica"])[["loss_a", "loss_b"]].mean().reset_index()
    weekly = weekly.sort_values(["ano", "semana_epidemiologica"]).reset_index(drop=True)

    d = weekly["loss_b"].values - weekly["loss_a"].values
    n = len(d)
    d_bar = np.mean(d)
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n

    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": round(float(dm_stat), 4),
        "p_value": float(p_value),
        "p_value_str": f"{p_value:.4f}",
        "n_semanas": n,
        "A_melhor": dm_stat > 0,
    }


def block_bootstrap(test, pred_a, pred_b, n_boot=5000, seed=42):
    """Bootstrap em blocos (mesorregião×ano) para IC de Δ R²_log."""
    rng = np.random.RandomState(seed)
    test_c = test.copy()
    test_c["pred_a"] = pred_a
    test_c["pred_b"] = pred_b
    test_c["block"] = test_c["cod_mesorregiao"].astype(str) + "_" + test_c["ano"].astype(str)

    blocks = test_c["block"].unique()
    block_data = {b: test_c[test_c["block"] == b] for b in blocks}
    n_blocks = len(blocks)

    deltas = np.zeros(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(blocks, size=n_blocks, replace=True)
        rows = pd.concat([block_data[b] for b in sampled], ignore_index=True)
        y_log = np.log1p(rows[TARGET].values)
        pa_log = np.log1p(np.maximum(rows["pred_a"].values, 0))
        pb_log = np.log1p(np.maximum(rows["pred_b"].values, 0))
        deltas[i] = r2_score(y_log, pa_log) - r2_score(y_log, pb_log)

    return {
        "n_boot": n_boot,
        "n_blocks": n_blocks,
        "delta_R2_log": {
            "mean": round(float(np.mean(deltas)), 4),
            "median": round(float(np.median(deltas)), 4),
            "ci_95": [round(float(np.percentile(deltas, 2.5)), 4),
                      round(float(np.percentile(deltas, 97.5)), 4)],
            "pct_positivo": round(float(np.mean(deltas > 0) * 100), 1),
        },
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v4-62: DM + Bootstrap — Clima vs Sem Clima")
    print("=" * 60)

    train, val, test = load_splits()
    add_notificacoes(train, val, test)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    features_all = get_realista_features()
    features_com, features_sem = split_clima(features_all)
    print(f"\nFeatures com clima: {len(features_com)} | sem clima: {len(features_sem)}")

    print("\nTreinando modelos...")
    pred_com, r2_com = train_and_predict(train, val, test, features_com, "com_clima")
    pred_sem, r2_sem = train_and_predict(train, val, test, features_sem, "sem_clima")
    delta_r2 = round(r2_com - r2_sem, 4)
    print(f"\nΔ R²_log = {delta_r2} (com - sem)")

    report = {
        "com_clima": {"R2_log": r2_com, "n_features": len([f for f in features_com if f in train.columns])},
        "sem_clima": {"R2_log": r2_sem, "n_features": len([f for f in features_sem if f in train.columns])},
        "delta_R2_log": delta_r2,
    }

    print("\n--- DM ponto-a-ponto (com_clima vs sem_clima) ---")
    dm_pw = dm_pointwise(test[TARGET].values, pred_com, pred_sem, h=4)
    report["dm_pointwise"] = dm_pw
    print(f"  DM={dm_pw['DM_stat']} | p={dm_pw['p_value_str']}")

    print("\n--- DM agregado semanal ---")
    dm_wk = dm_weekly_aggregate(test, pred_com, pred_sem, h=4)
    report["dm_weekly_aggregate"] = dm_wk
    print(f"  DM={dm_wk['DM_stat']} | p={dm_wk['p_value_str']} | n_semanas={dm_wk['n_semanas']}")

    print(f"\n--- Bootstrap em blocos (5000 iter) ---")
    boot = block_bootstrap(test, pred_com, pred_sem, n_boot=5000)
    report["bootstrap_blocos"] = boot
    d = boot["delta_R2_log"]
    print(f"  Δ R²_log: {d['mean']} [{d['ci_95'][0]}, {d['ci_95'][1]}] | {d['pct_positivo']}% positivo")

    report["tempo_total_s"] = round(time.time() - t0, 1)

    out_path = DATA_DIR / "62_dm_clima_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v4-63: Cenários de atraso com features realistas.

Refaz os cenários de blindness do script 56 usando o conjunto realista
(203 features, sem delayed_consolidation).

Cenários:
  1. Completo realista (203 features)
  2. Atraso 4w (remove lags 1-4w, diff, pct_change, media_movel_2_4w)
  3. Atraso 8w (atraso 4w + lags 5-8w, media_movel_4_8w)
  4. Só clima (INMET + week_of_year)

Cada cenário com_clima e sem_clima.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"
AUDIT_PATH = DATA_DIR / "feature_availability_audit.json"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"

CLIMA_PREFIXES = ("rain_", "temp_", "humidity_", "pressure_", "wind_", "radiation_")

BLIND_4W = [
    "notificacoes_lag_1w", "notificacoes_lag_2w", "notificacoes_lag_3w", "notificacoes_lag_4w",
    "notificacoes_diff_1w", "notificacoes_pct_change_1w", "notificacoes_media_movel_2_4w",
]
BLIND_8W = BLIND_4W + [
    "notificacoes_lag_5w", "notificacoes_lag_6w", "notificacoes_lag_7w", "notificacoes_lag_8w",
    "notificacoes_media_movel_4_8w",
]

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


def remove_clima(features):
    return [f for f in features if not any(f.startswith(p) for p in CLIMA_PREFIXES)]


def metrics(y_true, y_pred):
    y_log = np.log1p(y_true)
    p_log = np.log1p(np.maximum(y_pred, 0))
    return {
        "R2_log": round(r2_score(y_log, p_log), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
    }


def train_eval(train, val, test, features, label):
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
    m = metrics(test[TARGET].values, pred_raw)
    print(f"    R²_log={m['R2_log']} | R²={m['R2']} | MAE={m['MAE']} | iter={model.best_iteration}")
    return m, len(available)


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
    print("  v4-63: Cenários de Atraso com Features Realistas")
    print("=" * 60)

    train, val, test = load_splits()
    add_notificacoes(train, val, test)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    base_features = get_realista_features()

    scenarios = {
        "completo_realista": base_features,
        "atraso_4w": [f for f in base_features if f not in BLIND_4W],
        "atraso_8w": [f for f in base_features if f not in BLIND_8W],
        "so_clima": [f for f in base_features
                     if any(f.startswith(p) for p in CLIMA_PREFIXES)
                     or f in ("week_of_year_sin", "week_of_year_cos")],
    }

    report = {}
    for scenario_name, features in scenarios.items():
        print(f"\n{'='*40}")
        print(f"  Cenário: {scenario_name}")
        print(f"{'='*40}")

        report[scenario_name] = {}

        m_com, n_com = train_eval(train, val, test, features, f"{scenario_name} com_clima")
        report[scenario_name]["com_clima"] = {**m_com, "n_features": n_com}

        features_sem = remove_clima(features)
        if len([f for f in features_sem if f in train.columns]) > 0:
            m_sem, n_sem = train_eval(train, val, test, features_sem, f"{scenario_name} sem_clima")
            report[scenario_name]["sem_clima"] = {**m_sem, "n_features": n_sem}
            report[scenario_name]["delta_clima"] = round(m_com["R2_log"] - m_sem["R2_log"], 4)
        else:
            report[scenario_name]["sem_clima"] = None
            report[scenario_name]["delta_clima"] = None

    print("\n\n--- Resumo ---")
    print(f"{'Cenário':<25} {'Com clima':>10} {'Sem clima':>10} {'Δ clima':>10}")
    print("-" * 55)
    for name, data in report.items():
        com = data["com_clima"]["R2_log"]
        sem = data["sem_clima"]["R2_log"] if data["sem_clima"] else "N/A"
        delta = data["delta_clima"] if data["delta_clima"] is not None else "N/A"
        print(f"{name:<25} {com:>10} {sem:>10} {delta:>10}")

    report["tempo_total_s"] = round(time.time() - t0, 1)

    out_path = DATA_DIR / "63_atraso_realista_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

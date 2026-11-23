#!/usr/bin/env python3
"""v4-58: Avaliação do vazamento temporal nos atributos clínicos da semana t.

Treina 3 variantes do modelo B (série longa, INMET enriquecido):
  1. Completo (266 features) — reprodução do script 46
  2. Realista (exclui features delayed_consolidation, inclui notificacoes da semana t)
  3. Realista sem notificacoes_t (igual ao 2 mas sem notificacoes)

Para cada variante: R²_log, R², MAE, RMSE no teste e por ano, SHAP top 10.
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
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"
AUDIT_PATH = DATA_DIR / "feature_availability_audit.json"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"

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


def get_feature_sets():
    schema = pd.read_csv(SCHEMA_PATH)
    all_features = schema["feature"].tolist()

    with open(AUDIT_PATH) as f:
        audit = json.load(f)

    delayed = set(audit["delayed_consolidation"]["features"].keys())

    sinan_features = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_features = schema.loc[schema["category"] != "sinan", "feature"].tolist()

    realtime_sinan = [f for f in sinan_features if f not in delayed]

    return {
        "completo": all_features,
        "realista": realtime_sinan + ["notificacoes"] + inmet_features,
        "realista_sem_notif_t": realtime_sinan + inmet_features,
    }


def metrics(y_true, y_pred):
    y_log = np.log1p(y_true)
    p_log = np.log1p(np.maximum(y_pred, 0))
    return {
        "R2_log": round(r2_score(y_log, p_log), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
    }


def metrics_by_year(test, pred):
    result = {}
    test = test.copy()
    test["pred"] = pred
    for year, g in test.groupby("ano"):
        result[int(year)] = metrics(g[TARGET].values, g["pred"].values)
    return result


def train_and_eval(train, val, test, features, label):
    available = [f for f in features if f in train.columns]
    missing = [f for f in features if f not in train.columns]

    if "notificacoes" in features and "notificacoes" not in train.columns:
        print(f"  AVISO: 'notificacoes' não está no split — verificar")

    print(f"\n  [{label}] {len(available)} features ({len(missing)} faltando)")
    if missing:
        print(f"    Faltando: {missing[:5]}...")

    X_train = train[available].values
    y_train = np.log1p(train[TARGET].values)
    X_val = val[available].values
    y_val = np.log1p(val[TARGET].values)
    X_test = test[available].values
    y_test_raw = test[TARGET].values

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_log = model.predict(X_test)
    pred_raw = np.expm1(np.maximum(pred_log, 0))

    m = metrics(y_test_raw, pred_raw)
    m_by_year = metrics_by_year(test, pred_raw)
    best_iter = model.best_iteration

    print(f"    R²_log={m['R2_log']} | R²={m['R2']} | MAE={m['MAE']} | iter={best_iter}")

    return model, available, pred_raw, m, m_by_year, best_iter


def compute_shap(model, test, features, n_samples=10000):
    print(f"    Calculando SHAP ({n_samples} amostras)...")
    X = test[features].values
    if len(X) > n_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]

    explainer_data = xgb.DMatrix(X, feature_names=features)
    shap_values = model.get_booster().predict(explainer_data, pred_contribs=True)
    shap_values = shap_values[:, :-1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:10]
    top10 = [
        {"feature": features[i], "shap_mean_abs": round(float(mean_abs[i]), 4)}
        for i in top_idx
    ]
    return top10


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v4-58: Avaliação do Vazamento Temporal")
    print("=" * 60)

    train, val, test = load_splits()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    if "notificacoes" not in train.columns:
        print("\nCarregando 'notificacoes' do dataset integrado...")
        full = pd.read_parquet(
            DATA_DIR / "integrated_mesorregiao_v4.parquet",
            columns=ID_COLS + ["notificacoes"]
        )
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            merged = split_df.merge(full, on=ID_COLS, how="left", suffixes=("", "_orig"))
            if "notificacoes" not in split_df.columns:
                split_df["notificacoes"] = merged["notificacoes"].values
                if "notificacoes_orig" in merged.columns:
                    split_df["notificacoes"] = merged["notificacoes_orig"].values
            print(f"  {split_name}: notificacoes adicionada, NaN={split_df['notificacoes'].isna().sum()}")
        train, val, test = train, val, test

    feature_sets = get_feature_sets()
    report = {"cenarios": {}}

    persistencia = test.groupby("cod_mesorregiao")["notificacoes"].shift(0).values
    if np.isnan(persistencia).any():
        persistencia = np.nan_to_num(persistencia, 0)

    for label, features in feature_sets.items():
        model, used_features, pred, m, m_year, best_iter = train_and_eval(
            train, val, test, features, label
        )

        shap_top10 = compute_shap(model, test, used_features)

        report["cenarios"][label] = {
            "n_features": len(used_features),
            "features_usadas": used_features,
            "best_iteration": best_iter,
            "teste_agregado": m,
            "teste_por_ano": m_year,
            "shap_top10": shap_top10,
        }

        if label == "realista":
            model.save_model(str(MODEL_DIR / "xgb_B_realista.json"))
            print(f"    Modelo salvo: xgb_B_realista.json")

    c = report["cenarios"]["completo"]["teste_agregado"]["R2_log"]
    r = report["cenarios"]["realista"]["teste_agregado"]["R2_log"]
    delta = round(c - r, 4)

    if abs(delta) < 0.01:
        decisao = "marginal"
        desc = f"Queda de {delta} R²_log — vazamento é marginal. Reportar como limitação, manter completo como referência."
    elif abs(delta) < 0.03:
        decisao = "material"
        desc = f"Queda de {delta} R²_log — vazamento é material. Realista vira resultado principal."
    else:
        decisao = "grave"
        desc = f"Queda de {delta} R²_log — vazamento é grave. Apenas o realista é válido."

    report["comparacao"] = {
        "delta_R2_log_completo_vs_realista": delta,
        "decisao": decisao,
        "descricao": desc,
    }

    pers_m = metrics(test[TARGET].values, persistencia)
    report["persistencia"] = {
        "teste_agregado": pers_m,
        "nota": "Persistência usa notificacoes da semana t como previsão para t+4"
    }

    report["tempo_total_s"] = round(time.time() - t0, 1)

    out_path = DATA_DIR / "58_vazamento_temporal_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

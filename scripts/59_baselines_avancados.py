#!/usr/bin/env python3
"""v4-59: Baselines avançados e controles.

1. Climatologia por mesorregião-semana (média histórica do treino B)
2. Ridge autorregressivo (features realistas)
3. Controle região+semana (XGBoost com cod_mesorregiao + week_of_year)
4. Controle região+semana+clima
5. Retreino B com cobertura alta (2008-2022) e só linhas com clima
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
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


def add_notificacoes(train, val, test):
    if "notificacoes" in train.columns:
        return
    full = pd.read_parquet(
        DATA_DIR / "integrated_mesorregiao_v4.parquet",
        columns=ID_COLS + ["notificacoes"]
    )
    for name, df in [("train", train), ("val", val), ("test", test)]:
        merged = df.merge(full, on=ID_COLS, how="left", suffixes=("", "_new"))
        col = "notificacoes_new" if "notificacoes_new" in merged.columns else "notificacoes"
        df["notificacoes"] = merged[col].values


def metrics(y_true, y_pred):
    y_log = np.log1p(y_true)
    p_log = np.log1p(np.maximum(y_pred, 0))
    return {
        "R2_log": round(r2_score(y_log, p_log), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
    }


def metrics_by_year(df, pred):
    result = {}
    df = df.copy()
    df["pred"] = pred
    for year, g in df.groupby("ano"):
        result[int(year)] = metrics(g[TARGET].values, g["pred"].values)
    return result


def get_realista_features():
    schema = pd.read_csv(SCHEMA_PATH)
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    delayed = set(audit["delayed_consolidation"]["features"].keys())
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet = schema.loc[schema["category"] != "sinan", "feature"].tolist()
    realtime_sinan = [f for f in sinan if f not in delayed]
    return realtime_sinan + ["notificacoes"] + inmet


def climatologia(train, test):
    """Média histórica de notificacoes por mesorregião × semana+4 no treino."""
    print("\n--- Climatologia por mesorregião-semana ---")
    train_c = train[["cod_mesorregiao", "semana_epidemiologica", TARGET]].copy()
    clim = train_c.groupby(["cod_mesorregiao", "semana_epidemiologica"])[TARGET].mean()

    test_c = test.copy()
    test_c["sem_target"] = (test_c["semana_epidemiologica"] + 4 - 1) % 52 + 1
    pred = test_c.set_index(["cod_mesorregiao", "sem_target"]).index.map(
        lambda x: clim.get(x, clim.mean())
    )
    pred = np.array([clim.get((r, s), clim.mean()) for r, s in
                     zip(test_c["cod_mesorregiao"], test_c["semana_epidemiologica"])])

    m = metrics(test[TARGET].values, pred)
    m_year = metrics_by_year(test, pred)
    print(f"  R²_log={m['R2_log']} | R²={m['R2']} | MAE={m['MAE']}")
    return pred, m, m_year


def ridge_ar(train, val, test, features):
    """Ridge autorregressivo com features realistas em log1p."""
    print("\n--- Ridge autorregressivo ---")
    available = [f for f in features if f in train.columns]
    print(f"  {len(available)} features")

    X_train = train[available].fillna(0).values
    y_train = np.log1p(train[TARGET].values)
    X_val = val[available].fillna(0).values
    y_val = np.log1p(val[TARGET].values)
    X_test = test[available].fillna(0).values

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    model.fit(X_train, y_train)
    print(f"  alpha escolhido: {model.alpha_}")

    val_pred_log = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred_log)
    print(f"  R²_log validação: {val_r2:.4f}")

    pred_log = model.predict(X_test)
    pred_log = np.clip(pred_log, 0, 15)
    pred_raw = np.expm1(pred_log)

    m = metrics(test[TARGET].values, pred_raw)
    m_year = metrics_by_year(test, pred_raw)
    print(f"  R²_log={m['R2_log']} | R²={m['R2']} | MAE={m['MAE']}")
    return pred_raw, m, m_year, model.alpha_


def controle_regiao_semana(train, val, test, include_clima=False):
    """XGBoost com cod_mesorregiao + week_of_year (+ clima opcional)."""
    label = "região+semana+clima" if include_clima else "região+semana"
    print(f"\n--- Controle {label} ---")

    features = ["cod_mesorregiao", "week_of_year_sin", "week_of_year_cos"]

    if include_clima:
        schema = pd.read_csv(SCHEMA_PATH)
        inmet = schema.loc[schema["category"] != "sinan", "feature"].tolist()
        features += [f for f in inmet if f in train.columns]

    available = [f for f in features if f in train.columns]
    print(f"  {len(available)} features")

    params = XGB_PARAMS.copy()
    if not include_clima:
        params["enable_categorical"] = True

    X_train = train[available].copy()
    X_val = val[available].copy()
    X_test = test[available].copy()

    if not include_clima:
        for df in [X_train, X_val, X_test]:
            df["cod_mesorregiao"] = df["cod_mesorregiao"].astype("category")

    y_train = np.log1p(train[TARGET].values)
    y_val = np.log1p(val[TARGET].values)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_log = model.predict(X_test)
    pred_raw = np.expm1(np.maximum(pred_log, 0))

    m = metrics(test[TARGET].values, pred_raw)
    m_year = metrics_by_year(test, pred_raw)
    print(f"  R²_log={m['R2_log']} | R²={m['R2']} | MAE={m['MAE']} | iter={model.best_iteration}")
    return pred_raw, m, m_year


def cobertura_climatica(train, val, test, realista_features):
    """Retreino com anos de alta cobertura e só linhas com clima."""
    print("\n--- Controle de cobertura climática ---")

    full = pd.read_parquet(
        DATA_DIR / "integrated_mesorregiao_v4.parquet",
        columns=ID_COLS + ["flag_fonte_inmet"]
    )

    results = {}

    # B cobertura alta (2008-2022)
    print("\n  B cobertura alta (2008-2022):")
    train_hi = train[train["ano"] >= 2008].copy()
    print(f"    Train: {len(train_hi):,} linhas")

    for clima_label, include_inmet in [("com_clima", True), ("sem_clima", False)]:
        feats = realista_features if include_inmet else [
            f for f in realista_features
            if not any(f.startswith(p) for p in [
                "rain_", "temp_", "humidity_", "pressure_", "wind_", "radiation_"
            ])
        ]
        available = [f for f in feats if f in train.columns]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            train_hi[available].values, np.log1p(train_hi[TARGET].values),
            eval_set=[(val[available].values, np.log1p(val[TARGET].values))],
            verbose=False
        )
        pred = np.expm1(np.maximum(model.predict(test[available].values), 0))
        m = metrics(test[TARGET].values, pred)
        print(f"    {clima_label}: R²_log={m['R2_log']} ({len(available)} feats, iter={model.best_iteration})")
        results[f"B_cobertura_alta_{clima_label}"] = {**m, "n_features": len(available), "n_train": len(train_hi)}

    # B só linhas com clima
    print("\n  B só linhas com clima:")
    train_flag = train.merge(full, on=ID_COLS, how="left")
    has_clima = train_flag["flag_fonte_inmet"] == True
    train_clim = train[has_clima.values].copy()
    print(f"    Train: {len(train_clim):,} linhas (de {len(train):,})")

    for clima_label, include_inmet in [("com_clima", True), ("sem_clima", False)]:
        feats = realista_features if include_inmet else [
            f for f in realista_features
            if not any(f.startswith(p) for p in [
                "rain_", "temp_", "humidity_", "pressure_", "wind_", "radiation_"
            ])
        ]
        available = [f for f in feats if f in train.columns]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            train_clim[available].values, np.log1p(train_clim[TARGET].values),
            eval_set=[(val[available].values, np.log1p(val[TARGET].values))],
            verbose=False
        )
        pred = np.expm1(np.maximum(model.predict(test[available].values), 0))
        m = metrics(test[TARGET].values, pred)
        print(f"    {clima_label}: R²_log={m['R2_log']} ({len(available)} feats, iter={model.best_iteration})")
        results[f"B_so_linhas_clima_{clima_label}"] = {**m, "n_features": len(available), "n_train": len(train_clim)}

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v4-59: Baselines Avançados e Controles")
    print("=" * 60)

    train, val, test = load_splits()
    add_notificacoes(train, val, test)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    realista_features = get_realista_features()
    report = {}

    # 1. Climatologia
    _, m_clim, m_clim_year = climatologia(train, test)
    report["climatologia"] = {"teste_agregado": m_clim, "teste_por_ano": m_clim_year}

    # 2. Ridge AR
    _, m_ridge, m_ridge_year, alpha = ridge_ar(train, val, test, realista_features)
    report["ridge_ar"] = {"teste_agregado": m_ridge, "teste_por_ano": m_ridge_year, "alpha": alpha}

    # 3. Controle região+semana
    _, m_rs, m_rs_year = controle_regiao_semana(train, val, test, include_clima=False)
    report["controle_regiao_semana"] = {"teste_agregado": m_rs, "teste_por_ano": m_rs_year}

    # 4. Controle região+semana+clima
    _, m_rsc, m_rsc_year = controle_regiao_semana(train, val, test, include_clima=True)
    report["controle_regiao_semana_clima"] = {"teste_agregado": m_rsc, "teste_por_ano": m_rsc_year}

    # 5. Cobertura climática
    report["cobertura_climatica"] = cobertura_climatica(train, val, test, realista_features)

    # Interpretação do controle só-clima
    so_clima_r2 = 0.3992  # do script 56
    regiao_semana_r2 = m_rs["R2_log"]
    if regiao_semana_r2 >= so_clima_r2:
        interp = f"Controle região+semana ({regiao_semana_r2}) >= só clima ({so_clima_r2}): clima é proxy de localização/sazonalidade"
    else:
        interp = f"Controle região+semana ({regiao_semana_r2}) < só clima ({so_clima_r2}): clima carrega sinal independente"
    report["interpretacao_so_clima"] = {
        "R2_log_so_clima_script56": so_clima_r2,
        "R2_log_controle_regiao_semana": regiao_semana_r2,
        "R2_log_controle_regiao_semana_clima": m_rsc["R2_log"],
        "conclusao": interp,
    }

    report["tempo_total_s"] = round(time.time() - t0, 1)

    out_path = DATA_DIR / "59_baselines_avancados_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

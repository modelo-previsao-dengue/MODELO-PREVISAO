#!/usr/bin/env python3
"""US-010: Hyperparameter tuning com Optuna."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY = BASE_DIR / "data" / "model_ready"
MODEL_DIR = BASE_DIR / "models" / "xgb_regression_tuned"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"

N_TRIALS = 20


def mape(y_true, y_pred):
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    y_pred = np.maximum(model.predict(X_val), 0)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando splits...")
    train = pd.read_parquet(MODEL_READY / "train.parquet")
    val = pd.read_parquet(MODEL_READY / "val.parquet")
    test = pd.read_parquet(MODEL_READY / "test.parquet")

    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    # Subsample for faster tuning
    TUNE_SAMPLE = 500_000
    if len(X_train) > TUNE_SAMPLE:
        idx = np.random.RandomState(42).choice(len(X_train), TUNE_SAMPLE, replace=False)
        X_tune, y_tune = X_train.iloc[idx], y_train.iloc[idx]
        print(f"  Subsample para tuning: {TUNE_SAMPLE:,} de {len(X_train):,}")
    else:
        X_tune, y_tune = X_train, y_train

    print(f"Iniciando Optuna com {N_TRIALS} trials...")
    study = optuna.create_study(direction="minimize", study_name="xgb_dengue_tuning")
    study.optimize(
        lambda trial: objective(trial, X_tune, y_tune, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nMelhores hiperparâmetros:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  Best val RMSE: {study.best_value:.4f}")

    # Retrain with best params (subsample if needed to avoid OOM on 16GB)
    RETRAIN_MAX = 2_000_000
    if len(X_train) > RETRAIN_MAX:
        idx_r = np.random.RandomState(42).choice(len(X_train), RETRAIN_MAX, replace=False)
        X_retrain, y_retrain = X_train.iloc[idx_r], y_train.iloc[idx_r]
        print(f"\nRetreinando com {RETRAIN_MAX:,} amostras (de {len(X_train):,})...")
    else:
        X_retrain, y_retrain = X_train, y_train
        print("\nRetreinando com melhores hiperparâmetros...")
    best["tree_method"] = "hist"
    best["random_state"] = 42
    best["n_jobs"] = -1
    best["early_stopping_rounds"] = 50

    model = xgb.XGBRegressor(**best)
    model.fit(X_retrain, y_retrain, eval_set=[(X_val, y_val)], verbose=0)

    y_pred = np.maximum(model.predict(X_test), 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape_val = mape(y_test.values, y_pred)

    metrics = {
        "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "R2": round(r2, 4), "MAPE": round(mape_val, 2),
        "best_params": best,
        "n_trials": N_TRIALS,
        "best_val_rmse": round(study.best_value, 4),
    }
    print(f"\n=== Métricas Tuned no Teste ===")
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape_val:.2f}%")

    # Load MVP metrics for comparison
    mvp_path = BASE_DIR / "models" / "xgb_regression_mvp" / "metrics.json"
    if mvp_path.exists():
        with open(mvp_path) as f:
            mvp = json.load(f)
        improvement = (mvp["RMSE"] - rmse) / mvp["RMSE"] * 100
        metrics["mvp_rmse"] = mvp["RMSE"]
        metrics["improvement_pct"] = round(improvement, 2)
        print(f"\n  vs MVP: RMSE {mvp['RMSE']:.4f} → {rmse:.4f} ({improvement:+.2f}%)")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_model(str(MODEL_DIR / "model.json"))

    # Save study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(MODEL_DIR / "optuna_trials.csv", index=False)

    # Hyperparameter importance
    try:
        from optuna.importance import get_param_importances
        importance = get_param_importances(study)
        imp_df = pd.DataFrame(list(importance.items()), columns=["param", "importance"])
        imp_df.to_csv(MODEL_DIR / "hyperparam_importance.csv", index=False)
        print(f"\nImportância dos hiperparâmetros:")
        print(imp_df.to_string(index=False))
    except Exception:
        pass

    print(f"\nResultados salvos em {MODEL_DIR}")


if __name__ == "__main__":
    main()

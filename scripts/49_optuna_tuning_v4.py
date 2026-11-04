#!/usr/bin/env python3
"""v4-49: Otimização de hiperparâmetros com Optuna — Var B INMET enriquecido.

Executa 30 trials de TPE sobre o espaço de busca definido no PRD,
usando R²_log no val (2023) como métrica de otimização.
Se o melhor trial supera a baseline em ≥ 0.005, re-treina e salva o modelo.
"""

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

TARGET = "notificacoes_t4"

N_TRIALS = 30
IMPROVEMENT_THRESHOLD = 0.005

BASELINE_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)


def get_enriched_features():
    schema = pd.read_csv(SCHEMA_PATH)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return sinan + inmet_bruto + inmet_enriched


def eval_r2_log(model, X, y_orig):
    y_log = np.log1p(y_orig)
    pred_log = model.predict(X)
    return r2_score(y_log, pred_log)


def objective(trial, train, val, features):
    params = dict(
        n_estimators=1000,
        max_depth=trial.suggest_int("max_depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model = xgb.XGBRegressor(**params)
    y_train_log = np.log1p(train[TARGET])
    y_val_log = np.log1p(val[TARGET])

    model.fit(
        train[features], y_train_log,
        eval_set=[(val[features], y_val_log)],
        verbose=0,
    )

    r2 = eval_r2_log(model, val[features], val[TARGET])
    return r2


def main():
    t0 = time.time()
    print("=" * 70)
    print("  v4-49: Optuna Tuning — Var B INMET enriquecido")
    print("=" * 70)

    features = get_enriched_features()
    print(f"  Features: {len(features)} (INMET enriquecido)")

    var_dir = DATA_DIR / "splits_b"
    train = pd.read_parquet(var_dir / "train.parquet")
    val = pd.read_parquet(var_dir / "val.parquet")
    test = pd.read_parquet(var_dir / "test.parquet")
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("\n--- Baseline (params atuais) ---")
    baseline_model = xgb.XGBRegressor(**BASELINE_PARAMS)
    baseline_model.fit(
        train[features], np.log1p(train[TARGET]),
        eval_set=[(val[features], np.log1p(val[TARGET]))],
        verbose=0,
    )
    baseline_val_r2 = eval_r2_log(baseline_model, val[features], val[TARGET])
    baseline_test_r2 = eval_r2_log(baseline_model, test[features], test[TARGET])
    print(f"  Baseline R²_log val: {baseline_val_r2:.4f}")
    print(f"  Baseline R²_log test: {baseline_test_r2:.4f}")

    print(f"\n--- Optuna: {N_TRIALS} trials ---")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, train, val, features),
        n_trials=N_TRIALS,
        timeout=7200,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  Best trial #{best.number}: R²_log val = {best.value:.4f}")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    improvement = best.value - baseline_val_r2
    print(f"\n  Melhoria sobre baseline: {improvement:+.4f} (threshold: {IMPROVEMENT_THRESHOLD})")

    best_test_r2 = None
    adopted = False

    if improvement >= IMPROVEMENT_THRESHOLD:
        print("\n  -> Melhoria significativa! Re-treinando com params otimizados...")
        tuned_params = dict(BASELINE_PARAMS)
        tuned_params.update(best.params)
        tuned_model = xgb.XGBRegressor(**tuned_params)
        tuned_model.fit(
            train[features], np.log1p(train[TARGET]),
            eval_set=[(val[features], np.log1p(val[TARGET]))],
            verbose=0,
        )
        best_test_r2 = eval_r2_log(tuned_model, test[features], test[TARGET])
        print(f"  Tuned R²_log test: {best_test_r2:.4f} (baseline: {baseline_test_r2:.4f})")

        test_improvement = best_test_r2 - baseline_test_r2
        if test_improvement > 0:
            out_path = MODEL_DIR / "model_B_inmet_enriquecido_tuned.ubj"
            tuned_model.save_model(str(out_path))
            print(f"  Modelo salvo: {out_path.name}")
            adopted = True
        else:
            print(f"  AVISO: Melhoria no val nao se manteve no test (Δtest={test_improvement:+.4f})")
            print(f"  Mantendo params originais.")
    else:
        print(f"\n  -> Melhoria insuficiente ({improvement:+.4f} < {IMPROVEMENT_THRESHOLD})")
        print(f"  Mantendo params originais.")

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": round(float(t.value), 6) if t.value is not None else None,
            "params": t.params,
            "state": str(t.state),
        })

    report = {
        "n_trials": N_TRIALS,
        "baseline_params": {k: v for k, v in BASELINE_PARAMS.items() if k != "early_stopping_rounds"},
        "baseline_val_R2_log": round(baseline_val_r2, 4),
        "baseline_test_R2_log": round(baseline_test_r2, 4),
        "best_trial_number": best.number,
        "best_val_R2_log": round(best.value, 4),
        "best_params": best.params,
        "improvement_val": round(improvement, 4),
        "threshold": IMPROVEMENT_THRESHOLD,
        "exceeds_threshold": improvement >= IMPROVEMENT_THRESHOLD,
        "best_test_R2_log": round(best_test_r2, 4) if best_test_r2 is not None else None,
        "adopted": adopted,
        "trials": trials_data,
        "tempo_total_s": round(time.time() - t0, 1),
    }

    report_path = DATA_DIR / "49_optuna_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report salvo: {report_path.name}")

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-49:")
    print(f"  Baseline val: {baseline_val_r2:.4f} | test: {baseline_test_r2:.4f}")
    print(f"  Best trial val: {best.value:.4f} (Δ={improvement:+.4f})")
    if best_test_r2 is not None:
        print(f"  Best trial test: {best_test_r2:.4f}")
    print(f"  Adotado: {'SIM' if adopted else 'NAO'}")
    print(f"  Tempo: {time.time() - t0:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

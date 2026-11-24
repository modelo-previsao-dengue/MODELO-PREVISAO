#!/usr/bin/env python3
"""v4-60: Métricas corrigidas e testes estatísticos robustos.

US-007: MASE por mesorregião (escala local)
US-008: Diebold-Mariano + bootstrap em blocos (mesorregião×ano)
US-009: Separação 2026 incompleto
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

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


def get_realista_features():
    schema = pd.read_csv(SCHEMA_PATH)
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    delayed = set(audit["delayed_consolidation"]["features"].keys())
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet = schema.loc[schema["category"] != "sinan", "feature"].tolist()
    realtime_sinan = [f for f in sinan if f not in delayed]
    return realtime_sinan + ["notificacoes"] + inmet


def metrics(y_true, y_pred):
    y_log = np.log1p(y_true)
    p_log = np.log1p(np.maximum(y_pred, 0))
    return {
        "R2_log": round(r2_score(y_log, p_log), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
    }


def train_realista(train, val, features):
    available = [f for f in features if f in train.columns]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[available].values, np.log1p(train[TARGET].values),
        eval_set=[(val[available].values, np.log1p(val[TARGET].values))],
        verbose=False
    )
    return model, available


def predict(model, df, features):
    pred_log = model.predict(df[features].values)
    return np.expm1(np.maximum(pred_log, 0))


def mase_per_mesoregion_media(test, pred, train):
    """MASE por mesorregião: MAE_modelo / MAE_media_treino."""
    print("\n--- MASE vs média do treino (por mesorregião) ---")
    test_c = test.copy()
    test_c["pred"] = pred

    train_means = train.groupby("cod_mesorregiao")[TARGET].mean()

    results = {}
    for meso, g in test_c.groupby("cod_mesorregiao"):
        y = g[TARGET].values
        p = g["pred"].values
        mae_model = np.mean(np.abs(y - p))

        naive_mean = train_means.get(meso, y.mean())
        mae_naive = np.mean(np.abs(y - naive_mean))

        mase = mae_model / mae_naive if mae_naive > 0 else np.nan
        results[int(meso)] = {
            "MASE": round(float(mase), 3) if not np.isnan(mase) else None,
            "MAE_modelo": round(float(mae_model), 1),
            "MAE_naive": round(float(mae_naive), 1),
            "n_obs": len(g),
        }

    mase_vals = [v["MASE"] for v in results.values() if v["MASE"] is not None]
    summary = {
        "mediana": round(float(np.median(mase_vals)), 3),
        "media": round(float(np.mean(mase_vals)), 3),
        "p25": round(float(np.percentile(mase_vals, 25)), 3),
        "p75": round(float(np.percentile(mase_vals, 75)), 3),
        "pct_abaixo_1": round(float(np.mean(np.array(mase_vals) < 1.0) * 100), 1),
        "n_mesorregiao": len(mase_vals),
    }
    print(f"  Mediana MASE_media={summary['mediana']} | {summary['pct_abaixo_1']}% abaixo de 1")
    return results, summary


def mase_vs_persistencia(test, pred):
    """MASE por mesorregião: MAE_modelo / MAE_persistência (denominador = persistência)."""
    print("\n--- MASE vs persistência (por mesorregião) ---")
    test_c = test.copy()
    test_c["pred"] = pred

    results = {}
    acima_1 = []
    for meso, g in test_c.groupby("cod_mesorregiao"):
        y = g[TARGET].values
        p = g["pred"].values
        p_pers = g["notificacoes"].values

        mae_model = np.mean(np.abs(y - p))
        mae_pers = np.mean(np.abs(y - p_pers))

        mase = mae_model / mae_pers if mae_pers > 0 else np.nan
        results[int(meso)] = {
            "MASE_vs_pers": round(float(mase), 3) if not np.isnan(mase) else None,
            "MAE_modelo": round(float(mae_model), 1),
            "MAE_persistencia": round(float(mae_pers), 1),
            "n_obs": len(g),
        }
        if not np.isnan(mase) and mase >= 1.0:
            acima_1.append(int(meso))

    mase_vals = [v["MASE_vs_pers"] for v in results.values() if v["MASE_vs_pers"] is not None]
    summary = {
        "mediana": round(float(np.median(mase_vals)), 3),
        "media": round(float(np.mean(mase_vals)), 3),
        "p25": round(float(np.percentile(mase_vals, 25)), 3),
        "p75": round(float(np.percentile(mase_vals, 75)), 3),
        "pct_abaixo_1": round(float(np.mean(np.array(mase_vals) < 1.0) * 100), 1),
        "n_abaixo_1": int(np.sum(np.array(mase_vals) < 1.0)),
        "n_mesorregiao": len(mase_vals),
        "meso_acima_1": acima_1,
    }
    print(f"  Mediana MASE_pers={summary['mediana']} | {summary['n_abaixo_1']}/{summary['n_mesorregiao']} abaixo de 1 ({summary['pct_abaixo_1']}%)")
    if acima_1:
        print(f"  Mesorregiões com MASE_pers >= 1: {acima_1}")
    return results, summary


def diebold_mariano(test, pred_model, pred_baseline, h=4):
    """Teste Diebold-Mariano para previsões h-passos à frente."""
    print("\n--- Diebold-Mariano ---")
    e_model = test[TARGET].values - pred_model
    e_baseline = test[TARGET].values - pred_baseline

    d = e_baseline**2 - e_model**2

    n = len(d)
    d_bar = np.mean(d)
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n

    if var_d <= 0:
        print("  AVISO: variância negativa — usando HAC robusto")
        var_d = gamma[0] / n

    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    result = {
        "DM_stat": round(float(dm_stat), 4),
        "p_value": float(p_value),
        "p_value_str": f"{p_value:.2e}",
        "d_bar": round(float(d_bar), 2),
        "h": h,
        "n": n,
        "modelo_melhor": dm_stat > 0,
    }
    print(f"  DM={result['DM_stat']} | p={result['p_value_str']} | modelo melhor: {result['modelo_melhor']}")
    return result


def diebold_mariano_log(test, pred_model, pred_baseline, h=4):
    """DM em escala log1p."""
    e_model = np.log1p(test[TARGET].values) - np.log1p(np.maximum(pred_model, 0))
    e_baseline = np.log1p(test[TARGET].values) - np.log1p(np.maximum(pred_baseline, 0))

    d = e_baseline**2 - e_model**2
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
        "p_value_str": f"{p_value:.2e}",
        "modelo_melhor": dm_stat > 0,
    }


def block_bootstrap(test, pred_model, pred_baseline, n_boot=5000, seed=42):
    """Bootstrap em blocos (mesorregião×ano) para IC de R²_log."""
    print(f"\n--- Bootstrap em blocos ({n_boot} iterações) ---")
    rng = np.random.RandomState(seed)

    test_c = test.copy()
    test_c["pred_model"] = pred_model
    test_c["pred_baseline"] = pred_baseline
    test_c["block"] = test_c["cod_mesorregiao"].astype(str) + "_" + test_c["ano"].astype(str)

    blocks = test_c["block"].unique()
    block_data = {b: test_c[test_c["block"] == b] for b in blocks}
    n_blocks = len(blocks)

    deltas = np.zeros(n_boot)
    r2_model_boot = np.zeros(n_boot)
    r2_baseline_boot = np.zeros(n_boot)

    for i in range(n_boot):
        sampled = rng.choice(blocks, size=n_blocks, replace=True)
        rows = pd.concat([block_data[b] for b in sampled], ignore_index=True)

        y = rows[TARGET].values
        y_log = np.log1p(y)
        pm_log = np.log1p(np.maximum(rows["pred_model"].values, 0))
        pb_log = np.log1p(np.maximum(rows["pred_baseline"].values, 0))

        r2_m = r2_score(y_log, pm_log)
        r2_b = r2_score(y_log, pb_log)
        r2_model_boot[i] = r2_m
        r2_baseline_boot[i] = r2_b
        deltas[i] = r2_m - r2_b

    result = {
        "n_boot": n_boot,
        "n_blocks": n_blocks,
        "delta_R2_log": {
            "mean": round(float(np.mean(deltas)), 4),
            "median": round(float(np.median(deltas)), 4),
            "ci_95": [round(float(np.percentile(deltas, 2.5)), 4),
                      round(float(np.percentile(deltas, 97.5)), 4)],
            "pct_positivo": round(float(np.mean(deltas > 0) * 100), 1),
        },
        "R2_log_modelo": {
            "mean": round(float(np.mean(r2_model_boot)), 4),
            "ci_95": [round(float(np.percentile(r2_model_boot, 2.5)), 4),
                      round(float(np.percentile(r2_model_boot, 97.5)), 4)],
        },
        "R2_log_baseline": {
            "mean": round(float(np.mean(r2_baseline_boot)), 4),
            "ci_95": [round(float(np.percentile(r2_baseline_boot, 2.5)), 4),
                      round(float(np.percentile(r2_baseline_boot, 97.5)), 4)],
        },
    }
    d = result["delta_R2_log"]
    print(f"  Δ R²_log: {d['mean']} [{d['ci_95'][0]}, {d['ci_95'][1]}] | {d['pct_positivo']}% positivo")
    return result


def separar_2026(test, pred):
    """Métricas separando 2024, 2025, 2026 (incompleto)."""
    print("\n--- Separação por ano (2026 incompleto) ---")
    test_c = test.copy()
    test_c["pred"] = pred
    result = {}

    for year in sorted(test_c["ano"].unique()):
        g = test_c[test_c["ano"] == year]
        semanas = sorted(g["semana_epidemiologica"].unique())
        m = metrics(g[TARGET].values, g["pred"].values)
        result[int(year)] = {
            **m,
            "n_obs": len(g),
            "semanas": f"{min(semanas)}-{max(semanas)}",
            "n_semanas": len(semanas),
            "n_meso": g["cod_mesorregiao"].nunique(),
        }
        flag = " (INCOMPLETO)" if len(semanas) < 50 else ""
        print(f"  {year}: R²_log={m['R2_log']} | n={len(g):,} | sem={len(semanas)}{flag}")

    return result


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v4-60: Métricas Corrigidas e Testes Estatísticos")
    print("=" * 60)

    train, val, test = load_splits()
    add_notificacoes(train, val, test)
    features = get_realista_features()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    print("\nTreinando modelo realista...")
    model, used_features = train_realista(train, val, features)
    pred = predict(model, test, used_features)
    pred_pers = test["notificacoes"].values.copy()

    m_global = metrics(test[TARGET].values, pred)
    print(f"Realista global: R²_log={m_global['R2_log']} | MAE={m_global['MAE']}")

    report = {"realista_global": m_global}

    # US-007: MASE por mesorregião
    mase_media_detail, mase_media_summary = mase_per_mesoregion_media(test, pred, train)
    mase_pers_detail, mase_pers_summary = mase_vs_persistencia(test, pred)
    report["mase"] = {
        "vs_persistencia": {"summary": mase_pers_summary, "por_mesorregiao": mase_pers_detail},
        "vs_media_treino": {"summary": mase_media_summary, "por_mesorregiao": mase_media_detail},
    }

    # US-008: Diebold-Mariano
    dm_raw = diebold_mariano(test, pred, pred_pers, h=4)
    dm_log = diebold_mariano_log(test, pred, pred_pers, h=4)
    print(f"  DM log: stat={dm_log['DM_stat']} | p={dm_log['p_value_str']}")

    report["diebold_mariano"] = {
        "escala_original": dm_raw,
        "escala_log": dm_log,
        "nota": "h=4 para previsão 4 semanas à frente. DM>0 = modelo melhor que persistência."
    }

    # Bootstrap em blocos
    boot = block_bootstrap(test, pred, pred_pers, n_boot=5000)
    report["bootstrap_blocos"] = boot

    # US-009: Separação 2026
    por_ano = separar_2026(test, pred)
    report["por_ano"] = por_ano

    por_ano_pers = {}
    test_c = test.copy()
    test_c["pred"] = pred_pers
    for year in sorted(test_c["ano"].unique()):
        g = test_c[test_c["ano"] == year]
        por_ano_pers[int(year)] = metrics(g[TARGET].values, g["pred"].values)
    report["por_ano_persistencia"] = por_ano_pers

    # Métricas excluindo 2026
    test_no26 = test[test["ano"] <= 2025]
    pred_no26 = pred[test["ano"].values <= 2025]
    pred_pers_no26 = pred_pers[test["ano"].values <= 2025]
    report["sem_2026"] = {
        "realista": metrics(test_no26[TARGET].values, pred_no26),
        "persistencia": metrics(test_no26[TARGET].values, pred_pers_no26),
        "n_obs": len(test_no26),
    }
    print(f"\n  Sem 2026: R²_log={report['sem_2026']['realista']['R2_log']} (n={len(test_no26):,})")

    report["tempo_total_s"] = round(time.time() - t0, 1)

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

    out_path = DATA_DIR / "60_metricas_corrigidas_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v4-46: Regressão v4 — 3 variantes × 3 sub-modelos = 9 modelos.

Baseado em scripts/27_train_regression_v3.py.
Diferenças:
  - 3 variantes (A baseline 4yr, B full 22yr, C full + decay weight).
  - 3 sub-modelos por variante (SINAN-only, INMET bruto, INMET enriquecido).
  - Bootstrap IC 95% (n=1000) para R²_log.
  - Métricas por ano de teste e por flag_fonte_inmet.
  - Sample weight exponencial para variante C.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

VARIANTES = ["a", "b", "c"]
SUB_MODELS = ["sinan_only", "inmet_bruto", "inmet_enriquecido"]

N_BOOTSTRAP = 1000
RNG_BOOTSTRAP = np.random.RandomState(42)

UF_MAP = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}

REGIAO_MAP = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte",
    "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
}


def categorize_features(schema_path):
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return sinan, inmet_bruto, inmet_enriched


def get_feature_sets(schema_path):
    sinan, inmet_bruto, inmet_enriched = categorize_features(schema_path)
    return {
        "sinan_only": sinan,
        "inmet_bruto": sinan + inmet_bruto,
        "inmet_enriquecido": sinan + inmet_bruto + inmet_enriched,
    }


def compute_sample_weights(train_df, var_cfg):
    lam = var_cfg.get("sample_weight_lambda")
    ref = var_cfg.get("sample_weight_ref_year")
    if lam is None or ref is None:
        return None
    weights = np.exp(-lam * (ref - train_df["ano"].values.astype(float)))
    return weights


def bootstrap_r2_log(y_true_log, y_pred_log, n=N_BOOTSTRAP, rng=RNG_BOOTSTRAP):
    scores = np.empty(n)
    idx = np.arange(len(y_true_log))
    for i in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        scores[i] = r2_score(y_true_log[sample], y_pred_log[sample])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def train_and_eval(train, val, test, features, label, sample_weight=None):
    t0 = time.time()
    print(f"\n  [{label}] Treinando ({len(features)} features, {len(train):,} train rows)...")

    y_train = np.log1p(train[TARGET])
    y_val = np.log1p(val[TARGET])
    y_test_log = np.log1p(test[TARGET]).values
    y_test_orig = test[TARGET].values

    model = xgb.XGBRegressor(**XGB_PARAMS)

    fit_kwargs = dict(
        eval_set=[(val[features], y_val)],
        verbose=0,
    )
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(train[features], y_train, **fit_kwargs)

    pred_log = model.predict(test[features])
    pred_orig = np.maximum(np.expm1(pred_log), 0)

    rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
    mae = mean_absolute_error(y_test_orig, pred_orig)
    r2_orig = r2_score(y_test_orig, pred_orig)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, pred_log))
    r2_log = r2_score(y_test_log, pred_log)

    boot_mean, boot_lo, boot_hi = bootstrap_r2_log(y_test_log, pred_log)

    elapsed = time.time() - t0

    metrics = {
        "label": label,
        "n_features": len(features),
        "n_train": len(train),
        "R2_log": round(float(r2_log), 4),
        "R2_log_bootstrap_mean": round(boot_mean, 4),
        "R2_log_CI95_lo": round(boot_lo, 4),
        "R2_log_CI95_hi": round(boot_hi, 4),
        "R2": round(float(r2_orig), 4),
        "RMSE": round(float(rmse), 2),
        "RMSE_log": round(float(rmse_log), 4),
        "MAE": round(float(mae), 4),
        "best_iteration": int(model.best_iteration),
        "train_time_s": round(elapsed, 1),
    }

    print(f"    R²_log={r2_log:.4f} [{boot_lo:.4f}, {boot_hi:.4f}]  RMSE={rmse:.2f}  MAE={mae:.4f}  ({elapsed:.1f}s)")

    errors_sq = (y_test_log - pred_log) ** 2

    return model, metrics, pred_log, errors_sq


def eval_by_year(test, pred_log, label):
    rows = []
    test_tmp = test.copy()
    test_tmp["pred_log"] = pred_log
    test_tmp["y_log"] = np.log1p(test_tmp[TARGET])

    for year in sorted(test_tmp["ano"].unique()):
        sub = test_tmp[test_tmp["ano"] == year]
        if len(sub) < 10:
            continue
        r2 = r2_score(sub["y_log"], sub["pred_log"])
        rmse = np.sqrt(mean_squared_error(sub[TARGET], np.maximum(np.expm1(sub["pred_log"]), 0)))
        rows.append({
            "modelo": label,
            "ano": int(year),
            "R2_log": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "n": len(sub),
        })
    return rows


def eval_by_flag_inmet(test, pred_log, flag_series, label):
    test_tmp = test.copy()
    test_tmp["pred_log"] = pred_log
    test_tmp["y_log"] = np.log1p(test_tmp[TARGET])
    test_tmp["flag_fonte_inmet"] = flag_series.values

    rows = []
    for flag_val in [0, 1]:
        sub = test_tmp[test_tmp["flag_fonte_inmet"] == flag_val]
        if len(sub) < 10:
            continue
        r2 = r2_score(sub["y_log"], sub["pred_log"])
        rmse = np.sqrt(mean_squared_error(sub[TARGET], np.maximum(np.expm1(sub["pred_log"]), 0)))
        rows.append({
            "modelo": label,
            "flag_fonte_inmet": int(flag_val),
            "R2_log": round(float(r2), 4),
            "RMSE": round(float(rmse), 2),
            "n": len(sub),
        })
    return rows


def eval_by_mesorregiao(test, pred_log, label):
    test_tmp = test.copy()
    test_tmp["pred_log"] = pred_log
    test_tmp["y_log"] = np.log1p(test_tmp[TARGET])

    rows = []
    for cod, sub in test_tmp.groupby("cod_mesorregiao"):
        if len(sub) < 10:
            continue
        r2 = r2_score(sub["y_log"], sub["pred_log"])
        rows.append({"cod_mesorregiao": int(cod), "modelo": label, "R2_log": round(float(r2), 4), "n": len(sub)})
    return rows


def eval_by_uf(test, pred_log, label):
    test_tmp = test.copy()
    test_tmp["pred_log"] = pred_log
    test_tmp["y_log"] = np.log1p(test_tmp[TARGET])
    test_tmp["uf_code"] = test_tmp["cod_mesorregiao"] // 100
    test_tmp["uf"] = test_tmp["uf_code"].map(UF_MAP)
    test_tmp["regiao"] = test_tmp["uf"].map(REGIAO_MAP)

    rows = []
    for uf in sorted(test_tmp["uf"].dropna().unique()):
        sub = test_tmp[test_tmp["uf"] == uf]
        if len(sub) < 50:
            continue
        r2 = r2_score(sub["y_log"], sub["pred_log"])
        rmse = np.sqrt(mean_squared_error(sub[TARGET], np.maximum(np.expm1(sub["pred_log"]), 0)))
        rows.append({
            "uf": uf, "regiao": REGIAO_MAP.get(uf, "?"), "modelo": label,
            "R2_log": round(float(r2), 4), "RMSE": round(float(rmse), 2), "n": len(sub),
        })
    return rows


def plot_variante_comparison(report, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    variantes = ["A", "B", "C"]
    colors_var = {"A": "#1976D2", "B": "#FF9800", "C": "#E53935"}

    for ax, sub_model in zip(axes, SUB_MODELS):
        sub_label = {"sinan_only": "SINAN-only", "inmet_bruto": "INMET bruto",
                     "inmet_enriquecido": "INMET enriquecido"}[sub_model]
        x = np.arange(len(variantes))
        vals, errs_lo, errs_hi = [], [], []
        for var in variantes:
            key = f"{var}_{sub_model}"
            m = report["modelos"][key]
            vals.append(m["R2_log"])
            errs_lo.append(m["R2_log"] - m["R2_log_CI95_lo"])
            errs_hi.append(m["R2_log_CI95_hi"] - m["R2_log"])

        bars = ax.bar(x, vals, color=[colors_var[v] for v in variantes], width=0.6,
                      yerr=[errs_lo, errs_hi], capsize=4, error_kw={"linewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([f"Var {v}" for v in variantes], fontsize=10)
        ax.set_title(sub_label, fontsize=11)
        ax.set_ylabel("R²_log")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("R²_log por Variante e Sub-modelo (IC 95%)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_by_year(year_metrics, out_path):
    df = pd.DataFrame(year_metrics)
    if df.empty:
        return

    enriq = df[df["modelo"].str.contains("enriquecido")]
    if enriq.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted(enriq["ano"].unique())
    variantes = sorted(enriq["modelo"].unique())
    colors = ["#1976D2", "#FF9800", "#E53935"]
    width = 0.25

    for i, var in enumerate(variantes):
        sub = enriq[enriq["modelo"] == var]
        vals = [sub[sub["ano"] == y]["R2_log"].values[0] if len(sub[sub["ano"] == y]) > 0 else 0 for y in years]
        x = np.arange(len(years)) + i * width
        ax.bar(x, vals, width=width, label=var.split(":")[0].strip(), color=colors[i % len(colors)])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(np.arange(len(years)) + width)
    ax.set_xticklabels(years)
    ax.set_ylabel("R²_log")
    ax.set_title("R²_log por Ano de Teste (INMET enriquecido)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  v4-46: Regressão v4 — 3 Variantes × 3 Sub-modelos = 9 Modelos")
    print("=" * 70)

    feature_sets = get_feature_sets(SCHEMA_PATH)
    for name, feats in feature_sets.items():
        print(f"  {name}: {len(feats)} features")

    integrated = pd.read_parquet(
        DATA_DIR / "integrated_mesorregiao_full.parquet",
        columns=ID_COLS + ["flag_fonte_inmet"],
    )

    all_metrics = {}
    all_year_metrics = []
    all_flag_metrics = []
    all_meso_metrics = []
    all_uf_metrics = []
    all_errors = {}

    for var in VARIANTES:
        var_upper = var.upper()
        var_cfg = recorte_config.load_v4(f"variante_{var}")
        var_dir = DATA_DIR / f"splits_{var}"

        print(f"\n{'=' * 70}")
        print(f"  VARIANTE {var_upper}: train={var_cfg['train'][0]}-{var_cfg['train'][-1]} ({len(var_cfg['train'])} anos)")
        print(f"{'=' * 70}")

        train = pd.read_parquet(var_dir / "train.parquet")
        val = pd.read_parquet(var_dir / "val.parquet")
        test = pd.read_parquet(var_dir / "test.parquet")
        print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        sw = compute_sample_weights(train, var_cfg)
        if sw is not None:
            print(f"  Sample weights: min={sw.min():.4f}, max={sw.max():.4f}, mean={sw.mean():.4f}")
        else:
            print(f"  Sample weights: uniform (1.0)")

        test_flag = integrated.merge(
            test[ID_COLS], on=ID_COLS, how="inner"
        )["flag_fonte_inmet"]

        for sub_name in SUB_MODELS:
            feats = feature_sets[sub_name]
            sub_label_short = {"sinan_only": "SINAN-only", "inmet_bruto": "INMET bruto",
                               "inmet_enriquecido": "INMET enriquecido"}[sub_name]
            label = f"Var {var_upper}: {sub_label_short}"

            model, metrics, pred_log, errors_sq = train_and_eval(
                train, val, test, feats, label, sample_weight=sw,
            )

            key = f"{var_upper}_{sub_name}"
            all_metrics[key] = metrics
            all_errors[key] = errors_sq

            model.save_model(str(MODEL_DIR / f"model_{key}.ubj"))

            all_year_metrics.extend(eval_by_year(test, pred_log, label))
            all_flag_metrics.extend(eval_by_flag_inmet(test, pred_log, test_flag, label))
            all_meso_metrics.extend(eval_by_mesorregiao(test, pred_log, label))
            all_uf_metrics.extend(eval_by_uf(test, pred_log, label))

    print(f"\n{'=' * 70}")
    print("  TESTES T PAREADOS (erros quadráticos log)")
    print(f"{'=' * 70}")
    ttests = {}
    comparisons = [
        ("A_vs_B", "A_inmet_enriquecido", "B_inmet_enriquecido"),
        ("A_vs_C", "A_inmet_enriquecido", "C_inmet_enriquecido"),
        ("B_vs_C", "B_inmet_enriquecido", "C_inmet_enriquecido"),
    ]
    for name, k1, k2 in comparisons:
        t_stat, p_val = ttest_rel(all_errors[k1], all_errors[k2])
        ttests[name] = {"t_stat": round(float(t_stat), 4), "p_value": float(p_val)}
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {name}: t={t_stat:.4f}, p={p_val:.2e} ({sig})")

    print(f"\n{'=' * 70}")
    print("  BOOTSTRAP IC 95% — Diferenças R²_log")
    print(f"{'=' * 70}")
    boot_diffs = {}
    test_any = pd.read_parquet(DATA_DIR / "splits_a" / "test.parquet")
    y_test_log = np.log1p(test_any[TARGET]).values
    for name, k1, k2 in comparisons:
        pred1 = np.expm1(all_metrics[k1]["R2_log"])  # placeholder, re-derive from model
        # Use stored errors to compute bootstrap difference
        e1 = all_errors[k1]
        e2 = all_errors[k2]
        diffs = np.empty(N_BOOTSTRAP)
        idx = np.arange(len(e1))
        rng = np.random.RandomState(42)
        for i in range(N_BOOTSTRAP):
            s = rng.choice(idx, size=len(idx), replace=True)
            r2_1 = 1 - e1[s].sum() / ((y_test_log[s] - y_test_log[s].mean()) ** 2).sum()
            r2_2 = 1 - e2[s].sum() / ((y_test_log[s] - y_test_log[s].mean()) ** 2).sum()
            diffs[i] = r2_2 - r2_1
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        mean_diff = float(np.mean(diffs))
        boot_diffs[name] = {
            "mean_diff": round(mean_diff, 4),
            "CI95_lo": round(float(lo), 4),
            "CI95_hi": round(float(hi), 4),
        }
        print(f"  {name}: ΔR²_log = {mean_diff:.4f} [{lo:.4f}, {hi:.4f}]")

    print("\nGerando figuras...")
    report_data = {"modelos": all_metrics}
    plot_variante_comparison(report_data, FIG_DIR / "fig_v4_variante_comparison.png")
    plot_by_year(all_year_metrics, FIG_DIR / "fig_v4_r2_by_year.png")

    pd.DataFrame(all_year_metrics).to_csv(MODEL_DIR / "metrics_por_ano_v4.csv", index=False)
    pd.DataFrame(all_flag_metrics).to_csv(MODEL_DIR / "metrics_por_flag_inmet_v4.csv", index=False)
    pd.DataFrame(all_meso_metrics).to_csv(MODEL_DIR / "metrics_por_mesorregiao_v4.csv", index=False)
    pd.DataFrame(all_uf_metrics).to_csv(MODEL_DIR / "metrics_por_uf_v4.csv", index=False)

    enriq_a = all_metrics["A_inmet_enriquecido"]
    enriq_b = all_metrics["B_inmet_enriquecido"]
    enriq_c = all_metrics["C_inmet_enriquecido"]

    report = {
        "modelos": all_metrics,
        "deltas_enriquecido": {
            "B_vs_A": round(enriq_b["R2_log"] - enriq_a["R2_log"], 4),
            "C_vs_A": round(enriq_c["R2_log"] - enriq_a["R2_log"], 4),
            "C_vs_B": round(enriq_c["R2_log"] - enriq_b["R2_log"], 4),
        },
        "testes_t": ttests,
        "bootstrap_diffs": boot_diffs,
        "metricas_por_ano": all_year_metrics,
        "metricas_por_flag_inmet": all_flag_metrics,
        "conclusao": {
            "serie_completa_ajuda": enriq_b["R2_log"] > enriq_a["R2_log"],
            "peso_recencia_ajuda": enriq_c["R2_log"] > enriq_b["R2_log"],
            "melhor_variante": max(
                [("A", enriq_a["R2_log"]), ("B", enriq_b["R2_log"]), ("C", enriq_c["R2_log"])],
                key=lambda x: x[1],
            )[0],
        },
        "tempo_total_s": round(time.time() - t_total, 1),
    }
    with open(DATA_DIR / "46_regression_v4_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-46 (INMET enriquecido):")
    print(f"  Var A (baseline):      R²_log = {enriq_a['R2_log']:.4f} [{enriq_a['R2_log_CI95_lo']:.4f}, {enriq_a['R2_log_CI95_hi']:.4f}]")
    print(f"  Var B (série completa):R²_log = {enriq_b['R2_log']:.4f} [{enriq_b['R2_log_CI95_lo']:.4f}, {enriq_b['R2_log_CI95_hi']:.4f}]")
    print(f"  Var C (série + peso):  R²_log = {enriq_c['R2_log']:.4f} [{enriq_c['R2_log_CI95_lo']:.4f}, {enriq_c['R2_log_CI95_hi']:.4f}]")
    print(f"  Δ B vs A: {enriq_b['R2_log'] - enriq_a['R2_log']:+.4f}")
    print(f"  Δ C vs A: {enriq_c['R2_log'] - enriq_a['R2_log']:+.4f}")
    print(f"  Melhor variante: {report['conclusao']['melhor_variante']}")
    print(f"  Tempo total: {time.time() - t_total:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

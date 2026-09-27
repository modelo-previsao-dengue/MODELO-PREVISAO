#!/usr/bin/env python3
"""US-213: Baselines ingenuos no mesmo teste da v3 mesorregional.

A v3 comparava tres XGBoost entre si (A/B/C) e nunca contra a alternativa
trivial. Previsao de dengue tem autocorrelacao alta, e sem esta linha nao da
para dizer quanto do R2 o modelo aprendeu e quanto "repetir o valor de hoje"
ja entrega (ver scripts/baselines_naive.py, US-007 do retreino).

Tres baselines para notificacoes_t4, no mesmo test_v3.parquet do script 27:

    persistencia     casos(t+4) = casos(t)
    media_movel_4    casos(t+4) = media de casos(t), (t-1), (t-2), (t-3)
    sazonal          casos(t+4) = casos(t+4-52), a mesma semana do ano anterior
                     (shift de 48 semanas na serie densa da mesorregiao; em
                     anos de 53 semanas fica a uma semana do exato)

Metricas identicas as do script 27 (R2_log e RMSE_log em log1p, R2/RMSE/MAE
na escala original). A classificacao aplica os limiares de risco do treino
(p50/p75/p90 por mesorregiao, como no script 26) a previsao de cada baseline
e compara com risco_surto_t4, com as mesmas metricas do script 29.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
OUT_DIR = BASE_DIR / "models" / "baselines_v3"

GROUP_KEY = "cod_mesorregiao"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_NAMES = ["baixo", "moderado", "alto", "muito_alto"]


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    y_true_log, y_pred_log = np.log1p(y_true), np.log1p(y_pred)
    return {
        "R2_log": round(float(r2_score(y_true_log, y_pred_log)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "RMSE_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def risk_from_counts(counts, meso, thresholds):
    """Mesma regra de compute_risk_class() do script 26, sobre uma previsao."""
    t = thresholds.reindex(meso.values)
    counts = np.asarray(counts, dtype=float)
    risk = np.zeros(len(counts), dtype=int)
    risk[counts > t["p50"].values] = 1
    risk[counts > t["p75"].values] = 2
    risk[counts > t["p90"].values] = 3
    return risk


def classification_metrics(y_true, y_pred):
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "accuracy": round(float((np.asarray(y_true) == y_pred).mean()), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_per_class": {CLASS_NAMES[i]: round(float(f1_per[i]), 4) for i in range(4)},
    }


def markdown_table(df):
    """Tabela Markdown sem depender do pacote tabulate."""
    linhas = ["| " + " | ".join(df.columns) + " |",
              "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, row in df.iterrows():
        linhas.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(linhas) + "\n"


def seasonal_lookup(test):
    """casos(t+4-52) por mesorregiao, lido da serie densa com o warm-up."""
    full = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_v3.parquet",
                           columns=ID_COLS + ["notificacoes"])
    full = full.sort_values(ID_COLS).reset_index(drop=True)
    full["sazonal"] = full.groupby(GROUP_KEY)["notificacoes"].shift(48)
    merged = test[ID_COLS].merge(full[ID_COLS + ["sazonal"]], on=ID_COLS, how="left")
    return merged["sazonal"].values


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-213: Baselines ingenuos — teste v3 mesorregional")
    print("=" * 60)

    train = pd.read_parquet(DATA_DIR / "train_v3.parquet",
                            columns=ID_COLS + [TARGET])
    test = pd.read_parquet(DATA_DIR / "test_v3.parquet")
    test = test.sort_values(ID_COLS).reset_index(drop=True)
    print(f"  Teste: {len(test):,} linhas, anos {sorted(test['ano'].unique().tolist())}")

    thresholds = train.groupby(GROUP_KEY)[TARGET].quantile([0.50, 0.75, 0.90]).unstack()
    thresholds.columns = ["p50", "p75", "p90"]

    previsoes = {
        "persistencia": test["notificacoes"].values,
        "media_movel_4": test[["notificacoes", "notificacoes_lag_1w",
                               "notificacoes_lag_2w", "notificacoes_lag_3w"]].mean(axis=1).values,
        "sazonal": seasonal_lookup(test),
    }

    y_true = test[TARGET].values
    y_cls = test[CLASS_TARGET].values
    resultados = {}
    for nome, pred in previsoes.items():
        ok = ~np.isnan(pred)
        reg = regression_metrics(y_true[ok], pred[ok])
        cls = classification_metrics(
            y_cls[ok], risk_from_counts(pred[ok], test.loc[ok, GROUP_KEY], thresholds))
        resultados[nome] = {"n": int(ok.sum()), "regressao": reg, "classificacao": cls}
        print(f"  {nome:14s} n={ok.sum():,}  R2_log={reg['R2_log']:.4f}  R2={reg['R2']:.4f}"
              f"  MAE={reg['MAE']:.2f}  F1_macro={cls['f1_macro']:.4f}")

    # Tabela lado a lado com os XGBoost do mesmo teste (scripts 27 e 29).
    linhas = [{"modelo": f"baseline: {n}", "n": r["n"], **r["regressao"],
               "f1_macro": r["classificacao"]["f1_macro"]} for n, r in resultados.items()]
    reg_v3 = json.loads((DATA_DIR / "27_regression_v3_report.json").read_text(encoding="utf-8"))
    cls_path = DATA_DIR / "29_classification_v3_report.json"
    cls_v3 = json.loads(cls_path.read_text(encoding="utf-8")) if cls_path.exists() else {}
    for chave, m in reg_v3["modelos"].items():
        f1 = cls_v3.get("modelos", {}).get(chave, {}).get("f1_macro")
        linhas.append({"modelo": f"XGBoost {m['label']}", "n": len(test),
                       **{k: m[k] for k in ("R2_log", "R2", "RMSE", "RMSE_log", "MAE")},
                       "f1_macro": f1})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(OUT_DIR / "comparison.csv", index=False)
    (OUT_DIR / "comparison.md").write_text(markdown_table(tabela), encoding="utf-8")

    report = {
        "teste_anos": sorted(int(a) for a in test["ano"].unique()),
        "n_teste": len(test),
        "baselines": resultados,
        "nota_sazonal": "shift de 48 semanas na serie densa; em anos de 53 semanas fica a uma semana do exato",
    }
    with open(DATA_DIR / "33_baselines_v3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + tabela.to_string(index=False))
    print(f"\n  Salvo: {OUT_DIR / 'comparison.csv'} e 33_baselines_v3_report.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v4-51: Curva de aprendizado — R²_log vs tamanho da janela de treino.

RF-15: Janelas decrescentes [2001-2022, 2007-2022, 2012-2022, 2017-2022, 2019-2022].
Usa INMET enriquecido apenas.
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
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
GROUP_KEY = "cod_mesorregiao"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

WINDOWS = [
    {"label": "2001-2022 (22yr)", "start": 2001, "end": 2022},
    {"label": "2007-2022 (16yr)", "start": 2007, "end": 2022},
    {"label": "2012-2022 (11yr)", "start": 2012, "end": 2022},
    {"label": "2017-2022 (6yr)", "start": 2017, "end": 2022},
    {"label": "2019-2022 (4yr)", "start": 2019, "end": 2022},
]


def get_features(schema_path):
    schema = pd.read_csv(schema_path)
    return schema["feature"].tolist()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  v4-51: Curva de Aprendizado — R²_log vs Janela de Treino")
    print("=" * 70)

    features = get_features(SCHEMA_PATH)
    print(f"  Features (enriquecido): {len(features)}")

    print("\nCarregando dataset completo v4...")
    df = pd.read_parquet(DATA_DIR / "integrated_mesorregiao_v4.parquet")
    df = df.sort_values(ID_COLS).reset_index(drop=True)

    df[TARGET] = df.groupby(GROUP_KEY)["notificacoes"].shift(-4)
    df = df.dropna(subset=[TARGET])

    drop_cols = ["nome_mesorregiao", "uf_sigla", "nome_grande_regiao", "n_municipios",
                 "n_estacoes", "flag_fonte_inmet"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    non_numeric = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("datetime")]
    if non_numeric:
        df = df.drop(columns=non_numeric)

    val = df[df["ano"] == 2023]
    test = df[df["ano"].isin([2024, 2025, 2026])]
    print(f"  Val: {len(val):,} | Test: {len(test):,}")

    results = []
    for w in WINDOWS:
        train = df[(df["ano"] >= w["start"]) & (df["ano"] <= w["end"])]
        n_years = train["ano"].nunique()
        print(f"\n  {w['label']}: {len(train):,} train rows, {n_years} anos")

        y_train = np.log1p(train[TARGET])
        y_val = np.log1p(val[TARGET])
        y_test_log = np.log1p(test[TARGET]).values

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[features], y_train, eval_set=[(val[features], y_val)], verbose=0)

        pred_log = model.predict(test[features])
        r2_log = r2_score(y_test_log, pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_test_log, pred_log))

        print(f"    R²_log={r2_log:.4f}  RMSE_log={rmse_log:.4f}  best_iter={model.best_iteration}")

        results.append({
            "label": w["label"],
            "start_year": w["start"],
            "end_year": w["end"],
            "n_years": n_years,
            "n_train_rows": len(train),
            "R2_log": round(float(r2_log), 4),
            "RMSE_log": round(float(rmse_log), 4),
            "best_iteration": int(model.best_iteration),
        })

    fig, ax = plt.subplots(figsize=(10, 6))
    n_years_list = [r["n_years"] for r in results]
    r2s = [r["R2_log"] for r in results]
    labels = [r["label"] for r in results]

    ax.plot(n_years_list, r2s, "o-", color="#1976D2", linewidth=2, markersize=10)
    for x, y, lbl in zip(n_years_list, r2s, labels):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)

    ax.set_xlabel("Anos de treino", fontsize=11)
    ax.set_ylabel("R²_log no teste (2024-2026)", fontsize=11)
    ax.set_title("Curva de Aprendizado — Impacto da Janela Temporal", fontsize=12)
    ax.set_xticks(n_years_list)
    ax.set_xticklabels([f"{n}yr" for n in n_years_list])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_v4_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Salvo: fig_v4_learning_curve.png")

    report = {
        "windows": results,
        "best_window": max(results, key=lambda r: r["R2_log"])["label"],
        "tempo_s": round(time.time() - t0, 1),
    }
    with open(DATA_DIR / "51_learning_curve_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-51: Curva de Aprendizado")
    for r in results:
        print(f"  {r['label']:20s}  R²_log={r['R2_log']:.4f}")
    print(f"  Melhor: {report['best_window']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

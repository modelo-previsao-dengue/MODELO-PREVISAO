#!/usr/bin/env python3
"""v4-61: Classificação completa com baselines + antecedência corrigida.

US-010: Persistência, climatologia e regressão realista na tabela de classificação
US-013: Métrica de antecedência do alerta corrigida
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"
AUDIT_PATH = DATA_DIR / "feature_availability_audit.json"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
GROUP_KEY = "cod_mesorregiao"
CLASS_NAMES = ["baixo", "moderado", "alto", "surto"]

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


def compute_thresholds(train):
    """P50/P75/P90 de notificacoes_t4 por mesorregião no treino."""
    groups = train.groupby(GROUP_KEY)[TARGET]
    thresholds = groups.quantile([0.50, 0.75, 0.90]).unstack()
    thresholds.columns = ["p50", "p75", "p90"]
    return thresholds


def apply_thresholds(values, meso_codes, thresholds):
    """Converte previsões numéricas em classes de risco usando limiares por mesorregião."""
    df = pd.DataFrame({"cod_mesorregiao": meso_codes, "pred_value": values})
    df = df.merge(thresholds, left_on="cod_mesorregiao", right_index=True, how="left")

    risk = np.zeros(len(df), dtype=int)
    risk[df["pred_value"].values > df["p50"].values] = 1
    risk[df["pred_value"].values > df["p75"].values] = 2
    risk[df["pred_value"].values > df["p90"].values] = 3
    return risk


def get_realista_features():
    schema = pd.read_csv(SCHEMA_PATH)
    with open(AUDIT_PATH) as f:
        audit = json.load(f)
    delayed = set(audit["delayed_consolidation"]["features"].keys())
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet = schema.loc[schema["category"] != "sinan", "feature"].tolist()
    realtime_sinan = [f for f in sinan if f not in delayed]
    return realtime_sinan + ["notificacoes"] + inmet


def classify_metrics(y_true, y_pred, label):
    """F1 macro, accuracy, F1 por classe."""
    f1_macro = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)
    acc = round(accuracy_score(y_true, y_pred), 3)
    f1_per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        f1_per_class[name] = round(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 3)
    print(f"  {label:30s}: F1_macro={f1_macro} | Acc={acc}")
    return {"F1_macro": f1_macro, "accuracy": acc, "F1_por_classe": f1_per_class}


def train_realista_regression(train, val, features):
    available = [f for f in features if f in train.columns]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[available].values, np.log1p(train[TARGET].values),
        eval_set=[(val[available].values, np.log1p(val[TARGET].values))],
        verbose=False
    )
    return model, available


def predict_regression(model, df, features):
    pred_log = model.predict(df[features].values)
    return np.expm1(np.maximum(pred_log, 0))


def climatologia_pred(train, test):
    clim = train.groupby(["cod_mesorregiao", "semana_epidemiologica"])[TARGET].mean()
    global_mean = train[TARGET].mean()
    pred = np.array([clim.get((r, s), global_mean) for r, s in
                     zip(test["cod_mesorregiao"], test["semana_epidemiologica"])])
    return pred


def antecedencia_alerta(test, pred_model, pred_tendencia):
    """US-013: Mede semanas de antecipação da subida antes do pico.

    Para cada mesorregião-temporada (ano):
    - Pico real: semana com max(notificacoes_t4)
    - Alerta modelo: primeira semana onde pred[s] > pred[s-1] por 2+ semanas consecutivas antes do pico
    - Alerta tendência: primeira semana onde notif_lag1 > notif_lag2 por 2+ semanas consecutivas antes do pico
    - Antecipação = semana_pico - semana_primeiro_alerta
    """
    print("\n--- Antecedência do alerta ---")
    test_c = test[["cod_mesorregiao", "ano", "semana_epidemiologica", TARGET]].copy()
    test_c["pred_model"] = pred_model

    if "notificacoes_lag_1w" in test.columns:
        test_c["pred_tendencia"] = test["notificacoes_lag_1w"].values
    else:
        test_c["pred_tendencia"] = test_c[TARGET].values * 0

    results = []
    for (meso, ano), g in test_c.groupby(["cod_mesorregiao", "ano"]):
        if len(g) < 10:
            continue

        g = g.sort_values("semana_epidemiologica").reset_index(drop=True)
        pico_idx = g[TARGET].idxmax()
        pico_sem = g.loc[pico_idx, "semana_epidemiologica"]
        pico_val = g.loc[pico_idx, TARGET]

        if pico_val == 0:
            continue

        pre_pico = g[g["semana_epidemiologica"] < pico_sem]
        if len(pre_pico) < 3:
            continue

        def primeira_subida_consecutiva(values, n_consec=2):
            diff = np.diff(values)
            rising = diff > 0
            for i in range(len(rising) - n_consec + 1):
                if all(rising[i:i + n_consec]):
                    return i + 1
            return None

        idx_modelo = primeira_subida_consecutiva(pre_pico["pred_model"].values)
        idx_tendencia = primeira_subida_consecutiva(pre_pico["pred_tendencia"].values)

        antecip_modelo = None
        antecip_tend = None

        if idx_modelo is not None:
            sem_alerta_modelo = pre_pico.iloc[idx_modelo]["semana_epidemiologica"]
            antecip_modelo = int(pico_sem - sem_alerta_modelo)

        if idx_tendencia is not None:
            sem_alerta_tend = pre_pico.iloc[idx_tendencia]["semana_epidemiologica"]
            antecip_tend = int(pico_sem - sem_alerta_tend)

        is_surto = int(pico_val > g[TARGET].quantile(0.9)) if len(g) > 10 else 0

        results.append({
            "cod_mesorregiao": int(meso),
            "ano": int(ano),
            "semana_pico": int(pico_sem),
            "pico_valor": float(pico_val),
            "antecip_modelo_semanas": antecip_modelo,
            "antecip_tendencia_semanas": antecip_tend,
            "is_surto": is_surto,
        })

    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        return {"n_temporadas": 0}

    modelo_vals = df_res["antecip_modelo_semanas"].dropna()
    tendencia_vals = df_res["antecip_tendencia_semanas"].dropna()

    summary = {
        "n_temporadas": len(df_res),
        "n_surto": int(df_res["is_surto"].sum()),
        "modelo": {
            "n_com_alerta": len(modelo_vals),
            "mediana_semanas": round(float(modelo_vals.median()), 1) if len(modelo_vals) > 0 else None,
            "media_semanas": round(float(modelo_vals.mean()), 1) if len(modelo_vals) > 0 else None,
            "p25": round(float(modelo_vals.quantile(0.25)), 1) if len(modelo_vals) > 0 else None,
            "p75": round(float(modelo_vals.quantile(0.75)), 1) if len(modelo_vals) > 0 else None,
        },
        "tendencia_observada": {
            "n_com_alerta": len(tendencia_vals),
            "mediana_semanas": round(float(tendencia_vals.median()), 1) if len(tendencia_vals) > 0 else None,
            "media_semanas": round(float(tendencia_vals.mean()), 1) if len(tendencia_vals) > 0 else None,
        },
        "nota_limitacao": "Definição de temporada com surto condiciona no desfecho observado. Documentar como limitação.",
    }

    surto_rows = df_res[df_res["is_surto"] == 1]
    if len(surto_rows) > 0:
        sv = surto_rows["antecip_modelo_semanas"].dropna()
        summary["modelo_so_surtos"] = {
            "n_com_alerta": len(sv),
            "mediana_semanas": round(float(sv.median()), 1) if len(sv) > 0 else None,
        }

    print(f"  Temporadas: {summary['n_temporadas']} ({summary['n_surto']} com surto)")
    if summary["modelo"]["mediana_semanas"] is not None:
        print(f"  Modelo: mediana {summary['modelo']['mediana_semanas']} sem de antecipação")
    if summary["tendencia_observada"]["mediana_semanas"] is not None:
        print(f"  Tendência: mediana {summary['tendencia_observada']['mediana_semanas']} sem de antecipação")

    return summary


def main():
    t0 = time.time()
    print("=" * 60)
    print("  v4-61: Classificação Completa + Antecedência")
    print("=" * 60)

    train, val, test = load_splits()
    add_notificacoes(train, val, test)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    thresholds = compute_thresholds(train)
    y_true = test[CLASS_TARGET].values.astype(int)
    meso_codes = test["cod_mesorregiao"].values

    report = {"classificacao": {}}

    # 1. Persistência
    print("\n--- Classificação ---")
    pred_pers = test["notificacoes"].values
    cls_pers = apply_thresholds(pred_pers, meso_codes, thresholds)
    report["classificacao"]["persistencia"] = classify_metrics(y_true, cls_pers, "Persistência")

    # 2. Climatologia
    pred_clim = climatologia_pred(train, test)
    cls_clim = apply_thresholds(pred_clim, meso_codes, thresholds)
    report["classificacao"]["climatologia"] = classify_metrics(y_true, cls_clim, "Climatologia")

    # 3. Regressão XGBoost realista → limiares
    features = get_realista_features()
    model, used_features = train_realista_regression(train, val, features)
    pred_reg = predict_regression(model, test, used_features)
    cls_reg = apply_thresholds(pred_reg, meso_codes, thresholds)
    report["classificacao"]["regressao_realista"] = classify_metrics(y_true, cls_reg, "Regressão realista → limiares")

    # 4. Classificador XGBoost B (do script 54, se existir)
    cls_model_path = BASE_DIR / "models" / "classification_v4" / "model_B_inmet_enriquecido_cls.ubj"
    if cls_model_path.exists():
        schema = pd.read_csv(SCHEMA_PATH)
        all_features = schema["feature"].tolist()
        available_cls = [f for f in all_features if f in test.columns]

        cls_model = xgb.XGBClassifier()
        cls_model.load_model(str(cls_model_path))

        cls_xgb_pred = cls_model.predict(test[available_cls].values)
        report["classificacao"]["classificador_xgb_B"] = classify_metrics(y_true, cls_xgb_pred, "Classificador XGBoost B")
    else:
        print(f"  Classificador XGBoost B não encontrado em {cls_model_path}")

    # US-013: Antecedência corrigida
    pred_tendencia = test["notificacoes"].values
    antecedencia = antecedencia_alerta(test, pred_reg, pred_tendencia)
    report["antecedencia"] = antecedencia

    # Métricas somente 2024-2025
    test_consol = test[test["ano"] <= 2025]
    y_true_consol = test_consol[CLASS_TARGET].values.astype(int)
    meso_consol = test_consol["cod_mesorregiao"].values

    report["classificacao_sem_2026"] = {}

    pred_pers_c = test_consol["notificacoes"].values
    cls_pers_c = apply_thresholds(pred_pers_c, meso_consol, thresholds)
    report["classificacao_sem_2026"]["persistencia"] = classify_metrics(y_true_consol, cls_pers_c, "Persist. (2024-2025)")

    pred_reg_c = predict_regression(model, test_consol, used_features)
    cls_reg_c = apply_thresholds(pred_reg_c, meso_consol, thresholds)
    report["classificacao_sem_2026"]["regressao_realista"] = classify_metrics(y_true_consol, cls_reg_c, "Reg. realista (2024-2025)")

    report["tempo_total_s"] = round(time.time() - t0, 1)

    out_path = DATA_DIR / "61_classificacao_completa_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {out_path} ({report['tempo_total_s']} s)")


if __name__ == "__main__":
    main()

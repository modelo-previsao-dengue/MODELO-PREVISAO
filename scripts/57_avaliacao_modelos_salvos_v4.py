#!/usr/bin/env python3
"""v4-57: Avaliacao complementar com os modelos v4 ja treinados (sem retreino).

Carrega os modelos salvos pelo script 46 (models/regression_v4/model_*.ubj) e o
teste/validacao versionados (data/model_ready_v4/splits_*/{val,test}.parquet) e
calcula o que a v4 ainda nao reportava, sobre exatamente os mesmos dados:

  1. Baselines ingenuos no teste da v4 (persistencia, media movel 4w, sazonal).
  2. Metricas por ano com previsto/real e cobertura do topo de 5%.
  3. Momento: erro na semana do pico por mesorregiao-temporada e acerto de
     direcao (sobe/desce) entre t e t+4.
  4. Recorte do Distrito Federal (mesorregiao 5301).
  5. Serie nacional (soma das mesorregioes) para a figura do teste.

Nao precisa do treino. Data blindness e limiares de alerta, que dependem do
treino, ficam no script 56.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
GROUP = "cod_mesorregiao"
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
DF_MESO = 5301
MODELOS = ["A_sinan_only", "A_inmet_enriquecido", "B_sinan_only", "B_inmet_enriquecido",
           "C_inmet_enriquecido"]


def reg_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.maximum(np.asarray(pred, float), 0)
    return {
        "R2_log": round(float(r2_score(np.log1p(y), np.log1p(pred))), 4),
        "R2": round(float(r2_score(y, pred)), 4),
        "MAE": round(float(mean_absolute_error(y, pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
        "previsto_total": round(float(pred.sum())),
        "real_total": round(float(y.sum())),
    }


def top_share(y, pred, q=0.95):
    y = np.asarray(y, float)
    m = y >= np.quantile(y, q)
    return round(float(np.maximum(pred[m], 0).sum() / y[m].sum()), 3)


def peak_timing(te, pred):
    df = te[[GROUP, "ano", TARGET, CLASS_TARGET]].copy()
    df["pred"] = pred
    err = []
    for _, g in df[df["ano"].isin([2024, 2025])].groupby([GROUP, "ano"]):
        if g[CLASS_TARGET].max() < 3:            # temporada sem nenhuma semana de surto
            continue
        err.append(int(g["pred"].values.argmax()) - int(g[TARGET].values.argmax()))
    e = np.array(err)
    return {
        "n_temporadas_com_surto": int(len(e)),
        "erro_mediano_abs_semanas": float(np.median(np.abs(e))),
        "erro_medio_semanas": round(float(e.mean()), 2),
        "dentro_2_semanas_pct": round(float((np.abs(e) <= 2).mean() * 100), 1),
        "pico_exato_pct": round(float((e == 0).mean() * 100), 1),
    }


def direction(te, pred):
    y0 = te["notificacoes"].to_numpy(float)
    change = te[TARGET].to_numpy(float) - y0
    rel = np.abs(change) >= np.maximum(5, 0.2 * y0)
    actual = np.sign(change[rel])
    trend = y0 - te["notificacoes_lag_1w"].to_numpy(float)
    return {
        "n_variacoes_relevantes": int(rel.sum()),
        "modelo_acerto_pct": round(float((np.sign(pred[rel] - y0[rel]) == actual).mean() * 100), 1),
        "tendencia_ingenua_acerto_pct": round(float((np.sign(trend[rel]) == actual).mean() * 100), 1),
    }


def main():
    te = pd.read_parquet(DATA_DIR / "splits_b" / "test.parquet").sort_values(ID_COLS).reset_index(drop=True)
    va = pd.read_parquet(DATA_DIR / "splits_b" / "val.parquet").sort_values(ID_COLS).reset_index(drop=True)
    y = te[TARGET].to_numpy(float)
    print(f"Teste: {len(te):,} linhas")

    # baselines
    full = pd.concat([va, te], ignore_index=True)[ID_COLS + ["notificacoes"]].sort_values(ID_COLS)
    assert full.groupby(GROUP).size().nunique() == 1, "serie nao densa"
    full["sazonal"] = full.groupby(GROUP)["notificacoes"].shift(48)
    sazonal = te[ID_COLS].merge(full[ID_COLS + ["sazonal"]], on=ID_COLS, how="left")["sazonal"].to_numpy()
    assert not np.isnan(sazonal).any()
    preds = {
        "persistencia": te["notificacoes"].to_numpy(float),
        "media_movel_4": te[["notificacoes", "notificacoes_lag_1w", "notificacoes_lag_2w",
                             "notificacoes_lag_3w"]].mean(axis=1).to_numpy(float),
        "sazonal": sazonal,
    }

    # modelos salvos
    ref = json.loads((DATA_DIR / "46_regression_v4_report.json").read_text(encoding="utf-8"))["modelos"]
    conferencia = {}
    for name in MODELOS:
        booster = xgb.Booster()
        booster.load_model(MODEL_DIR / f"model_{name}.ubj")
        feats = booster.feature_names
        p = booster.predict(xgb.DMatrix(te[feats], feature_names=feats))
        preds[name] = np.maximum(np.expm1(p), 0)
        r2l = float(r2_score(np.log1p(y), p))
        conferencia[name] = {"R2_log_aqui": round(r2l, 4), "R2_log_script46": round(ref[name]["R2_log"], 4)}
        print(f"  {name}: R2_log {r2l:.4f} (script 46: {ref[name]['R2_log']:.4f})")

    report = {"conferencia_script46": conferencia, "n_teste": int(len(te))}
    report["teste_agregado"] = {}
    for k, v in preds.items():
        r = reg_metrics(y, v)
        r["topo5pct_cobertura"] = top_share(y, v)
        report["teste_agregado"][k] = r

    rows = []
    for ano in sorted(te["ano"].unique()):
        m = (te["ano"] == ano).to_numpy()
        for k, v in preds.items():
            r = reg_metrics(y[m], v[m])
            r.update(ano=int(ano), modelo=k, topo5pct_cobertura=top_share(y[m], v[m]))
            rows.append(r)
    report["por_ano"] = rows
    pd.DataFrame(rows).to_csv(MODEL_DIR / "57_metricas_por_ano_com_baselines.csv", index=False)

    report["momento_pico"] = {k: peak_timing(te, v) for k, v in preds.items()}
    report["direcao"] = {k: direction(te, preds[k]) for k in MODELOS}

    m = (te[GROUP] == DF_MESO).to_numpy()
    report["distrito_federal"] = {
        "semanas_teste": int(m.sum()),
        "metricas": {k: reg_metrics(y[m], preds[k][m]) for k in
                     ("persistencia", "A_sinan_only", "A_inmet_enriquecido", "B_sinan_only", "B_inmet_enriquecido")},
        "notificacoes_semana_t_por_ano": {int(a): round(float(s)) for a, s in
                                          te[m].groupby("ano")["notificacoes"].sum().items()},
    }

    nac = te[["ano", "semana_epidemiologica"]].copy()
    nac["real"] = y
    for k in ("persistencia", "A_sinan_only", "B_inmet_enriquecido"):
        nac[k] = preds[k]
    nac.groupby(["ano", "semana_epidemiologica"]).sum().reset_index() \
       .to_csv(MODEL_DIR / "57_serie_nacional_teste.csv", index=False)

    out = DATA_DIR / "57_avaliacao_modelos_salvos_v4_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relatorio: {out}")


if __name__ == "__main__":
    main()

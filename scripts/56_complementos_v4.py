#!/usr/bin/env python3
"""v4-56: Analises complementares da v4 (INMET corrigido).

O que a v4 (scripts 40-55) ainda nao tinha:
  1. Baselines ingenuos no teste da v4 (persistencia, media movel 4w, sazonal),
     com as mesmas metricas do script 46, no agregado e por ano.
  2. Reproducao dos modelos A/B (so SINAN e clima enriquecido) para conferir o
     script 46 e guardar as previsoes.
  3. Metricas de MOMENTO: erro na semana do pico por mesorregiao-temporada,
     antecedencia na entrada em risco alto e acerto de direcao (sobe/desce).
  4. Alertas: limiares de risco do treino aplicados a previsao da regressao e
     aos baselines (F1 macro), comparaveis ao classificador do script 54.
  5. Data blindness na variante B (atraso 4w, 8w, so clima), com e sem clima,
     e repeticao com sementes diferentes para medir a variacao entre treinos.
  6. Recorte do Distrito Federal (mesorregiao 5301).

Saidas: data/model_ready_v4/56_complementos_v4_report.json e
models/regression_v4/56_*.csv.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
OUT_DIR = BASE_DIR / "models" / "regression_v4"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
GROUP = "cod_mesorregiao"
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
DF_MESO = 5301

XGB_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method="hist",
    random_state=42, n_jobs=-1, early_stopping_rounds=50,
)

BLIND_4W = [
    "notificacoes_lag_1w", "notificacoes_lag_2w", "notificacoes_lag_3w", "notificacoes_lag_4w",
    "notificacoes_diff_1w", "notificacoes_pct_change_1w", "notificacoes_media_movel_2_4w",
]
BLIND_8W = BLIND_4W + [
    "notificacoes_lag_5w", "notificacoes_lag_6w", "notificacoes_lag_7w", "notificacoes_lag_8w",
    "notificacoes_media_movel_4_8w",
]


# --------------------------------------------------------------------------- dados

def load_split(variant, ate_semana_ultimo_ano=None):
    """Carrega os splits; opcionalmente corta o ultimo ano de teste numa semana.

    O corte existe para comparar com uma rodada feita sobre um snapshot do SINAN
    com menos semanas no ultimo ano (ex.: 2026 ate a semana 12).
    """
    d = DATA_DIR / f"splits_{variant}"
    out = [pd.read_parquet(d / f"{s}.parquet").sort_values(ID_COLS).reset_index(drop=True)
           for s in ("train", "val", "test")]
    if ate_semana_ultimo_ano is not None:
        te = out[2]
        ult = te["ano"].max()
        keep = (te["ano"] < ult) | (te["semana_epidemiologica"] <= ate_semana_ultimo_ano)
        out[2] = te[keep].reset_index(drop=True)
    return tuple(out)


def feature_sets():
    s = pd.read_csv(SCHEMA_PATH)
    sinan = s.loc[s["category"] == "sinan", "feature"].tolist()
    bruto = s.loc[s["category"] == "inmet_bruto", "feature"].tolist()
    enr = s.loc[s["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"].tolist()
    return sinan, bruto + enr


def thresholds_from_train(train):
    t = train.groupby(GROUP)[TARGET].quantile([0.50, 0.75, 0.90]).unstack()
    t.columns = ["p50", "p75", "p90"]
    return t


def to_risk(values, meso, thr):
    m = thr.reindex(meso).to_numpy()
    v = np.asarray(values, dtype=float)
    r = np.zeros(len(v), dtype=int)
    r[v > m[:, 0]] = 1
    r[v > m[:, 1]] = 2
    r[v > m[:, 2]] = 3
    return r


# --------------------------------------------------------------------------- metricas

def reg_metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.maximum(np.asarray(pred, float), 0)
    yl, pl = np.log1p(y), np.log1p(pred)
    return {
        "R2_log": round(float(r2_score(yl, pl)), 4),
        "R2": round(float(r2_score(y, pred)), 4),
        "MAE": round(float(mean_absolute_error(y, pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
        "previsto_total": float(pred.sum()),
        "real_total": float(y.sum()),
    }


def top_share(y, pred, q=0.95):
    """Fracao do total observado coberta pela previsao nas linhas do topo q."""
    y = np.asarray(y, float)
    cut = np.quantile(y, q)
    m = y >= cut
    return round(float(np.maximum(pred[m], 0).sum() / y[m].sum()), 3)


def timing_metrics(test, pred, thr, label):
    """Erro na semana do pico, antecedencia na entrada em risco alto e direcao."""
    df = test[ID_COLS + ["notificacoes", TARGET, CLASS_TARGET]].copy()
    df["pred"] = np.maximum(pred, 0)
    df["pred_risk"] = to_risk(df["pred"], df[GROUP], thr)

    peak_err, onset_lead = [], []
    for (meso, ano), g in df[df["ano"].isin([2024, 2025])].groupby([GROUP, "ano"]):
        g = g.reset_index(drop=True)
        if g[CLASS_TARGET].max() < 3:          # temporada sem nenhuma semana de surto
            continue
        peak_err.append(int(g["pred"].values.argmax()) - int(g[TARGET].values.argmax()))
        obs_on = np.flatnonzero(g[CLASS_TARGET].values >= 2)
        pred_on = np.flatnonzero(g["pred_risk"].values >= 2)
        if len(obs_on) and len(pred_on):
            onset_lead.append(int(obs_on[0]) - int(pred_on[0]))   # >0: previsao antecipa

    pe = np.array(peak_err)
    ol = np.array(onset_lead)
    out = {
        "n_temporadas_com_surto": int(len(pe)),
        "pico_erro_mediano_abs_semanas": float(np.median(np.abs(pe))),
        "pico_erro_medio_semanas": round(float(pe.mean()), 2),
        "pico_dentro_2_semanas_pct": round(float((np.abs(pe) <= 2).mean() * 100), 1),
        "pico_exato_pct": round(float((pe == 0).mean() * 100), 1),
        "entrada_risco_alto_n": int(len(ol)),
        "entrada_risco_alto_antecedencia_mediana": float(np.median(ol)) if len(ol) else None,
        "entrada_risco_alto_no_prazo_ou_antes_pct": round(float((ol >= 0).mean() * 100), 1) if len(ol) else None,
    }
    print(f"    [{label}] pico: |erro| mediano {out['pico_erro_mediano_abs_semanas']} sem, "
          f"{out['pico_dentro_2_semanas_pct']}% em +-2 sem (n={len(pe)})")
    return out


def direction_metrics(test, pred):
    """Acerto do sentido da variacao entre t e t+4, so nas variacoes relevantes."""
    y0 = test["notificacoes"].to_numpy(float)
    y4 = test[TARGET].to_numpy(float)
    change = y4 - y0
    relevant = np.abs(change) >= np.maximum(5, 0.2 * y0)
    actual = np.sign(change[relevant])
    out = {"n_variacoes_relevantes": int(relevant.sum())}
    out["xgboost_acerto_pct"] = round(float((np.sign(pred[relevant] - y0[relevant]) == actual).mean() * 100), 1)
    trend = y0 - test["notificacoes_lag_1w"].to_numpy(float)        # "continua como na ultima semana"
    out["tendencia_ingenua_acerto_pct"] = round(float((np.sign(trend[relevant]) == actual).mean() * 100), 1)
    return out


# --------------------------------------------------------------------------- modelos

def fit_predict(train, val, test, feats, seed=42, sample_weight=None):
    params = dict(XGB_PARAMS, random_state=seed)
    model = xgb.XGBRegressor(**params)
    kw = dict(eval_set=[(val[feats], np.log1p(val[TARGET]))], verbose=0)
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    model.fit(train[feats], np.log1p(train[TARGET]), **kw)
    return model.predict(test[feats]), model.best_iteration


def blind(df, cols):
    out = df.copy()
    out[[c for c in cols if c in out.columns]] = np.nan
    return out


# --------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    sinan, clima = feature_sets()
    enr = sinan + clima
    report = {}

    tr_b, va_b, te = load_split("b", ARGS.ate_semana_ultimo_ano)
    tr_a, va_a, te_a = load_split("a", ARGS.ate_semana_ultimo_ano)
    report["corte_ultimo_ano_semana"] = ARGS.ate_semana_ultimo_ano
    assert te[ID_COLS].equals(te_a[ID_COLS]), "teste difere entre variantes"
    y = te[TARGET].to_numpy(float)
    print(f"Teste: {len(te):,} linhas | treino B {len(tr_b):,} | treino A {len(tr_a):,}")

    # 1. baselines --------------------------------------------------------------
    full = pd.concat([tr_b, va_b, te], ignore_index=True)[ID_COLS + ["notificacoes"]]
    full = full.sort_values(ID_COLS).reset_index(drop=True)
    assert full.groupby(GROUP).size().nunique() == 1, "serie nao densa: shift(48) invalido"
    full["sazonal"] = full.groupby(GROUP)["notificacoes"].shift(48)
    sazonal = te[ID_COLS].merge(full[ID_COLS + ["sazonal"]], on=ID_COLS, how="left")["sazonal"].to_numpy()
    assert not np.isnan(sazonal).any()
    preds = {
        "persistencia": te["notificacoes"].to_numpy(float),
        "media_movel_4": te[["notificacoes", "notificacoes_lag_1w", "notificacoes_lag_2w",
                             "notificacoes_lag_3w"]].mean(axis=1).to_numpy(float),
        "sazonal": sazonal,
    }

    # 2. modelos ----------------------------------------------------------------
    print("\nTreinando modelos de referencia (semente 42)...")
    iters = {}
    for name, (tr, va, feats) in {
        "A_sinan_only": (tr_a, va_a, sinan), "A_inmet_enriquecido": (tr_a, va_a, enr),
        "B_sinan_only": (tr_b, va_b, sinan), "B_inmet_enriquecido": (tr_b, va_b, enr),
    }.items():
        p, it = fit_predict(tr, va, te, feats)
        preds[name] = np.maximum(np.expm1(p), 0)
        iters[name] = int(it)
        print(f"  {name}: R2_log={r2_score(np.log1p(y), p):.4f} (iter {it})")

    ref46 = json.loads((DATA_DIR / "46_regression_v4_report.json").read_text(encoding="utf-8"))["modelos"]
    report["reproducao_script46"] = {
        k: {"R2_log_aqui": reg_metrics(y, preds[k])["R2_log"], "R2_log_script46": round(ref46[k]["R2_log"], 4),
            "best_iteration": iters[k]} for k in iters
    }

    report["teste_agregado"] = {k: reg_metrics(y, v) for k, v in preds.items()}
    for k, v in report["teste_agregado"].items():
        v["topo5pct_cobertura"] = top_share(y, preds[k])

    por_ano = []
    for ano in sorted(te["ano"].unique()):
        m = (te["ano"] == ano).to_numpy()
        for k, v in preds.items():
            r = reg_metrics(y[m], v[m])
            r.update(ano=int(ano), modelo=k, topo5pct_cobertura=top_share(y[m], v[m]))
            por_ano.append(r)
    pd.DataFrame(por_ano).to_csv(OUT_DIR / "56_metricas_por_ano_com_baselines.csv", index=False)
    report["por_ano"] = por_ano

    # serie nacional (soma das mesorregioes) para a figura
    nac = te[ID_COLS[1:]].copy()
    nac["real"] = y
    for k in ("persistencia", "A_sinan_only", "B_inmet_enriquecido"):
        nac[k] = preds[k]
    nac.groupby(["ano", "semana_epidemiologica"]).sum(numeric_only=True).reset_index() \
       .to_csv(OUT_DIR / "56_serie_nacional_teste.csv", index=False)

    # 3/4. momento e alertas ----------------------------------------------------
    print("\nMetricas de momento e alertas...")
    thr_b, thr_a = thresholds_from_train(tr_b), thresholds_from_train(tr_a)
    check = (to_risk(y, te[GROUP], thr_b) == te[CLASS_TARGET].to_numpy()).mean()
    report["limiares_reproduzem_rotulo_B_pct"] = round(float(check * 100), 2)
    timing, alerts = {}, {}
    for k, v in preds.items():
        thr = thr_a if k.startswith("A_") else thr_b
        timing[k] = timing_metrics(te, v, thr, k)
        r = to_risk(v, te[GROUP], thr)
        alerts[k] = {
            "F1_macro": round(float(f1_score(te[CLASS_TARGET], r, average="macro")), 4),
            "F1_por_classe": [round(float(x), 4) for x in f1_score(te[CLASS_TARGET], r, average=None)],
        }
    report["momento"] = timing
    report["direcao"] = {k: direction_metrics(te, preds[k]) for k in
                         ("A_sinan_only", "B_sinan_only", "B_inmet_enriquecido")}
    report["alertas_por_limiar"] = alerts

    # 5. data blindness (variante B) --------------------------------------------
    print("\nData blindness (variante B)...")
    yl = np.log1p(y)
    scen = {"completo": [], "atraso_4w": BLIND_4W, "atraso_8w": BLIND_8W, "so_clima": sinan}
    bl = {}
    for sname, cols in scen.items():
        trb, vab, teb = blind(tr_b, cols), blind(va_b, cols), blind(te, cols)
        p_sem, _ = fit_predict(trb, vab, teb, sinan)
        p_com, _ = fit_predict(trb, vab, teb, enr)
        e_sem, e_com = (yl - p_sem) ** 2, (yl - p_com) ** 2
        bl[sname] = {
            "R2_log_sem_clima": round(float(r2_score(yl, p_sem)), 4),
            "R2_log_com_clima": round(float(r2_score(yl, p_com)), 4),
            "delta": round(float(r2_score(yl, p_com) - r2_score(yl, p_sem)), 4),
            "p_valor": float(ttest_rel(e_sem, e_com).pvalue),
            "R2_com_clima_escala_original": round(float(r2_score(y, np.maximum(np.expm1(p_com), 0))), 4),
        }
        print(f"  {sname}: sem {bl[sname]['R2_log_sem_clima']} | com {bl[sname]['R2_log_com_clima']}")
    report["blindness_B"] = bl

    print("\nVariacao entre sementes (sem clima)...")
    seeds = {}
    for sname in ("completo", "atraso_4w"):
        cols = scen[sname]
        trb, vab, teb = blind(tr_b, cols), blind(va_b, cols), blind(te, cols)
        vals = [round(float(r2_score(yl, fit_predict(trb, vab, teb, sinan, seed=s)[0])), 4) for s in (1, 2, 3)]
        seeds[sname] = {"sementes_1_2_3": vals, "media": round(float(np.mean(vals)), 4)}
        print(f"  {sname}: {vals}")
    report["blindness_B_sementes"] = seeds

    # 6. Distrito Federal ---------------------------------------------------------
    m = (te[GROUP] == DF_MESO).to_numpy()
    report["distrito_federal"] = {
        "semanas_teste": int(m.sum()),
        "metricas": {k: reg_metrics(y[m], preds[k][m]) for k in
                     ("persistencia", "A_sinan_only", "B_sinan_only", "B_inmet_enriquecido")},
        "notificacoes_por_ano_semana_t": {
            int(a): float(v) for a, v in
            full[full[GROUP] == DF_MESO].groupby("ano")["notificacoes"].sum().items() if a >= 2019
        },
    }

    report["tempo_s"] = round(time.time() - t0, 1)
    out = DATA_DIR / "56_complementos_v4_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatorio: {out} ({report['tempo_s']} s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ate-semana-ultimo-ano", type=int, default=None,
                    help="Corta o ultimo ano do teste nesta semana (comparabilidade entre snapshots)")
    ARGS = ap.parse_args()
    main()

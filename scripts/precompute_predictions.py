"""
Pré-computa previsões XGBoost para todas as combinações var×sub.
Salva parquets leves em dashboard/data/ para uso em produção
(onde train.parquet não está disponível).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "model_ready_v4"
REG_DIR = ROOT / "models" / "regression_v4"
CLS_DIR = ROOT / "models" / "classification_v4"
OUT_DIR = ROOT / "dashboard" / "data"

RISK = {0: "Baixo", 1: "Moderado", 2: "Alto", 3: "Surto"}
UF = {
    11:"RO",12:"AC",13:"AM",14:"RR",15:"PA",16:"AP",17:"TO",
    21:"MA",22:"PI",23:"CE",24:"RN",25:"PB",26:"PE",27:"AL",
    28:"SE",29:"BA",31:"MG",32:"ES",33:"RJ",35:"SP",
    41:"PR",42:"SC",43:"RS",50:"MS",51:"MT",52:"GO",53:"DF",
}

VARIANTS = ["A", "B", "C"]
SUBS = ["inmet_enriquecido", "inmet_bruto", "sinan_only"]


def load_splits(var: str):
    dfs = []
    for split in ["train", "val", "test"]:
        p = DATA / f"splits_{var.lower()}" / f"{split}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["split"] = split
            dfs.append(df)
    if not dfs:
        return None
    full = pd.concat(dfs, ignore_index=True)
    full["uf_code"] = full["cod_mesorregiao"] // 100
    full["uf"] = full["uf_code"].map(UF)
    return full


def run_predictions(rm, cm, df):
    feats = rm.feature_names
    dm = xgb.DMatrix(df[feats], feature_names=feats)
    pred_log = rm.predict(dm)
    pred_notif = np.maximum(np.expm1(pred_log), 0)
    probs = cm.predict(dm)
    pred_cls = probs.argmax(axis=1)

    out = df[["cod_mesorregiao", "ano", "semana_epidemiologica",
              "notificacoes", "uf", "notificacoes_t4", "risco_surto_t4", "split"]].copy()
    out["pred_notif"] = pred_notif
    out["pred_risco"] = pred_cls
    out["pred_label"] = pd.Series(pred_cls).map(RISK).values
    out["real_label"] = df["risco_surto_t4"].map(RISK).values
    for i, nm in RISK.items():
        out[f"p_{nm.lower()}"] = probs[:, i]
    out["periodo"] = (out["ano"].astype(str) + "-S"
                      + out["semana_epidemiologica"].astype(str).str.zfill(2))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for var in VARIANTS:
        df = load_splits(var)
        if df is None:
            print(f"[SKIP] Var {var}: splits not found")
            continue

        for sub in SUBS:
            rp = REG_DIR / f"model_{var}_{sub}.ubj"
            cp = CLS_DIR / f"model_{var}_{sub}_cls.ubj"
            if not rp.exists() or not cp.exists():
                print(f"[SKIP] {var}_{sub}: models not found")
                continue

            rm = xgb.Booster()
            rm.load_model(str(rp))
            cm = xgb.Booster()
            cm.load_model(str(cp))

            preds = run_predictions(rm, cm, df)
            out_path = OUT_DIR / f"predictions_{var}_{sub}.parquet"
            preds.to_parquet(out_path, index=False, compression="snappy")
            sz = out_path.stat().st_size / 1e6
            print(f"[OK] {var}_{sub}: {len(preds):,} rows, {sz:.1f}MB -> {out_path.name}")
            total += 1

    print(f"\nDone: {total} prediction files generated in {OUT_DIR}")


if __name__ == "__main__":
    main()

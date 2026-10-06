#!/usr/bin/env python3
"""US-010: intervalos de predicao por regressao quantilica.

2024 foi 5,6x o maximo historico da serie e derrubou o R2 de teste de todos
os modelos. Uma previsao pontual erra feio nesse tipo de evento e nao tem
como se defender; um intervalo de predicao contem o surto na banda superior
e explicita a incerteza. Nao e para virar metrica oficial do TCC (isso
seria WIS/CRPS, decisao adiada) — sao bandas para as figuras e um
diagnostico de calibracao.

XGBoost 3.x aceita `quantile_alpha` como array e treina um unico modelo
multi-saida — nao um modelo por quantil. Treina em log1p e aplica expm1 a
cada quantil: quantis sao equivariantes sob transformacao monotona, entao
expm1(q_a(log1p Y)) = q_a(Y) exatamente, sem o vies de retransformacao que
a media sofre.

Script separado de train_ablation.py de proposito: os hiperparametros ja
estao em melhores_params_* no metrics.json do braco, entao nao precisa
re-tunar, e o tempo de execucao dos bracos existentes fica intocado.

Uso:
    python3 scripts/train_quantis.py --arm meso_surto_sinan_inmet
    python3 scripts/train_quantis.py --arm meso_sinan_inmet --raiz models/ablacao
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canal_endemico  # noqa: E402
import metrics_common  # noqa: E402
import train_ablation as ta  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def _params_do_braco(metrics_path):
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    for chave in ("melhores_params_regressao", "melhores_params", "melhores"):
        if chave in d:
            return d[chave]
    raise KeyError(f"{metrics_path} sem hiperparametros de regressao salvos")


def _cfg_do_braco(arm, dataset_rel):
    """Reconstroi o cfg de ta.carregar_cfg a partir do nome do braco.

    Os bracos de surto (mesorregiao_surto) precisam do meta estendido para
    carregar as colunas de rotulo; os demais usam o meta padrao.
    """
    origens = ["sinan", "inmet"] if arm.endswith("inmet") else ["sinan"]
    cfg = {"dataset": dataset_rel, "origens": origens}
    if "mesorregiao_surto" in dataset_rel:
        cfg["meta"] = ta.NAO_FEATURE + canal_endemico.ROTULO_COLS
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--raiz", default="models/ablacao_surto")
    ap.add_argument("--quantis", default="0.05,0.5,0.95")
    args = ap.parse_args()

    alphas = np.array([float(x) for x in args.quantis.split(",")])
    if not (np.all(np.diff(alphas) > 0) and alphas[0] > 0 and alphas[-1] < 1):
        raise ValueError("--quantis deve ser crescente e estritamente entre 0 e 1")

    raiz = BASE_DIR / args.raiz / args.arm
    metrics_path = raiz / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} nao existe — rode o braco primeiro")

    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset_rel = d.get("dataset")
    params = _params_do_braco(metrics_path)

    print(f"Braco: {args.arm} | dataset: {dataset_rel} | quantis: {list(alphas)}")
    partes, feats, dataset = ta.carregar_cfg(_cfg_do_braco(args.arm, dataset_rel))
    n = {k: len(v["meta"]) for k, v in partes.items()}
    print(f"  features: {len(feats)} | train {n['train']:,} | "
          f"val {n['val']:,} | test {n['test']:,}")

    ytr = np.log1p(partes["train"]["meta"][ta.TARGET].to_numpy(dtype=float))
    yva = np.log1p(partes["val"]["meta"][ta.TARGET].to_numpy(dtype=float))

    fixos = ta.fixos_para(len(partes["train"]["X"]))
    modelo = xgb.XGBRegressor(objective="reg:quantileerror",
                              quantile_alpha=alphas, **params, **fixos)
    print("Treinando (multi-quantil, um so modelo)...")
    modelo.fit(partes["train"]["X"], ytr,
               eval_set=[(partes["val"]["X"], yva)], verbose=False)

    pred_log = modelo.predict(partes["test"]["X"])
    if pred_log.ndim == 1:
        pred_log = pred_log[:, None]
    pred = np.expm1(pred_log)

    n_cruz = int((np.diff(pred, axis=1) < 0).any(axis=1).sum())
    pred = np.sort(pred, axis=1)
    print(f"  cruzamento de quantis antes do sort: {n_cruz} linhas "
          f"({n_cruz / len(pred) * 100:.1f}%)")

    y_true = partes["test"]["meta"][ta.TARGET].to_numpy(dtype=float)
    ids = partes["test"]["meta"][ta.ID_COLS].reset_index(drop=True)

    nomes_q = [f"q{int(round(a * 100)):02d}" for a in alphas]
    saida = ids.copy()
    saida["y_true"] = y_true
    for j, nome in enumerate(nomes_q):
        saida[nome] = pred[:, j]
    saida.to_parquet(raiz / "predicoes_quantis.parquet", index=False)

    lo, hi = pred[:, 0], pred[:, -1]
    dentro = (y_true >= lo) & (y_true <= hi)
    largura = hi - lo
    intervalos = {
        "quantis": [float(a) for a in alphas],
        "nominal": round(float(alphas[-1] - alphas[0]), 4),
        "cobertura_empirica": round(float(dentro.mean()), 4),
        "largura_media": round(float(largura.mean()), 2),
        "largura_mediana": round(float(np.median(largura)), 2),
        "n_cruzamento_pre_sort": n_cruz,
        "n_teste": int(len(y_true)),
    }
    # Diagnostico da mediana como previsao pontual, so para contexto.
    if len(nomes_q) >= 2 and abs(alphas[len(alphas) // 2] - 0.5) < 1e-9:
        q50 = pred[:, len(alphas) // 2]
        intervalos["q50_como_pontual"] = metrics_common.regression_metrics(
            y_true, q50)

    d["intervalos"] = intervalos
    metrics_path.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print(f"  cobertura empirica [{nomes_q[0]},{nomes_q[-1]}]: "
          f"{intervalos['cobertura_empirica']:.3f} (nominal {intervalos['nominal']:.2f})")
    print(f"  largura: media {intervalos['largura_media']:.1f} | "
          f"mediana {intervalos['largura_mediana']:.1f} casos")
    print(f"\n-> {raiz / 'predicoes_quantis.parquet'}")
    print(f"-> {metrics_path} (chave 'intervalos' adicionada)")


if __name__ == "__main__":
    main()

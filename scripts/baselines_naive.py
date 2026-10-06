#!/usr/bin/env python3
"""US-007: baselines ingenuos avaliados no mesmo teste dos modelos.

O TCC comparava versoes de XGBoost entre si e nunca contra a alternativa
trivial. Um R2 de 0,42 parece bom ate se descobrir quanto dele um "repete o
valor de hoje" ja entrega de graca: previsao de dengue tem autocorrelacao
alta, e sem essa linha nao da para dizer que o modelo aprendeu alguma coisa.

Tres baselines, todos sobre o mesmo recorte, mesmo teste e mesmas metricas:

    persistencia     casos(t+4) = casos(t)
    sazonal          casos(t+4) = casos da mesma semana epidemiologica do ano
                     anterior
    media_movel_4    casos(t+4) = media das 4 ultimas semanas observadas

A classificacao usa os limiares do treino gravados pela US-002, os mesmos que
os modelos usam, para que a comparacao seja de previsao e nao de rotulo.

Uso:
    python3 scripts/baselines_naive.py
    python3 scripts/baselines_naive.py --dataset data/model_ready/e2 --saida models/baselines_e2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common

BASE_DIR = Path(__file__).resolve().parent.parent

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
QUANTIS = ["p50", "p75", "p90"]


def carregar(dataset):
    """Serie completa dos tres splits, so com o que os baselines usam."""
    cols = ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
    partes = {}
    for nome in ("train", "val", "test"):
        caminho = dataset / f"{nome}.parquet"
        # Ler so estas colunas importa: o treino do E2 tem 6,1 milhoes de
        # linhas por 133 colunas e abrir o parquet inteiro para descartar
        # quase tudo estoura a memoria da maquina.
        disponiveis = {f.name for f in pq.read_schema(caminho)}
        faltando = [c for c in cols if c not in disponiveis]
        if faltando:
            raise KeyError(f"{nome}.parquet sem as colunas {faltando}")
        d = pd.read_parquet(caminho, columns=cols)
        d["split"] = nome
        partes[nome] = d
    return pd.concat(partes.values(), ignore_index=True)


def prever(serie):
    """Adiciona uma coluna por baseline sobre a serie inteira.

    Os tres olham so para o passado da semana t, que e o que estaria
    disponivel na hora de prever t+4.
    """
    serie = serie.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])

    # Persistencia: o valor de hoje repetido quatro semanas a frente.
    serie["pred_persistencia"] = serie["notificacoes"]

    # Media movel das 4 ultimas semanas, inclusive a atual.
    serie["pred_media_movel_4"] = (
        serie.groupby("ibge_municipio", sort=False)["notificacoes"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    # Sazonal: o alvo da mesma semana no ano anterior e exatamente o valor
    # observado em t+4 um ano atras, entao basta deslocar a chave em um ano.
    anterior = serie[ID_COLS + [TARGET]].copy()
    anterior["ano"] = anterior["ano"] + 1
    anterior = anterior.rename(columns={TARGET: "pred_sazonal"})
    serie = serie.merge(anterior, on=ID_COLS, how="left")

    return serie


def classificar(pred, limiares):
    """Aplica os limiares do treino a previsao, como os modelos fazem."""
    lim = limiares[limiares["escopo"] == "municipio"][["ibge_municipio"] + QUANTIS]
    m = pred.merge(lim, on="ibge_municipio", how="left")
    v = m["_pred"]
    return np.select(
        [v <= m["p50"],
         (v > m["p50"]) & (v <= m["p75"]),
         (v > m["p75"]) & (v <= m["p90"]),
         v > m["p90"]],
        [0, 1, 2, 3], default=0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/model_ready")
    ap.add_argument("--saida", default="models/baselines")
    args = ap.parse_args()

    dataset = BASE_DIR / args.dataset
    saida = BASE_DIR / args.saida
    saida.mkdir(parents=True, exist_ok=True)

    serie = prever(carregar(dataset))
    limiares = pd.read_csv(dataset / "risk_thresholds.csv",
                           dtype={"ibge_municipio": str})

    teste = serie[serie["split"] == "test"].copy()
    print(f"Teste: {len(teste):,} linhas, anos {sorted(teste['ano'].unique())}\n")

    baselines = {
        "persistencia": "pred_persistencia",
        "sazonal": "pred_sazonal",
        "media_movel_4": "pred_media_movel_4",
    }

    resultados, linhas = {}, []
    for nome, col in baselines.items():
        valido = teste[teste[col].notna()].copy()
        cobertura = len(valido) / len(teste) * 100
        valido["_pred"] = valido[col]

        reg = metrics_common.regression_metrics(valido[TARGET], valido["_pred"])
        clf = metrics_common.classification_metrics(
            valido[CLASS_TARGET], classificar(valido, limiares))
        reg["cobertura_teste_pct"] = round(cobertura, 2)

        resultados[nome] = {"regressao": reg, "classificacao": clf}
        linha = metrics_common.to_row(f"baseline_{nome}", reg, clf)
        linha["cobertura_teste_pct"] = round(cobertura, 2)
        linhas.append(linha)

        print(f"{nome:16s} n={reg['n']:,} ({cobertura:.1f}% do teste)")
        print(f"  MAE  {reg['MAE_orig']:>10.3f} casos   | log {reg['MAE_log']:.4f}")
        print(f"  RMSE {reg['RMSE_orig']:>10.3f} casos")
        print(f"  R2   {reg['R2_orig']:>10.4f} (orig)  | log {reg['R2_log']:.4f}")
        print(f"  F1_macro {clf['F1_macro']:.4f} | acuracia {clf['acuracia']:.4f}\n")

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(saida / "comparison.csv", index=False)
    with open(saida / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "baselines": resultados},
                  f, indent=2, ensure_ascii=False)

    with open(saida / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Baselines ingenuos (US-007)\n\n")
        f.write(f"Dataset: `{args.dataset}` | teste: "
                f"{sorted(teste['ano'].unique().tolist())} | "
                f"{len(teste):,} linhas\n\n")
        f.write("MAE e RMSE em casos por municipio-semana; R2_orig na escala de\n"
                "contagem e R2_log sobre log1p.\n\n")
        f.write(tabela.to_markdown(index=False))
        f.write("\n")

    print(f"-> {saida}")


if __name__ == "__main__":
    main()

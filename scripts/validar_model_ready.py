#!/usr/bin/env python3
"""Checagem de sanidade dos datasets antes de subir para o HuggingFace.

Existe porque subir um dataset errado custa caro: o notebook do Kaggle nao
tem como saber que baixou a coisa errada, e o erro so aparece nas metricas,
tarde demais.

A verificacao mais importante e a ultima: as colunas que nunca podem virar
feature precisam existir nos parquets, para auditoria, e precisam estar
FORA do feature_schema.csv, que e de onde o notebook monta a lista de
features. Se uma delas vazasse para o schema, o modelo receberia o alvo.

Uso:
    python3 scripts/validar_model_ready.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
RAIZ = BASE_DIR / "data" / "model_ready"

CONJUNTOS = {"e1": RAIZ, "e2": RAIZ / "e2",
             "mesorregiao": RAIZ / "mesorregiao", "uf": RAIZ / "uf",
             "mesorregiao_surto": RAIZ / "mesorregiao_surto"}

OBRIGATORIOS = ["train.parquet", "val.parquet", "test.parquet",
                "feature_schema.csv", "risk_thresholds.csv",
                "resumo_dataset.json"]

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"

# Precisam existir nos parquets e estar fora do feature_schema.
NUNCA_FEATURE = [TARGET, "risco_surto_t4", "risco_surto_t4_com_vazamento",
                 "threshold_source"] + ID_COLS

# US-010: rotulo binario de surto por canal endemico (mesorregiao_surto). As
# flags "_t" sao funcao direta de notificacoes na propria semana — deixa-las
# virar feature seria vazar um proxy do alvo t+4; semana_alvo/ano_alvo sao a
# marcacao de auditoria do deslocamento, derivada do proprio alinhamento do
# alvo.
NUNCA_FEATURE += ["surto_media_t", "surto_media_2dp_t",
                  "surto_media_t4", "surto_media_2dp_t4",
                  "semana_alvo", "ano_alvo"]


def checar(nome, pasta):
    falhas, avisos = [], []

    faltando = [f for f in OBRIGATORIOS if not (pasta / f).exists()]
    if faltando:
        return [f"arquivos ausentes: {faltando}"], []

    resumo = json.loads((pasta / "resumo_dataset.json").read_text())
    schema = pd.read_csv(pasta / "feature_schema.csv")
    feats = schema["feature"].tolist()

    colunas_por_split, anos_por_split = {}, {}
    for split in ("train", "val", "test"):
        caminho = pasta / f"{split}.parquet"
        arq = pq.ParquetFile(caminho)
        cols = {f.name for f in arq.schema_arrow}
        colunas_por_split[split] = cols

        n = arq.metadata.num_rows
        esperado = resumo["linhas"][split]
        if n != esperado:
            falhas.append(f"{split}: {n:,} linhas, resumo diz {esperado:,}")

        faltam = [c for c in ID_COLS + [TARGET] if c not in cols]
        if faltam:
            falhas.append(f"{split}: colunas essenciais ausentes {faltam}")

        d = pd.read_parquet(caminho, columns=ID_COLS + [TARGET])
        if d[TARGET].isna().any():
            falhas.append(f"{split}: {int(d[TARGET].isna().sum()):,} alvos nulos")
        dup = int(d.duplicated(subset=ID_COLS).sum())
        if dup:
            falhas.append(f"{split}: {dup:,} chaves duplicadas")
        anos_por_split[split] = set(d["ano"].unique().tolist())
        del d

    if colunas_por_split["train"] != colunas_por_split["test"]:
        falhas.append("train e test tem conjuntos de colunas diferentes")

    # Splits temporais nao podem se sobrepor.
    for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
        comum = anos_por_split[a] & anos_por_split[b]
        if comum:
            falhas.append(f"anos em comum entre {a} e {b}: {sorted(comum)}")

    # Toda feature do schema precisa existir no parquet.
    ausentes = [f for f in feats if f not in colunas_por_split["train"]]
    if ausentes:
        falhas.append(f"{len(ausentes)} features do schema ausentes no train: "
                      f"{ausentes[:5]}")

    # A checagem que importa: nada proibido pode estar no schema.
    vazadas = [f for f in feats if f in NUNCA_FEATURE]
    if vazadas:
        falhas.append(f"VAZAMENTO: colunas proibidas no feature_schema {vazadas}")

    # E as colunas de auditoria precisam existir, para o notebook poder
    # exclui-las por nome em vez de adivinhar.
    for c in ("risco_surto_t4_com_vazamento", "threshold_source"):
        if c not in colunas_por_split["train"]:
            avisos.append(f"{c} ausente — auditoria do vazamento fica impossivel")

    # Nenhuma feature pode ser texto: o XGBoost recusa.
    esq = pq.read_schema(pasta / "train.parquet")
    texto = [f.name for f in esq
             if f.name in feats and "string" in str(f.type)]
    if texto:
        falhas.append(f"features de texto no schema: {texto}")

    if resumo.get("n_features") != len(feats):
        falhas.append(f"resumo diz {resumo.get('n_features')} features, "
                      f"schema tem {len(feats)}")

    return falhas, avisos


def main():
    print(f"Validando {RAIZ}\n")
    total_falhas = 0
    for nome, pasta in CONJUNTOS.items():
        falhas, avisos = checar(nome, pasta)
        marca = "OK  " if not falhas else "FALHA"
        print(f"[{marca}] {nome}")
        for a in avisos:
            print(f"         aviso: {a}")
        for f in falhas:
            print(f"         {f}")
        total_falhas += len(falhas)

    tamanho = sum(f.stat().st_size for f in RAIZ.rglob("*") if f.is_file())
    n = sum(1 for f in RAIZ.rglob("*") if f.is_file())
    print(f"\n{n} arquivos, {tamanho / 1024**2:.0f} MB para subir")

    if total_falhas:
        print(f"\n{total_falhas} problema(s). NAO suba.")
        sys.exit(1)
    print("\nPronto para subir.")


if __name__ == "__main__":
    main()

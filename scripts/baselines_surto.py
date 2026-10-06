#!/usr/bin/env python3
"""US-010: baselines do alvo binario de surto (canal endemico, mesorregiao).

F1_positivo de teste de 0,72 na variante 'media' do classificador nao diz
nada sozinho — precisa de algo para bater. Cinco competidores, todos sobre
o mesmo teste e o mesmo rotulo que o classificador usa (lido do mesmo
parquet de scripts/canal_endemico.py, entao nao ha como divergir):

    persistencia         surto(t+4) = surto ja observado em t (o rotulo
                         contemporaneo, que ja sai pronto do canal_endemico)
    sazonal              previsao continua da mesma semana no ano anterior,
                         classificada pelo limiar do canal
    media_movel_4        idem, com a media movel de 4 semanas
    sempre_positivo      surto em toda linha
    sempre_negativo      nunca surto

sempre_positivo e sempre_negativo importam porque a prevalencia de teste e
alta (0,60 na variante 'media'): sempre_positivo tem F1_positivo=0,75 so por
acertar a classe majoritaria, e se essa linha nao aparecer aqui alguem na
banca vai calcula-la.

Os tres primeiros usam a mesma previsao continua de baselines_naive.py,
reclassificada pelo limiar do canal endemico na semana DO ALVO — a mesma
climatologia que definiu o rotulo verdadeiro, para a comparacao ser de
previsao e nao de regra de corte.

Uso:
    python3 scripts/baselines_surto.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines_naive  # noqa: E402
import canal_endemico  # noqa: E402
import metrics_common  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data" / "model_ready" / "mesorregiao_surto"
SAIDA = BASE_DIR / "models" / "baselines_surto_meso"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
VARIANTES = {"media": ("surto_media_t", "surto_media_t4", "limiar_media"),
             "media_2dp": ("surto_media_2dp_t", "surto_media_2dp_t4", "limiar_media_2dp")}


def carregar_com_rotulos():
    """Como baselines_naive.carregar(), mais as colunas do canal endemico.

    prever() so olha para ID_COLS, TARGET e notificacoes — preserva
    qualquer coluna extra do frame sem tocar nela, entao acrescentar as
    colunas de rotulo aqui e suficiente para reusa-la sem modificacao.
    """
    cols = ID_COLS + [TARGET, "notificacoes"] + canal_endemico.ROTULO_COLS
    partes = {}
    for nome in ("train", "val", "test"):
        caminho = DATASET / f"{nome}.parquet"
        disponiveis = {f.name for f in pq.read_schema(caminho)}
        faltando = [c for c in cols if c not in disponiveis]
        if faltando:
            raise KeyError(f"{nome}.parquet sem as colunas {faltando}")
        d = pd.read_parquet(caminho, columns=cols)
        d["split"] = nome
        partes[nome] = d
    return pd.concat(partes.values(), ignore_index=True)


def classificar_pela_previsao(pred_col, teste, canal):
    """Classifica uma previsao continua pelo limiar do canal na semana do alvo.

    E o mesmo limiar que produziu o rotulo verdadeiro (semana_alvo, ja
    presente em ROTULO_COLS) — comparar contra ele, e nao contra um limiar
    recalculado sobre a previsao, e o que torna a comparacao justa.
    """
    out = {}
    m = teste.copy()
    m["se_clim_alvo"] = m["semana_alvo"].clip(upper=52)
    m = m.merge(
        canal.rename(columns={"mesorregiao": "ibge_municipio"}),
        left_on=["ibge_municipio", "se_clim_alvo"],
        right_on=["ibge_municipio", "se_clim"], how="left")
    for nome_var, (_, col_t4, col_limiar) in VARIANTES.items():
        out[nome_var] = (m[pred_col] > m[col_limiar]).astype(int).to_numpy()
    return out, m


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    canal = pd.read_csv(DATASET / "canal_endemico.csv",
                        dtype={"mesorregiao": str})

    serie = baselines_naive.prever(carregar_com_rotulos())
    teste = serie[serie["split"] == "test"].copy()
    print(f"Teste: {len(teste):,} linhas, anos {sorted(teste['ano'].unique())}\n")

    resultados, linhas = {}, []

    # --- persistencia, sazonal, media_movel_4: previsao continua + limiar do canal ---
    continuos = {
        "persistencia": "pred_persistencia",
        "sazonal": "pred_sazonal",
        "media_movel_4": "pred_media_movel_4",
    }
    for nome, col in continuos.items():
        valido = teste[teste[col].notna()].copy()
        cobertura = len(valido) / len(teste) * 100
        decisoes, m = classificar_pela_previsao(col, valido, canal)

        reg = metrics_common.regression_metrics(valido[TARGET], valido[col])
        reg["cobertura_teste_pct"] = round(cobertura, 2)
        resultados[nome] = {"regressao": reg, "classificacao_binaria": {}}
        print(f"{nome:16s} n={reg['n']:,} ({cobertura:.1f}% do teste) | "
              f"R2_orig {reg['R2_orig']:.4f}")

        for nome_var, (_, col_t4, _) in VARIANTES.items():
            yte = m[col_t4].astype(int).to_numpy()
            clf = metrics_common.binary_metrics(
                yte, decisoes[nome_var], y_score=m[col].to_numpy(),
                alvo=col_t4, limiar=None)
            resultados[nome]["classificacao_binaria"][nome_var] = clf
            linhas.append(metrics_common.to_row_binario(
                f"{nome}__{nome_var}", None, clf) | {
                "grupo": nome_var, "cobertura_teste_pct": round(cobertura, 2)})
            print(f"    [{nome_var}] F1_pos {clf['F1_positivo']:.4f} | "
                  f"prevalencia {clf['prevalencia']:.3f}")

    # --- persistencia de rotulo: surto(t+4) = surto ja observado em t ---
    for nome_var, (col_t, col_t4, _) in VARIANTES.items():
        yte = teste[col_t4].astype(int).to_numpy()
        pred = teste[col_t].astype(int).to_numpy()
        clf = metrics_common.binary_metrics(
            yte, pred, y_score=None, alvo=col_t4, limiar=None)
        resultados.setdefault("persistencia_de_rotulo",
                              {"classificacao_binaria": {}})
        resultados["persistencia_de_rotulo"]["classificacao_binaria"][nome_var] = clf
        linhas.append(metrics_common.to_row_binario(
            f"persistencia_de_rotulo__{nome_var}", None, clf) | {
            "grupo": nome_var, "cobertura_teste_pct": 100.0})
        print(f"persistencia_de_rotulo [{nome_var}] F1_pos {clf['F1_positivo']:.4f} "
              f"| prevalencia {clf['prevalencia']:.3f}")

    # --- sempre-positivo / sempre-negativo ---
    for nome, valor in [("sempre_positivo", 1), ("sempre_negativo", 0)]:
        resultados.setdefault(nome, {"classificacao_binaria": {}})
        for nome_var, (_, col_t4, _) in VARIANTES.items():
            yte = teste[col_t4].astype(int).to_numpy()
            pred = np.full(len(yte), valor)
            clf = metrics_common.binary_metrics(
                yte, pred, y_score=None, alvo=col_t4, limiar=None)
            resultados[nome]["classificacao_binaria"][nome_var] = clf
            linhas.append(metrics_common.to_row_binario(
                f"{nome}__{nome_var}", None, clf) | {
                "grupo": nome_var, "cobertura_teste_pct": 100.0})
            print(f"{nome} [{nome_var}] F1_pos {clf['F1_positivo']:.4f} "
                  f"| prevalencia {clf['prevalencia']:.3f}")

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(SAIDA / "comparison.csv", index=False)
    with open(SAIDA / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "data/model_ready/mesorregiao_surto",
                  "baselines": resultados}, f, indent=2, ensure_ascii=False)

    with open(SAIDA / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Baselines do surto binario (US-010)\n\n")
        f.write(f"Dataset: `data/model_ready/mesorregiao_surto` | teste: "
                f"{sorted(teste['ano'].unique().tolist())} | "
                f"{len(teste):,} linhas\n\n")
        f.write(tabela.to_markdown(index=False))
        f.write("\n")

    print(f"\n-> {SAIDA}")


if __name__ == "__main__":
    main()

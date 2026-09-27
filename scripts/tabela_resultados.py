#!/usr/bin/env python3
"""Junta modelos e baselines numa tabela so (FR-4).

A regra do plano e que nenhum resultado de XGBoost seja reportado sem a linha
de baseline ao lado. Este script le o que cada execucao gravou e monta a
tabela unica, para nao existir a possibilidade de publicar uma metade.

Uso:
    python3 scripts/tabela_resultados.py
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ABLACAO = BASE_DIR / "models" / "ablacao"
DOCS = BASE_DIR / "docs"

# Cada grupo compara coisas avaliadas no mesmo teste. Misturar grupos numa
# tabela so compararia numeros de conjuntos de teste diferentes.
GRUPOS = {
    "E1 — recorte 2019-2023, 1.369 municipios, teste 2023": {
        "baselines": "models/baselines/metrics.json",
        "bracos": ["e1_sinan", "e1_sinan_inmet"],
    },
    "E2 — historico completo 2000-2023, 5.565 municipios, teste 2023": {
        "baselines": "models/baselines_e2/metrics.json",
        "bracos": ["e2_sinan"],
    },
    "US-009 — mesmo recorte agregado por mesorregiao (43 series), teste 2023": {
        "baselines": "models/baselines_meso/metrics.json",
        "bracos": ["meso_sinan", "meso_sinan_inmet"],
    },
    "US-009 — mesmo recorte agregado por UF (6 series), teste 2023": {
        "baselines": "models/baselines_uf/metrics.json",
        "bracos": ["uf_sinan", "uf_sinan_inmet"],
    },
}

COLUNAS = ["modelo", "n_teste", "MAE_orig", "RMSE_orig", "R2_orig",
           "MAE_log", "R2_log", "F1_macro", "AUC_macro_ovr"]


def linhas_baseline(caminho):
    p = BASE_DIR / caminho
    if not p.exists():
        return []
    dados = json.loads(p.read_text(encoding="utf-8"))
    saida = []
    for nome, m in dados["baselines"].items():
        r, c = m["regressao"], m["classificacao"]
        saida.append({
            "modelo": f"baseline: {nome}", "n_teste": r["n"],
            "MAE_orig": r["MAE_orig"], "RMSE_orig": r["RMSE_orig"],
            "R2_orig": r["R2_orig"], "MAE_log": r["MAE_log"], "R2_log": r["R2_log"],
            "F1_macro": c["F1_macro"], "AUC_macro_ovr": c.get("AUC_macro_ovr"),
        })
    return saida


def linha_braco(braco):
    p = ABLACAO / braco / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    r, c = d["regressao"], d["classificacao"]
    return {
        "modelo": f"XGBoost: {braco}", "n_teste": r["n"],
        "MAE_orig": r["MAE_orig"], "RMSE_orig": r["RMSE_orig"],
        "R2_orig": r["R2_orig"], "MAE_log": r["MAE_log"], "R2_log": r["R2_log"],
        "F1_macro": c["F1_macro"], "AUC_macro_ovr": c.get("AUC_macro_ovr"),
    }


def inflacoes():
    linhas = []
    for p in sorted(ABLACAO.glob("*/metrics.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        inf = d.get("inflacao_do_vazamento", {})
        linhas.append({
            "braco": d["braco"],
            "F1_macro_sem_vazamento": d["classificacao"]["F1_macro"],
            "F1_macro_com_vazamento": d["classificacao_com_vazamento"]["F1_macro"],
            "inflacao_F1_macro": inf.get("F1_macro"),
            "AUC_sem_vazamento": d["classificacao"].get("AUC_macro_ovr"),
            "AUC_com_vazamento": d["classificacao_com_vazamento"].get("AUC_macro_ovr"),
            "inflacao_AUC": inf.get("AUC_macro_ovr"),
        })
    return pd.DataFrame(linhas)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    partes_md = ["# Resultados: modelos e baselines (FR-4)\n",
                 "MAE e RMSE em casos por municipio-semana. `_orig` e a escala de",
                 "contagem, `_log` e sobre log1p, que e o que o treino otimiza.\n"]
    todas = []

    for titulo, cfg in GRUPOS.items():
        linhas = linhas_baseline(cfg["baselines"])
        for b in cfg["bracos"]:
            linha = linha_braco(b)
            if linha:
                linhas.append(linha)
        if not linhas:
            continue
        t = pd.DataFrame(linhas).reindex(columns=COLUNAS)
        t.insert(0, "grupo", titulo)
        todas.append(t)
        partes_md += [f"\n## {titulo}\n",
                      t.drop(columns=["grupo"]).to_markdown(index=False), ""]

    if not todas:
        print("Nenhum resultado ainda.")
        return

    completo = pd.concat(todas, ignore_index=True)
    completo.to_csv(DOCS / "resultados_modelos_e_baselines.csv", index=False)

    inf = inflacoes()
    if not inf.empty:
        partes_md += ["\n## Inflacao das metricas pelo vazamento no rotulo (US-002)\n",
                      "Quanto a classificacao parecia melhor quando os limiares de",
                      "risco eram calculados sobre o dataset inteiro em vez de so",
                      "sobre o treino.\n",
                      inf.to_markdown(index=False), ""]
        inf.to_csv(DOCS / "inflacao_vazamento.csv", index=False)

    (DOCS / "resultados_modelos_e_baselines.md").write_text(
        "\n".join(partes_md) + "\n", encoding="utf-8")

    print(completo.to_string(index=False))
    print(f"\n-> {DOCS / 'resultados_modelos_e_baselines.md'}")


if __name__ == "__main__":
    main()

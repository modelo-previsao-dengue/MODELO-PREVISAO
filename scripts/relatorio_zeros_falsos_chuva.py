#!/usr/bin/env python3
"""US-001 / FR-2: relatorio antes/depois dos zeros falsos de precipitacao.

A agregacao anterior fazia sum() sobre a chuva horaria. Em pandas a soma de
um grupo vazio e 0.0, nao nulo, entao a semana em que a estacao nao reportou
nada entrava no modelo como "choveu 0 mm", indistinguivel de uma semana seca
de verdade. rain_days e rain_heavy_days sofriam o mesmo, contando (x > 0)
sobre grupo vazio.

O antes e reconstruivel de forma exata sem reprocessar nada: as linhas com
n_valid_rain_hours == 0 sao precisamente as que a logica antiga reportava
como 0.0 e a nova reporta como nulo. O Silver legado no disco cobre so anos
fora da janela, entao esta e a comparacao possivel — e tambem a mais limpa,
porque isola a US-001 sem misturar com a correcao de mapeamento da US-000.

Uso:
    python3 scripts/relatorio_zeros_falsos_chuva.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
SILVER = BASE_DIR / "data" / "inmet" / "silver"
DOCS = BASE_DIR / "docs"

COLS = ["ano_epi", "semana_epidemiologica", "codigo_wmo", "rain_sum_mm",
        "n_valid_rain_hours"]


def main():
    cfg = recorte_config.load()
    DOCS.mkdir(parents=True, exist_ok=True)

    linhas = []
    for ano in cfg["anos_pipeline"]:
        caminho = SILVER / f"weekly_stations_{ano}.parquet"
        if not caminho.exists():
            continue
        d = pd.read_parquet(caminho, columns=COLS)

        falsos = d["n_valid_rain_hours"] == 0
        reais = (~falsos) & (d["rain_sum_mm"] == 0)

        linhas.append({
            "ano": int(ano),
            "estacao_semanas": len(d),
            "zeros_falsos_antes": int(falsos.sum()),
            "pct_zeros_falsos": round(float(falsos.mean() * 100), 2),
            "semanas_secas_reais": int(reais.sum()),
            # Antes, os dois grupos eram indistinguiveis: toda semana sem
            # observacao entrava na conta de "semana sem chuva".
            "pct_das_semanas_sem_chuva_que_eram_falsas": round(float(
                falsos.sum() / max(falsos.sum() + reais.sum(), 1) * 100), 2),
            "chuva_media_mm_antes": round(float(
                d["rain_sum_mm"].fillna(0).mean()), 3),
            "chuva_media_mm_depois": round(float(
                d.loc[~falsos, "rain_sum_mm"].mean()), 3),
        })

    tabela = pd.DataFrame(linhas)
    total = pd.DataFrame([{
        "ano": "TOTAL",
        "estacao_semanas": int(tabela["estacao_semanas"].sum()),
        "zeros_falsos_antes": int(tabela["zeros_falsos_antes"].sum()),
        "pct_zeros_falsos": round(float(
            tabela["zeros_falsos_antes"].sum() / tabela["estacao_semanas"].sum()
            * 100), 2),
        "semanas_secas_reais": int(tabela["semanas_secas_reais"].sum()),
        "pct_das_semanas_sem_chuva_que_eram_falsas": round(float(
            tabela["zeros_falsos_antes"].sum()
            / (tabela["zeros_falsos_antes"].sum()
               + tabela["semanas_secas_reais"].sum()) * 100), 2),
        "chuva_media_mm_antes": None,
        "chuva_media_mm_depois": None,
    }])
    tabela = pd.concat([tabela, total], ignore_index=True)
    tabela.to_csv(DOCS / "relatorio_zeros_falsos_chuva.csv", index=False)

    print(tabela.to_string(index=False))
    print(f"\n-> {DOCS / 'relatorio_zeros_falsos_chuva.csv'}")


if __name__ == "__main__":
    main()

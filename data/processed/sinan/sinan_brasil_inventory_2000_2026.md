# Inventário Completo - SINAN/Dengue Brasil (2000-2026)

Este consolidado reúne a contagem real de registros da série oficial nacional de dengue do SINAN, em escala Brasil inteiro, a partir dos arquivos anuais oficiais.

## Execução utilizada

Comandos executados:

```bash
python3 MODELO-PREVISAO/scripts/sinan_pipeline.py --full-dengue-series --full-brazil
python3 MODELO-PREVISAO/scripts/sinan_full_series_inventory.py --start-year 2025 --end-year 2026 --download-missing
```

## Resultado consolidado

- Escopo: `Brasil inteiro`
- Período: `2000-2026`
- Total de anos consolidados: `27`
- Total de registros confirmados: `33.482.123`
- Ano de maior volume: `2024` com `6.564.924` registros
- Segundo maior volume: `2015` com `2.398.060` registros
- Terceiro maior volume: `2016` com `2.298.020` registros

## Registros por ano

| Ano | Registros |
| --- | ---: |
| 2000 | 172.855 |
| 2001 | 488.590 |
| 2002 | 897.093 |
| 2003 | 416.609 |
| 2004 | 136.867 |
| 2005 | 261.501 |
| 2006 | 411.022 |
| 2007 | 717.097 |
| 2008 | 919.324 |
| 2009 | 600.658 |
| 2010 | 1.381.254 |
| 2011 | 1.150.011 |
| 2012 | 950.180 |
| 2013 | 2.035.119 |
| 2014 | 966.619 |
| 2015 | 2.398.060 |
| 2016 | 2.298.020 |
| 2017 | 518.483 |
| 2018 | 478.880 |
| 2019 | 2.261.956 |
| 2020 | 1.495.117 |
| 2021 | 1.010.359 |
| 2022 | 1.405.095 |
| 2023 | 1.645.956 |
| 2024 | 6.564.924 |
| 2025 | 1.646.533 |
| 2026 | 253.941 |

## Leitura analítica

- A série nacional mostra ciclos relevantes antes de 2024, com destaques para `2013`, `2015`, `2016` e `2019`.
- O ano de `2024` foge claramente do padrão histórico observado no restante da série e domina o volume nacional consolidado.
- O ano de `2026` aparece muito abaixo dos anos anteriores porque a base oficial ainda está em atualização no momento da coleta.

## Arquivos relacionados

- `data/processed/sinan/sinan_brasil_inventory_2000_2026.csv`
- `data/processed/sinan/full_brazil_progress_2000_2015.csv`
- `data/processed/sinan/sinan_brasil_inventory_2025_2026.csv`

# Progresso Parcial - SINAN Dengue Brasil Completo

Este consolidado parcial foi extraído da execução real da pipeline oficial:

```bash
python3 MODELO-PREVISAO/scripts/sinan_pipeline.py --full-dengue-series --full-brazil
```

## Status confirmado

- Escopo: Brasil inteiro
- Fonte: microdados oficiais do SINAN/Dengue
- Série em execução: `2000-2026`
- Anos concluídos até este ponto: `2000-2015`

## Registros nacionais por ano já processado

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

## Leitura analítica parcial

- Total acumulado já confirmado entre `2000` e `2015`: `13.902.859` registros
- Maior volume parcial até aqui: `2015`, com `2.398.060` registros
- Segundo maior volume parcial: `2013`, com `2.035.119` registros
- O bloco `2010-2015` concentra volumes muito superiores aos anos iniciais da série

## Arquivo auxiliar

Os mesmos valores estão em:

- `data/processed/sinan/full_brazil_progress_2000_2015.csv`

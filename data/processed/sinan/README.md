# Entrega SINAN Processado

Esta pasta contem os artefatos finais da task `Coleta e Processamento de Dados - SINAN`.

## Recorte

- Municipio de referencia: Brasilia/DF
- Codigo de referencia: 5300108
- Periodo: 2022-2024
- Registros normalizados: 399.094
- Semanas epidemiologicas: 156

## Principais arquivos

- `recorte_*.parquet/csv`: microdados filtrados e normalizados
- `weekly_features_*.csv`: agregacao semanal pronta para modelagem
- `cluster_*`: artefatos de clusterizacao por intensidade e por perfil clinico
- `feature_ranking*` e `selected_features*`: selecao de atributos
- `quality_*`: validacao e qualidade dos dados
- `relatorio_sinan_*.md`: relatorio consolidado para a monografia

## Reexecucao

```bash
cd MODELO-PREVISAO
python3 scripts/sinan_pipeline.py --skip-download
```
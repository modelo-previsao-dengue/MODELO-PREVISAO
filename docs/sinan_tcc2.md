# Pipeline SINAN - TCC2

Esta etapa substitui o recorte simplificado do TCC1 por uma coleta reprodutível dos microdados oficiais do SINAN.

## Decisão de recorte

O TCC1 documenta o problema como "Distrito Federal", mas o código realmente executado usou:

- `geocode 5300108`
- município de `Brasília`
- período `2022-2024`
- série agregada da API `InfoDengue`

Por isso, a pipeline do TCC2 adota duas decisões:

1. Fazer o download dos arquivos anuais completos do SINAN a partir da fonte oficial.
2. Extrair, como recorte analítico padrão, `Brasília/DF` no período `2022-2024`, para manter comparabilidade com o TCC1.

## Saídas geradas

Ao executar `scripts/sinan_pipeline.py`, o pipeline gera:

- `data/raw/sinan/`: arquivos anuais oficiais `.json.zip`
- `data/processed/sinan/tcc1_scope_summary.json`: resumo do que o TCC1 efetivamente usou
- `data/processed/sinan/recorte_brasilia_2022_2024.parquet`: microdados filtrados
- `data/processed/sinan/normalized_recorte_brasilia_2022_2024.parquet`: cache normalizado para reexecuções rápidas
- `data/processed/sinan/weekly_features_brasilia_2022_2024.csv`: atributos semanais derivados do SINAN
- `data/processed/sinan/cluster_assignments_brasilia_2022_2024.csv`: clusters por intensidade de transmissão
- `data/processed/sinan/cluster_evaluation_brasilia_2022_2024.csv`: avaliação de `k` para intensidade
- `data/processed/sinan/feature_ranking_brasilia_2022_2024.csv`: ranking de atributos para intensidade
- `data/processed/sinan/selected_features_brasilia_2022_2024.json`: atributos selecionados para intensidade
- `data/processed/sinan/cluster_assignments_profile_brasilia_2022_2024.csv`: clusters por perfil clínico-epidemiológico
- `data/processed/sinan/cluster_evaluation_profile_brasilia_2022_2024.csv`: avaliação de `k` para perfil clínico
- `data/processed/sinan/feature_ranking_profile_brasilia_2022_2024.csv`: ranking de atributos para perfil clínico
- `data/processed/sinan/selected_features_profile_brasilia_2022_2024.json`: atributos selecionados para perfil clínico
- `data/processed/sinan/relatorio_sinan_brasilia_2022_2024.md`: relatório consolidado

## Modos analíticos

A pipeline roda duas leituras complementares:

1. `intensity`: clusteriza semanas usando também `NOTIFICACOES`, útil para separar baixa transmissão versus ondas epidêmicas.
2. `profile`: remove `NOTIFICACOES` do espaço de cluster, forçando a separação por sintomas, critérios diagnósticos, sinais de alarme e gravidade.

## Execução

```bash
cd MODELO-PREVISAO
python3 scripts/sinan_pipeline.py
```

Para reaproveitar a mesma pipeline com outro recorte:

```bash
python3 scripts/sinan_pipeline.py --years 2020 2021 2022 --uf GO --municipality-code 5208707 --municipality-name Goiania
```

Para processar a série oficial completa de dengue do SINAN publicada no portal, em escala Brasil:

```bash
python3 scripts/sinan_pipeline.py --full-dengue-series --full-brazil
```

Observação:

- conforme verificado em `28/04/2026`, o portal `Sinan/Dengue` do Dados Abertos do SUS publica arquivos anuais de `2000` até `2026`
- os anos mais recentes permanecem sujeitos a atualização, pois o conjunto é atualizado semanalmente

Para um consolidado anual leve, sem materializar todos os microdados nacionais em memória:

```bash
python3 scripts/sinan_full_series_inventory.py --download-missing
```

Esse comando gera um inventário auditável com a contagem real de registros por ano em `data/processed/sinan/`.

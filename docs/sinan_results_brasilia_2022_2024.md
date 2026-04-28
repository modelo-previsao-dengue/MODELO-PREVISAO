# Resultados SINAN - Brasília/DF (2022-2024)

Este documento resume os resultados finais da task `Coleta e Processamento de Dados - SINAN`.

## Escopo final

- Fonte: microdados oficiais do SINAN/OpenDataSUS
- Recorte de referência: `Brasília/DF`
- Código de referência: `5300108`
- Período: `2022-2024`

## Volumetria

- Registros extraídos do recorte: `399.094`
- Semanas epidemiológicas após agregação: `156`
- Cobertura temporal: `2022-01-02` até `2024-12-28`

Distribuição anual:

- `2022`: `73.245` registros
- `2023`: `43.127` registros
- `2024`: `283.712` registros

## Qualidade dos dados

- Linhas duplicadas exatas no recorte final: `0`
- Semanas duplicadas: `0`
- Missing nos campos-chave monitorados:
  - `DT_NOTIFIC`: `0%`
  - `DT_SIN_PRI`: `0%`
  - `SEM_PRI`: `0%`
  - `CS_SEXO`: `0%`
  - `CS_RACA`: `0%`
  - `CLASSI_FIN`: `0%`
  - `CRITERIO`: `0%`
  - `HOSPITALIZ`: `0%`
  - `EVOLUCAO`: `0%`

## Clusterização

### 1. Intensidade de transmissão

- Melhor `k`: `2`
- Silhouette: `0.125845`

Clusters:

- `baixa_transmissao`: `138` semanas
- `transicao`: `18` semanas

Principais atributos:

- `NOTIFICACOES`
- `dor_retro_flag`
- `febre_flag`
- `cefaleia_flag`
- `nausea_flag`
- `laco_flag`
- `dor_costas_flag`
- `artrite_flag`

### 2. Perfil clínico-epidemiológico

- Melhor `k`: `6`
- Silhouette: `0.106929`

Clusters:

- `perfil_basal`
- `perfil_intermediario`
- `perfil_alterado`
- `perfil_agudo`
- `perfil_extremo`
- `perfil_raro`

Principais atributos:

- `dor_retro_flag`
- `exantema_flag`
- `caso_confirmado_provavel`
- `raca_parda`
- `caso_inconclusivo`
- `cefaleia_flag`
- `mialgia_flag`
- `nausea_flag`

## Artefatos operacionais

Os artefatos gerados pela pipeline ficam em `data/processed/sinan/`, com destaque para:

- `relatorio_sinan_brasilia_2022_2024.md`
- `quality_summary_brasilia_2022_2024.json`
- `artifact_manifest_brasilia_2022_2024.json`
- `selected_features_brasilia_2022_2024.json`
- `selected_features_profile_brasilia_2022_2024.json`

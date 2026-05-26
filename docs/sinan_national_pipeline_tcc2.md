# Trilha SINAN Nacional - Contrato Oficial TCC2

## Status do legado encontrado

O estado real do repositório mostrou dois blocos distintos:

1. Um bloco confiável e reutilizável para o recorte de `Brasilia/DF` em `2022-2024`, com microdados normalizados, agregação semanal, clusterização e seleção de atributos.
2. Um bloco nacional inconsistente para o TCC2, porque os artefatos `brasil_2000_2026` existentes foram consolidados em `Brasil-semana`, e não em `municipio-semana`.

Essa divergência torna os artefatos nacionais legados inadequados como camada oficial de modelagem do TCC2, embora continuem válidos como evidência histórica do trabalho já executado.

## Decisão metodológica oficial

Esta trilha congela o contrato nacional do SINAN para o TCC2 com as regras abaixo.

### Unidade analítica principal

- `ibge_municipio + ano_semana`

### Chave primária oficial

- `ibge_municipio`
- `ano_semana`

### Janela temporal oficial

- série nacional disponível localmente nos brutos oficiais `DENGBR00` até `DENGBR26`
- janela operacional padrão: `2000` até `2026`
- anos recentes devem ser tratados como série sujeita a atualização do portal oficial

### Regra temporal oficial

- a métrica central da trilha é `notificacoes`, agregada por semana epidemiológica de notificacao
- prioridade de derivação temporal: `SEM_NOT` -> `DT_NOTIFIC` -> `SEM_PRI` -> `DT_SIN_PRI`
- semanas derivadas só são aceitas quando plausíveis em relação ao ano-fonte do arquivo (`source_year +/- 1`)
- spillovers ISO limítrofes entre anos consecutivos são aceitos, por exemplo `199952` no arquivo de `2000` ou `200201` no arquivo de `2001`
- `SEM_PRI` e `DT_SIN_PRI` são mantidos apenas como fallback controlado, para evitar contaminar a série oficial com semanas antigas, ambíguas ou inconsistentes

### Regra territorial oficial

- a trilha usa o código de residência do caso como referência primária
- quando o microdado traz código municipal em formato de 6 dígitos, a normalização converte para o código oficial de 7 dígitos do IBGE por reconciliação com a tabela oficial de municípios
- códigos não reconciliáveis são registrados em relatório de exceções e descartados da camada oficial
- exceção documentada: registros do Distrito Federal com codificação territorial interna não padronizada podem ser reconciliados para `5300108`

### Convenção de nomes

- `snake_case`
- identificadores em minúsculas
- proporções prefixadas com `prop_`
- contagens prefixadas com `qt_` quando necessário
- labels prefixadas com `label_`

## Contrato das camadas

### Bronze

Objetivo:

- preservar o bruto oficial anual do SINAN/Dengue
- registrar inventário, checksum, tamanho, origem e cobertura temporal por arquivo

Regras:

- sem transformação semântica do conteúdo oficial
- arquivos físicos brutos permanecem em `data/raw/sinan`
- a camada bronze oficial do TCC2 referencia esses arquivos por manifesto e checksum

### Silver

Objetivo:

- consolidar a série nacional em `municipio-semana observado`

Regras:

- normalização de datas, tipos e chaves territoriais
- derivação de `ano_semana` orientada por semana epidemiológica de notificacao
- remoção de duplicidades exatas no nível de registro canônico
- descarte documentado de registros sem semana epidemiológica válida
- descarte documentado de registros sem município reconciliável
- uma linha por `ibge_municipio + ano_semana` com métricas agregadas

### Gold

Objetivo:

- disponibilizar a base final pronta para modelagem e alertas

Regras:

- expansão para grade temporal densa por município nas semanas observadas nacionalmente
- semanas sem notificações são mantidas com `notificacoes = 0`
- features temporais calculadas por município em ordem cronológica
- índices clínico-epidemiológicos agregados calculados sobre a silver
- labels auxiliares geradas para semanas críticas

## Versionamento

- uma versão oficial por execução consolidada
- slug padrão: `sinan_tcc2_v2`
- `sinan_tcc2_v1` deve ser tratado como execucao interrompida e invalida para uso analitico final
- datasets oficiais da versão são publicados em:
  - `data/sinan/bronze/<versao>`
  - `data/sinan/silver/<versao>`
  - `data/sinan/gold/<versao>`
  - `data/sinan/governance/<versao>`
  - `data/sinan/serving/<versao>`

## Lineage

Cada dataset oficial deve ter:

- `run_manifest.json`
- `schema_*.json`
- `lineage_*.json`
- `quality_summary_*.json`
- `discard_report_*.csv`

## Reexecução incremental

- bronze: reusa arquivos anuais existentes e recalcula manifestos
- silver: pode ser reprocessada por ano
- gold: depende do conjunto silver oficial da versão e deve ser reexecutada após qualquer mudança estrutural na silver

## Regra de descarte

Um registro só entra na camada oficial se satisfizer simultaneamente:

- arquivo bruto oficial identificado
- semana epidemiológica válida ou derivável
- município reconciliável

Casos excluídos devem ser documentados em:

- `discard_report_silver.csv`
- `silver_quality_summary.json`
- `final_run_report.md`

## Relação com os artefatos legados

Itens preservados como evidência histórica:

- `data/processed/sinan/recorte_brasilia_2022_2024.*`
- `data/processed/sinan/weekly_features_brasilia_2022_2024.csv`
- `data/processed/sinan/weekly_features_brasil_2000_2026.csv`
- relatórios e evidências em `TCC2-DOCS`

Classificação oficial:

- recorte Brasília: `concluído e confiável para evidência local`
- nacional Brasil-semana legado: `concluído mas inconsistente para o contrato oficial do TCC2`
- nova trilha `municipio-semana`: `camada oficial`

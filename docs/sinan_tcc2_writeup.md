# Material TCC2 - Trilha SINAN

- Versao oficial: `sinan_tcc2_v2`

## Metodologia executada

A trilha SINAN do TCC2 foi consolidada como pipeline nacional em `municipio-semana`, com preservacao do bronze oficial anual, agregacao observada na silver e densificacao da serie na gold para modelagem e alertas.

Fluxo executado:

1. Inventario e checksum dos arquivos anuais oficiais do SINAN/Dengue.
2. Parsing incremental dos JSON anuais, sem materializar toda a serie em memoria.
3. Reconciliacao territorial dos codigos municipais com a tabela oficial do IBGE.
4. Derivacao e validacao de semana epidemiologica por registro, priorizando a semana de notificacao.
5. Remocao de duplicidades exatas por assinatura canonica do registro.
6. Agregacao para `ibge_municipio + ano_semana` na camada silver.
7. Densificacao temporal por municipio na camada gold, com `notificacoes = 0` em semanas sem casos.
8. Engenharia de atributos temporais e clinico-epidemiologicos.
9. Clusterizacao exploratoria em amostra ativa da gold e selecao de atributos.
10. Geracao de schema, lineage, missingness, discard report e camada operacional.

## Contratos principais

- Silver: `municipio-semana observado`
- Gold: `municipio-semana denso`
- Chave primaria oficial: `ibge_municipio, ano_semana`

## Resultados consolidados

- Notificacoes aproveitadas na silver: `33.411.887`
- Linhas da silver observada: `1.841.975`
- Municipios cobertos na silver: `5.565`
- Linhas da gold densa: `7.665.428`
- Janela final: `199952` ate `202616`

## Principais semanas nacionais

- 202415: 420.542 notificacoes
- 202412: 415.450 notificacoes
- 202414: 402.295 notificacoes
- 202416: 391.794 notificacoes
- 202413: 378.153 notificacoes
- 202411: 375.131 notificacoes
- 202417: 363.804 notificacoes
- 202410: 352.500 notificacoes
- 202419: 324.471 notificacoes
- 202409: 321.834 notificacoes
# Texto-Base TCC2 - Trilha SINAN

- Versao oficial: `sinan_tcc2_v2`

## Metodologia

A trilha SINAN do TCC2 foi consolidada como uma pipeline nacional reproduzivel e auditavel em nivel municipio-semana. O processo partiu dos arquivos anuais oficiais de dengue do SINAN, preservados em Bronze, seguiu para uma Silver com reconciliacao territorial, normalizacao de tipos, remocao de duplicidades exatas e agregacao por `ibge_municipio + ano_semana`, e culminou em uma Gold densa com atributos temporais e clinico-epidemiologicos prontos para modelagem.

A regra temporal oficial foi congelada sobre semana epidemiologica de notificacao. A derivacao priorizou `SEM_NOT`, depois `DT_NOTIFIC`, e somente usou `SEM_PRI` ou `DT_SIN_PRI` como fallback controlado, para evitar contaminar a serie oficial com semanas antigas, ambiguas ou semanticamente inadequadas para a contagem de notificacoes.

## Execucao real

Foram inventariados `27` arquivos anuais oficiais, cobrindo a janela operacional `2000-2026`. A camada Silver aproveitou `33.411.887` notificacoes validas, distribuidas em `1.841.975` linhas observadas de municipio-semana e `5.565` municipios unicos.

A camada Gold produziu `7.665.428` linhas densas para consumo analitico e preditivo. A governanca final incluiu schema versionado, lineage, quality summaries, missingness, discard report, manifesto de execucao e camada operacional para armazenamento em objetos e carga em PostgreSQL/Supabase.

## Resultados nacionais

O ano com maior volume agregado de notificacoes na serie foi 2024, com 6.560.720 notificacoes consolidadas na camada gold.
A maior semana nacional observada foi 202415, com 420.542 notificacoes somadas entre municipios.

A analise exploratoria foi complementada por clusterizacao de semanas e selecao de atributos, produzindo regimes exploratorios e rankings reproduziveis para apoiar a etapa de modelagem preditiva.

## Limitacoes

- a serie depende da qualidade e estabilidade dos microdados oficiais publicados pelo SINAN/OpenDataSUS
- anos recentes podem ser atualizados no portal oficial apos a execucao desta versao
- parte das variaveis clinicas apresenta ausencias estruturais ou preenchimento heterogeneo entre anos
- a camada oficial nao incorpora, nesta etapa, variaveis climaticas ou socioambientais exogenas

## Ameacas a validade

- divergencias historicas de preenchimento entre anos podem afetar comparabilidade longitudinal
- subnotificacao, atraso de digitacao e inconsistencias locais podem impactar o sinal epidemiologico observado
- a densificacao da Gold preserva semanas sem notificacao como zero, o que e adequado para modelagem temporal, mas exige interpretacao cuidadosa em municipios com baixa cobertura historica

## Contribuicoes da trilha SINAN

- congelamento de um contrato oficial `municipio-semana` para a frente epidemiologica do TCC2
- separacao profissional em Bronze, Silver, Gold, governanca e camada operacional
- producao de features temporais, clinicas, labels auxiliares, clusterizacao e feature selection em uma unica trilha reproduzivel
- preparacao da base para integracao futura com a trilha exogena e para exposicao por API

## Relacao com a modelagem preditiva

A camada Gold final foi desenhada para alimentar modelos de previsao de surtos de dengue em nivel municipal, com suporte a lags, medias moveis, aceleracao, sazonalidade e indicadores clinico-epidemiologicos agregados. Essa estrutura tambem favorece a construcao de labels auxiliares para semanas criticas e uma futura camada de alertas.

## Divergencias que devem ser refletidas no texto final

- os artefatos nacionais legados em `data/processed/sinan` nao sao a camada oficial final, porque foram consolidados em `Brasil-semana`
- a escrita do TCC2 deve substituir referencias a nacionalizacao futura por descricao da pipeline nacional efetivamente executada
- a descricao da fonte epidemiologica principal deve ser alinhada ao SINAN/OpenDataSUS, e nao ao uso anterior de InfoDengue como recorte comparativo do TCC1

## Pontos de qualidade a citar

- duplicatas exatas removidas na Silver: `70.217`
- descartes por semana invalida: `0`
- descartes por municipio invalido: `19`

Campos com maior missingness na Silver:

- prop_con_fhd (99.28%)
- prop_evidencia (99.13%)
- prop_laco_n (98.82%)
- prop_grav_mioc (98.71%)
- prop_grav_ast (98.71%)
- prop_grav_orgao (98.71%)
- prop_grav_sang (98.71%)
- prop_grav_metro (98.71%)

Campos com maior missingness na Gold:

- ano (0.00%)
- ano_semana (0.00%)
- atraso_notificacao_medio_dias (0.00%)
- ibge_municipio (0.00%)
- idade_media_anos (0.00%)
- indice_alarme (0.00%)
- indice_carga_clinica (0.00%)
- indice_comorbidades (0.00%)
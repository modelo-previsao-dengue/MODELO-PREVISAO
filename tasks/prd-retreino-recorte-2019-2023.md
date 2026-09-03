# PRD: Retreino com Recorte 2019-2023 e Correcao da Pipeline Climatica

## Introduction

Auditoria da base integrada (7.704.766 linhas, 2000-2026) identificou tres defeitos que invalidam parcialmente o resultado central do TCC2 — o achado de que dados climaticos do INMET pioram o modelo (SINAN-only R2=0.42 vs SINAN+INMET R2=0.31). Antes de aceitar esse resultado como cientifico, e preciso descartar que ele seja artefato de bug.

Este PRD cobre: (1) correcao dos tres defeitos, (2) reprocessamento da pipeline para a janela util, (3) recorte espaco-temporal para 6 UFs x 2019-2023, (4) redesenho do split temporal, (5) retreino com ablacao limpa, e (6) teste da hipotese de granularidade espacial.

**Escopo do recorte**: DF, ES, GO, RJ, MG, SP no periodo 2019-2023, filtrado a municipios com >=80% de semanas com clima completo.

**Dimensionamento**: 423.416 linhas, 1.616 municipios, cobertura INMET 95.7% (contra 7.7M linhas e 74.6% de cobertura no dataset atual).

## Goals

- Eliminar os zeros falsos de precipitacao que contaminam 14.6% das linhas da janela
- Eliminar o vazamento temporal na definicao das classes de risco
- Estender os lags climaticos ate 12 semanas, onde o sinal efetivamente esta
- Produzir um dataset de treino com cobertura climatica >=95%, viabilizando Optuna e SHAP sem subsampling
- Reexecutar a ablacao SINAN-only vs SINAN+INMET em condicoes controladas e responder se o clima ajuda
- Estabelecer baselines ingenuos como referencia obrigatoria de comparacao
- Testar se a granularidade municipal destroi o sinal climatico regional

## Evidencia que motiva este trabalho

| Achado | Evidencia | Impacto |
|---|---|---|
| Zeros falsos de chuva | 10.372 estacao-semanas de 2021 com `n_valid_hours == 0`; 99.7% delas com `rain_sum_mm == 0.0`. `rain_sum_mm` e 100% nao-nulo em todos os anos do Silver | 212.391 linhas (14.6%) da janela 2019-2023 com valor confiante e errado |
| Vazamento no rotulo de risco | `compute_risk_class()` calcula percentis p50/p75/p90 sobre o dataframe inteiro antes do split | AUC de 0.84 esta inflado |
| Lags climaticos truncados | `LAG_PERIODS = [1,2,4,8]`, mas o pico da correlacao clima-casos esta em lag 9-12 em 11 de 12 UFs (umidade) | Perda mediana de 8.1% na chuva, ate 26.2% em SC |
| Sinal climatico existe no agregado | Correlacao de Spearman umidade-casos: SP 0.777 (lag 10), GO 0.774 (lag 9), MG 0.770 (lag 10), DF 0.743 (lag 11) | Contradiz o SHAP, que colocou clima em 143o-153o de 161 no nivel municipio |
| Cobertura real != cobertura aparente | Chuva aparenta 90.9% na janela; cobertura real (3 variaveis) e 76.3% | Decisao de janela estava baseada em numero inflado |

## User Stories

### US-001: Corrigir zeros falsos de precipitacao

**Description:** Como pesquisador, quero que semanas sem observacao meteorologica apareçam como ausentes e nao como "choveu 0 mm", para que o XGBoost trate o dado faltante com seu mecanismo nativo de missing em vez de aprender um valor errado.

**Acceptance Criteria:**
- [ ] `scripts/inmet_weekly_aggregate.py` passa a contar horas validas de precipitacao separadamente (`n_valid_rain_hours=("precipitacao_mm", "count")`), independente de `n_valid_hours` que hoje conta `temp_inst_c`
- [ ] `rain_sum_mm` retorna NaN quando nao ha nenhuma observacao valida na semana (via `min_count=1` ou mascara pos-agregacao)
- [ ] `rain_days` e `rain_heavy_days` recebem o mesmo tratamento — hoje contam `(x > 0).sum()` sobre grupos vazios e retornam 0
- [ ] Teste de regressao: no Silver de 2021, nenhuma linha com `n_valid_rain_hours == 0` tem `rain_sum_mm` nao-nulo
- [ ] Relatorio antes/depois: numero de linhas que mudaram de 0.0 para NaN, por ano e por UF

### US-002: Corrigir vazamento temporal no rotulo de risco

**Description:** Como pesquisador, quero que os limiares de classificacao de risco sejam derivados apenas do conjunto de treino, para que a metrica de classificacao seja honesta.

**Acceptance Criteria:**
- [ ] `compute_risk_class()` em `scripts/prepare_model_dataset.py` passa a receber o conjunto de treino como parametro para calcular os percentis
- [ ] Os limiares por municipio sao calculados so no treino e aplicados a val e test
- [ ] Municipios presentes em val/test mas ausentes no treino recebem fallback documentado (percentil da UF ou exclusao — decidir e registrar)
- [ ] Os limiares sao persistidos em `data/model_ready/risk_thresholds.csv` para auditoria
- [ ] Relatorio: AUC e F1 antes e depois da correcao, para dimensionar a inflacao

### US-003: Estender lags climaticos ate 12 semanas

**Description:** Como pesquisador, quero que as features climaticas cubram o horizonte onde a correlacao com casos e maxima, e que sejam simetricas as features epidemiologicas.

**Acceptance Criteria:**
- [ ] `LAG_PERIODS` em `scripts/inmet_municipal_features.py` passa de `[1,2,4,8]` para `[1,2,4,8,12]`
- [ ] Medias moveis climaticas passam a incluir janelas de 8 e 12 semanas, alem da atual de 4
- [ ] O catalogo `inmet_feature_catalog.csv` e regerado refletindo as novas features
- [ ] Documentado no PRD/README que as features climaticas agora espelham as epidemiologicas (lag ate 12, movel ate 12)

### US-004: Reprocessar a pipeline para a janela util

**Description:** Como pesquisador, quero reprocessar apenas os anos necessarios, para nao pagar as 12-15h da pipeline completa.

**Acceptance Criteria:**
- [ ] Reprocessamento cobre 2018-2024, nao 2000-2026
- [ ] Warm-up: 2018 e processado para que os lags de 12 semanas das primeiras semanas de 2019 nao fiquem NaN
- [ ] Cool-down: inicio de 2024 e processado para que o target `shift(-4)` das ultimas semanas de 2023 exista
- [ ] O recorte final entregue e 2019-2023; 2018 e 2024 existem apenas como margem
- [ ] Camadas regeradas: INMET Silver -> INMET Gold -> integrado -> model_ready
- [ ] `coverage_by_year.csv` e regerado e passa a reportar cobertura por variavel (chuva, temperatura, umidade, completo), nao apenas uma

### US-005: Aplicar o recorte espaco-temporal

**Description:** Como pesquisador, quero treinar sobre o subconjunto com cobertura climatica alta, para que a ablacao clima vs sem-clima nao seja confundida por dados faltantes.

**Acceptance Criteria:**
- [ ] Novo parametro de recorte configuravel (UFs e intervalo de anos), nao hardcoded
- [ ] UFs do recorte: DF, ES, GO, RJ, MG, SP
- [ ] Filtro municipal: manter apenas municipios com >=80% de semanas com as 3 variaveis climaticas presentes
- [ ] O filtro municipal e **recalculado apos** a correcao da US-001, nao reaproveitado da auditoria
- [ ] Relatorio do recorte: linhas, municipios, cobertura, notificacoes, por UF
- [ ] Alvo dimensional aproximado: ~420K linhas, ~1.6K municipios, cobertura >=95%

**Justificativa da selecao de UFs** (auditoria da janela 2019-2023):

| UF | INMET | Semanas c/ caso | Munis c/ serie usavel | Atraso (dias) | Conf. lab | Veredito |
|---|---|---|---|---|---|---|
| DF | 99.6% | 100.0% | 100.0% | 6.4 | 78.4% | aprova |
| ES | 90.1% | 42.8% | 89.7% | 4.0 | 48.3% | aprova |
| GO | 92.4% | 45.5% | 86.2% | 4.0 | 54.8% | aprova |
| RJ | 93.3% | 41.5% | 75.0% | 4.6 | 56.3% | aprova |
| MG | 93.8% | 29.5% | 55.7% | 3.4 | 47.0% | aprova (mais fraco) |
| SP | 84.2% | 52.4% | 89.0% | 4.0 | 66.9% | aprova |
| SC | 85.6% | 13.3% | 20.5% | 4.0 | 79.4% | **reprova — SINAN fraco** |
| TO | 80.1% | 22.5% | 44.6% | 4.0 | 40.7% | **reprova — SINAN fraco** |
| PB | 78.4% | 20.2% | 38.1% | 3.8 | 36.7% | **reprova — SINAN fraco** |
| PR | 75.3% | 38.5% | 70.7% | 3.0 | 58.4% | ressalva — INMET no limite |
| BA | 76.7% | 28.2% | 62.4% | 4.0 | 36.4% | ressalva — INMET + conf. lab |
| PE | 78.5% | 35.9% | 73.5% | 4.0 | 28.0% | ressalva — conf. lab baixa |

### US-006: Redesenhar o split temporal

**Description:** Como pesquisador, quero um split coerente com a nova janela, porque o split atual (`train <= 2019`) deixaria o treino com um unico ano.

**Acceptance Criteria:**
- [ ] Split parametrizado, nao hardcoded em `prepare_model_dataset.py:85-87`
- [ ] Split do recorte: train 2019-2021, val 2022, test 2023
- [ ] Volumetria esperada: train ~300K (60%), val ~100K (20%), test ~101K (20%)
- [ ] Documentado explicitamente que 19 anos de historico epidemiologico foram trocados por cobertura climatica, e que isso penaliza o braco SINAN-only

### US-007: Baselines ingenuos

**Description:** Como banca, quero saber se o modelo supera a previsao trivial, porque o SHAP mostra `notificacoes_lag_1` dominando por 23x e ha risco real de o XGBoost estar reproduzindo persistencia.

**Acceptance Criteria:**
- [ ] Baseline de persistencia: `casos_t+4 = casos_t`
- [ ] Baseline sazonal-naive: `casos_t+4 = casos da mesma semana epidemiologica do ano anterior`
- [ ] Ambos avaliados no mesmo conjunto de teste e com as mesmas metricas dos modelos
- [ ] Resultados em `models/baselines/` com o mesmo formato de `metrics.json` dos demais
- [ ] Nenhum resultado de XGBoost e reportado sem a linha de baseline ao lado

### US-008: Retreinar e reexecutar a ablacao

**Description:** Como pesquisador, quero responder se o clima ajuda, em condicoes controladas, sobre as mesmas linhas nos dois bracos.

**Acceptance Criteria:**
- [ ] **E1 — ablacao limpa**: SINAN-only vs SINAN+INMET, mesmas linhas do recorte 2019-2023
- [ ] **E2 — referencia de producao**: SINAN-only sobre o historico completo, para nao perder os 27 anos
- [ ] E1 e E2 sao reportados separadamente e a diferenca entre eles e explicada no texto
- [ ] Optuna roda sem subsampling (hoje usa 500K de 5.8M)
- [ ] SHAP roda sem subsampling (hoje usa 50K)
- [ ] Toda tabela de resultado reporta R2_log e R2_orig lado a lado, mais MAE, com a escala no cabecalho
- [ ] Classificacao reavaliada com os limiares sem vazamento (US-002)
- [ ] Comparacao antes/depois documentada em `docs/`

### US-009: Testar a hipotese de granularidade espacial

**Description:** Como pesquisador, quero saber se a atribuicao estacao->municipio destroi o sinal climatico, porque o clima correlaciona 0.74-0.78 com casos no agregado por UF mas aparece como irrelevante no SHAP municipal.

**Acceptance Criteria:**
- [ ] Dataset agregado por mesorregiao construido a partir do mesmo recorte
- [ ] Ablacao SINAN-only vs SINAN+INMET repetida na granularidade de mesorregiao
- [ ] Comparacao das tres granularidades: municipio, mesorregiao, UF
- [ ] Conclusao documentada: se o clima ganha importancia em granularidade mais grossa, isso e contribuicao metodologica e nao resultado negativo

## Functional Requirements

1. Nenhum script pode assumir janela ou UFs fixas; recorte e split vem de configuracao
2. Toda correcao de bug produz um relatorio antes/depois quantificado
3. Metricas sempre reportadas com a escala explicita (log ou original)
4. Baselines sempre presentes na mesma tabela dos modelos
5. Artefatos de modelo salvos com versionamento que permita reproducao (ver Non-Goals)

## Non-Goals

- Migrar a fonte climatica para ERA5-Land ou CHIRPS (decisao pendente com o orientador; registrado como trabalho futuro)
- Implementar modelos alem do XGBoost (LSTM, INLA, LightGBM) — depende da decisao de escopo da reunião
- Reprocessar anos fora de 2018-2024
- Corrigir a API e o dashboard (`models/nacional/*.ubj` ausentes) — trabalho separado
- Replicar Granger e STL na escala nacional

## Technical Considerations

- **Warm-up e cool-down**: processar 2018-2024, entregar 2019-2023. Sem isso, lags de 12 semanas e target `shift(-4)` produzem NaN silenciosos nas bordas
- **Ordem de dependencia**: a correcao US-001 muda a definicao de cobertura, portanto o filtro municipal da US-005 so pode ser calculado depois
- **Memoria**: com ~420K linhas o dataset cabe folgado; as adaptacoes de subsampling feitas para 16GB deixam de ser necessarias
- **Comparabilidade**: E1 e E2 nao sao comparaveis entre si; qualquer diferenca de desempenho mistura efeito de recorte com efeito de clima
- **Fallback de limiares**: municipios ausentes no treino precisam de regra explicita na US-002

## Success Metrics

- Cobertura climatica do dataset de treino >= 95%
- Zero linhas com `rain_sum_mm` preenchido e `n_valid_rain_hours == 0`
- Ablacao E1 conclusiva: SINAN+INMET supera ou nao supera SINAN-only, com as mesmas linhas e sem zeros falsos
- Baselines documentados e superados pelo modelo (ou a diferenca explicada)
- AUC de classificacao reportado sem vazamento

## Open Questions

1. Municipios em val/test ausentes do treino: excluir ou usar percentil da UF como fallback?
2. E2 (historico completo) usa quais anos de split, ja que nao esta limitado pela cobertura climatica?
3. A agregacao por mesorregiao (US-009) usa media simples do clima ou ponderada por populacao?
4. Se a ablacao E1 continuar desfavoravel ao clima, isso vira o resultado principal do TCC ou dispara a migracao para ERA5?

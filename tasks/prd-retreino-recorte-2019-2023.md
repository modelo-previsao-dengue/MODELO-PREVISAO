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
- [x] `scripts/inmet_weekly_aggregate.py` passa a contar horas validas de precipitacao separadamente (`n_valid_rain_hours=("precipitacao_mm", "count")`), independente de `n_valid_hours` que hoje conta `temp_inst_c`
- [x] `rain_sum_mm` retorna NaN quando nao ha nenhuma observacao valida na semana (via `min_count=1` ou mascara pos-agregacao)
- [x] `rain_days` e `rain_heavy_days` recebem o mesmo tratamento — hoje contam `(x > 0).sum()` sobre grupos vazios e retornam 0
- [x] Teste de regressao: no Silver de 2021, nenhuma linha com `n_valid_rain_hours == 0` tem `rain_sum_mm` nao-nulo
- [x] Relatorio antes/depois: numero de linhas que mudaram de 0.0 para NaN, por ano e por UF

### US-002: Corrigir vazamento temporal no rotulo de risco

**Description:** Como pesquisador, quero que os limiares de classificacao de risco sejam derivados apenas do conjunto de treino, para que a metrica de classificacao seja honesta.

**Acceptance Criteria:**
- [x] `compute_risk_class()` em `scripts/prepare_model_dataset.py` passa a receber o conjunto de treino como parametro para calcular os percentis
- [x] Os limiares por municipio sao calculados so no treino e aplicados a val e test
- [x] Municipios presentes em val/test mas ausentes no treino recebem fallback documentado (percentil da UF ou exclusao — decidir e registrar)
- [x] Os limiares sao persistidos em `data/model_ready/risk_thresholds.csv` para auditoria
- [x] Relatorio: AUC e F1 antes e depois da correcao, para dimensionar a inflacao

### US-003: Estender lags climaticos ate 12 semanas

**Description:** Como pesquisador, quero que as features climaticas cubram o horizonte onde a correlacao com casos e maxima, e que sejam simetricas as features epidemiologicas.

**Acceptance Criteria:**
- [x] `LAG_PERIODS` em `scripts/inmet_municipal_features.py` passa de `[1,2,4,8]` para `[1,2,4,8,12]`
- [x] Medias moveis climaticas passam a incluir janelas de 8 e 12 semanas, alem da atual de 4
- [x] O catalogo `inmet_feature_catalog.csv` e regerado refletindo as novas features
- [x] Documentado no PRD/README que as features climaticas agora espelham as epidemiologicas (lag ate 12, movel ate 12)

### US-004: Reprocessar a pipeline para a janela util

**Description:** Como pesquisador, quero reprocessar apenas os anos necessarios, para nao pagar as 12-15h da pipeline completa.

**Acceptance Criteria:**
- [x] Reprocessamento cobre 2018-2024, nao 2000-2026
- [x] Warm-up: 2018 e processado para que os lags de 12 semanas das primeiras semanas de 2019 nao fiquem NaN
- [x] Cool-down: inicio de 2024 e processado para que o target `shift(-4)` das ultimas semanas de 2023 exista
- [x] O recorte final entregue e 2019-2023; 2018 e 2024 existem apenas como margem
- [x] Camadas regeradas: INMET Silver -> INMET Gold -> integrado -> model_ready
- [x] `coverage_by_year.csv` e regerado e passa a reportar cobertura por variavel (chuva, temperatura, umidade, completo), nao apenas uma

### US-005: Aplicar o recorte espaco-temporal

**Description:** Como pesquisador, quero treinar sobre o subconjunto com cobertura climatica alta, para que a ablacao clima vs sem-clima nao seja confundida por dados faltantes.

**Acceptance Criteria:**
- [x] Novo parametro de recorte configuravel (UFs e intervalo de anos), nao hardcoded
- [x] UFs do recorte: DF, ES, GO, RJ, MG, SP
- [x] Filtro municipal: manter apenas municipios com >=80% de semanas com as 3 variaveis climaticas presentes
- [x] O filtro municipal e **recalculado apos** a correcao da US-001, nao reaproveitado da auditoria
- [x] Relatorio do recorte: linhas, municipios, cobertura, notificacoes, por UF
- [~] Alvo dimensional aproximado: ~420K linhas, ~1.6K municipios, cobertura >=95%
      **Entregue: 358.678 linhas, 1.369 municipios, 94,26%.** A meta foi
      estimada antes da US-001. Corrigidos os zeros falsos de chuva, a
      cobertura real do INMET e menor do que parecia, e o filtro de 80%
      barra mais municipios. Nao foi afrouxado para bater a meta.

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
- [x] Split parametrizado, nao hardcoded em `prepare_model_dataset.py:85-87`
- [x] Split do recorte: train 2019-2021, val 2022, test 2023
- [~] Volumetria esperada: train ~300K (60%), val ~100K (20%), test ~101K (20%)
      **Entregue: 214.933 / 71.188 / 72.557.** As proporcoes batem
      (59,9% / 19,8% / 20,2%); o volume absoluto segue o desvio da US-005.
- [x] Documentado explicitamente que 19 anos de historico epidemiologico foram trocados por cobertura climatica, e que isso penaliza o braco SINAN-only

### US-007: Baselines ingenuos

**Description:** Como banca, quero saber se o modelo supera a previsao trivial, porque o SHAP mostra `notificacoes_lag_1` dominando por 23x e ha risco real de o XGBoost estar reproduzindo persistencia.

**Acceptance Criteria:**
- [x] Baseline de persistencia: `casos_t+4 = casos_t`
- [x] Baseline sazonal-naive: `casos_t+4 = casos da mesma semana epidemiologica do ano anterior`
- [x] Ambos avaliados no mesmo conjunto de teste e com as mesmas metricas dos modelos
- [x] Resultados em `models/baselines/` com o mesmo formato de `metrics.json` dos demais
- [x] Nenhum resultado de XGBoost e reportado sem a linha de baseline ao lado

### US-008: Retreinar e reexecutar a ablacao

**Description:** Como pesquisador, quero responder se o clima ajuda, em condicoes controladas, sobre as mesmas linhas nos dois bracos.

**Acceptance Criteria:**
- [x] **E1 — ablacao limpa**: SINAN-only vs SINAN+INMET, mesmas linhas do recorte 2019-2023
- [x] **E2 — referencia de producao**: SINAN-only sobre o historico completo, para nao perder os 27 anos
- [x] E1 e E2 sao reportados separadamente e a diferenca entre eles e explicada no texto
- [x] Optuna roda sem subsampling (hoje usa 500K de 5.8M)
- [x] SHAP roda sem subsampling (hoje usa 50K)
- [x] Toda tabela de resultado reporta R2_log e R2_orig lado a lado, mais MAE, com a escala no cabecalho
- [x] Classificacao reavaliada com os limiares sem vazamento (US-002)
- [x] Comparacao antes/depois documentada em `docs/`

### US-009: Testar a hipotese de granularidade espacial

**Description:** Como pesquisador, quero saber se a atribuicao estacao->municipio destroi o sinal climatico, porque o clima correlaciona 0.74-0.78 com casos no agregado por UF mas aparece como irrelevante no SHAP municipal.

**Acceptance Criteria:**
- [x] Dataset agregado por mesorregiao construido a partir do mesmo recorte
- [x] Ablacao SINAN-only vs SINAN+INMET repetida na granularidade de mesorregiao
- [x] Comparacao das tres granularidades: municipio, mesorregiao, UF
- [x] Conclusao documentada: se o clima ganha importancia em granularidade mais grossa, isso e contribuicao metodologica e nao resultado negativo

### US-010: Classificacao binaria de surto por canal endemico

**Description:** Como pesquisador, quero que o alerta de surto deixe de ser 4
classes de percentil sazonalmente confundido e passe a ser uma decisao
binaria contra um canal endemico, com limiar calibrado e baselines ao lado,
porque o classificador de 4 classes perde para a persistencia nos sete
bracos (F1_macro 0,399 contra 0,484) e o rotulo p90 agregado e, na pratica,
"e verao".

**Contexto bibliografico:** canal endemico e o metodo padrao de vigilancia
(OPAS); o trabalho mais proximo (Kirstein et al. 2025, deteccao de surto em
municipios brasileiros) usa duas variantes — `> media(SE)` e
`> media(SE) + 2*DP(SE)`. Brady et al. (2015) testaram 102 definicoes de
surto em dados brasileiros e acharam 65% de variacao na proporcao
classificada como surto conforme a definicao, entao reportar as duas lado a
lado nao e indecisao, e exigencia.

**Acceptance Criteria:**
- [x] `scripts/canal_endemico.py`: climatologia por (mesorregiao, semana_epi)
      sobre 2010-2018 (estritamente anterior a todos os splits). Nao 2014-2018:
      com n=5 o maior z-score possivel e (n-1)/sqrt(n) = 1,789 < 2 e a variante
      estrita seria inatingivel in-sample.
- [x] Rotulo alinhado a semana DO ALVO (t+4), nao da linha — via o mesmo
      `shift(-4)` por mesorregiao que `pmd.add_target()` usa. `se_clim = min(SE, 52)`
      obrigatorio (unico ano de 53 semanas em 2010-2018 e 2014).
- [x] Assercao mestra: `notificacoes_t4` reconstruido do gold bate exatamente
      com o parquet do E1 mesorregional (11.266 linhas, diff 0,0).
- [x] `scripts/metrics_common.py`: `binary_metrics`, `calibrar_limiar` (F1/F2/
      Youden), `aplicar_limiar`, `to_row_binario`. Funcoes de 4 classes
      intocadas — `roc_auc_score(multi_class="ovr")` quebra no binario e o
      `except` nu gravaria AUC=None em silencio.
- [x] `scripts/train_ablation.py`: `carregar_cfg(cfg)` extraida de `carregar()`
      com `cfg["meta"]` opcional; parametro `objetivo` no Optuna (`rmse_log`
      default preserva os 7 bracos bit a bit, `rmse_orig` para os novos).
- [x] `scripts/train_surto_binario.py`: bracos `meso_surto_sinan` e
      `meso_surto_sinan_inmet`. Sem `scale_pos_weight` — o desbalanceamento e
      tratado pelo limiar. Limiar calibrado na metade rigorosa da validacao
      (SE>=27, nunca vista por Optuna nem early stopping); versao de reuso da
      validacao inteira reportada ao lado.
- [x] `scripts/baselines_surto.py`: persistencia, sazonal, media-movel-4,
      **persistencia de rotulo** (`surto_t4 = surto_t`), sempre-positivo,
      sempre-negativo — todos classificados pelo mesmo limiar do canal.
- [x] `scripts/tabela_surto_binario.py` -> `docs/resultados_surto_binario.{csv,md}`,
      uma secao por variante. Raiz `models/ablacao_surto/` separada de
      `models/ablacao/` para nao quebrar `tabela_resultados.py` (que indexa
      `classificacao.F1_macro` sem `.get()`).
- [x] `scripts/train_quantis.py`: intervalos de predicao via `reg:quantileerror`
      multi-quantil (um so modelo), treino em log1p com `expm1` por quantil
      (equivariante, exato). Nao vira metrica oficial — WIS/CRPS fica adiado.
- [x] `scripts/validar_model_ready.py`: `mesorregiao_surto` em CONJUNTOS, as 6
      colunas de rotulo em NUNCA_FEATURE.

**Resultado (teste 2023, variante `media`, F1_positivo):**

| abordagem | F1_pos |
|---|---|
| sempre-negativo | 0,000 |
| sazonal | 0,526 |
| sempre-positivo | 0,750 |
| classificador dedicado (XGBoost binario) | 0,72-0,74 |
| media-movel-4 | 0,809 |
| persistencia | 0,825 |
| **limiar sobre a previsao da regressao** | **0,830-0,832** |
| persistencia de rotulo | 0,846 |

**Achado central:** o classificador binario dedicado, mesmo com rotulo
corrigido, limiar calibrado e Optuna proprio, **perde para a persistencia**.
A saida: como "surto" e literalmente `notificacoes_t4 > limiar`, limiarizar a
previsao continua da regressao (R2_orig 0,44-0,49) pelo canal — sem
classificador, sem parametro livre — supera a persistencia. O gargalo era o
classificador, nao o sinal. Por essa via SINAN-only e SINAN+INMET empatam
(0,832 vs 0,830): o ganho de clima do R2_orig continuo nao se traduz em
ganho de F1 binario. `classificar_via_regressao` em `train_surto_binario.py`
e agora resultado oficial em `models/ablacao_surto/*/metrics.json`.

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

1. ~~Municipios em val/test ausentes do treino: excluir ou usar percentil da UF como fallback?~~
   **Resolvido: percentil da UF, com a origem registrada.** Cada linha carrega
   `threshold_source` (municipio / uf_fallback / nacional_fallback). Na pratica
   o fallback nao chegou a ser exercido no E1 — todos os 1.369 municipios do
   recorte aparecem no treino —, mas fica no codigo porque um recorte diferente
   pode precisar.

2. ~~E2 (historico completo) usa quais anos de split?~~
   **Resolvido: treino ate 2020, validacao 2021-2022, teste 2023, a partir de
   2000.** O ano de teste e deliberadamente o mesmo do E1, para que a diferenca
   entre os dois nao misture efeito de recorte com efeito de periodo de teste.
   Registrado em `config/recorte.json` sob `split_e2`.

3. ~~A agregacao por mesorregiao usa media simples do clima ou ponderada por populacao?~~
   **Resolvido: media entre estacoes distintas, sem ponderacao populacional.**
   O dataset nao carrega populacao e traze-la do IBGE ampliaria o escopo. A
   media por estacao distinta ja resolve o pior do problema, que era uma
   estacao atendendo oito municipios pesar oito vezes numa media por municipio.
   A ponderacao populacional fica como trabalho futuro.

4. ~~Se a ablacao E1 continuar desfavoravel ao clima, isso vira o resultado
   principal ou dispara a migracao para ERA5?~~
   **Nao se aplica: a ablacao passou a favorecer o clima.** SINAN+INMET supera
   SINAN-only nas quatro metricas no recorte municipal, e o ganho triplica na
   mesorregiao. A migracao para ERA5 nao e necessaria para sustentar a
   conclusao. Ela continua sendo o caminho obvio para cobertura, ja que o
   recorte precisou descartar 21 UFs por falta de estacao, mas isso e escopo
   de outro trabalho.

## Questoes que este trabalho abriu

1. **Existe uma granularidade espacial otima e ela nao foi procurada.** O
   ganho do clima e +0,029 no municipio, +0,106 na mesorregiao e -0,097 na UF.
   O otimo esta entre municipio e UF, e a mesorregiao foi o unico ponto
   intermediario testado. Microrregiao e regiao imediata ficaram de fora.

2. **A busca de hiperparametros otimiza RMSE em log e o TCC reporta R2 em
   contagem.** Rodar o E2 com 1 e com 20 trials melhorou o primeiro e piorou o
   segundo. Alinhar o objetivo da busca a metrica reportada e provavelmente o
   ajuste de maior retorno que sobrou.

3. ~~**O classificador perde para a persistencia em F1_macro nos sete bracos.**~~
   **Encaminhado na US-010.** Rotulo trocado para canal endemico binario,
   limiar calibrado, Optuna proprio testado. O classificador dedicado
   *continua* perdendo para a persistencia — o que aponta que o problema nao
   era desbalanceamento nem calibracao. A saida foi abandonar o classificador
   separado e classificar limiarizando a previsao da regressao pelo canal, que
   supera a persistencia. Fica aberto: por que um XGBClassifier probabilistico
   captura a autocorrelacao pior do que "copiar o valor de hoje", e se isso
   vale so nesta granularidade.

4. **Os intervalos de predicao subcobrem em 2023.** `train_quantis.py` da
   cobertura empirica de 0,79 (SINAN+INMET) e 0,85 (SINAN-only) contra 0,90
   nominal — a mesma degradacao em anos recentes que aparece em todo o
   projeto. WIS/CRPS como metrica oficial fica para trabalho futuro.

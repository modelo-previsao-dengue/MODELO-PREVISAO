# PRD: Notebooks TCC2 — Previsao Dengue DF

## Introduction

Reestruturar o trabalho de modelagem do TCC2 em notebooks separados por etapa, rodando no Kaggle (GPU T4, 30GB RAM). O foco e exclusivamente o Distrito Federal (~1.200 rows de dados model-ready). Os notebooks devem: explorar e documentar todas as camadas de dados (Bronze/Silver/Gold) com narrativa completa; treinar modelos XGBoost (regressao + classificacao) com Optuna tuning; analisar os modelos com SHAP e walk-forward CV; e organizar todos os resultados de forma clara para a banca do TCC2.

**Dados**: SINAN (dengue) + INMET (meteorologia), estacao A001 (1.18km de Brasilia).
**Ambiente**: Kaggle Notebooks (GPU T4, 30GB RAM), dados via HuggingFace Datasets.
**Escopo**: Apenas Distrito Federal (IBGE 5300108).

## Goals

- Produzir 3 notebooks autocontidos e executaveis no Kaggle
- Documentar visualmente todas as transformacoes de dados (bronze -> silver -> gold)
- Treinar modelos XGBoost otimizados com Optuna (regressao t+4 e classificacao de risco)
- Validar modelos com walk-forward CV temporal
- Gerar analise SHAP para interpretabilidade
- Comparar abordagens: baseline vs tuned, com/sem INMET, com/sem log-transform
- Organizar metricas, graficos e tabelas comparativas para apresentacao na banca

## User Stories

### US-001: Notebook 1 — Exploracao e Documentacao dos Dados

**Description:** Como leitor do TCC, quero ver os dados em cada camada (Bronze, Silver, Gold) com explicacoes e visualizacoes, para entender o pipeline de tratamento.

**Acceptance Criteria:**
- [ ] Notebook `1_dados_df.ipynb` criado e executavel no Kaggle
- [ ] Carrega dados do HuggingFace (repo `thiagorfreitas/dengue-tcc2-data`)
- [ ] Secao SINAN Bronze: `head()`, `shape`, `describe()`, distribuicao temporal, tipos de colunas, exemplos de registros brutos, explicacao markdown do que sao os dados
- [ ] Secao SINAN Silver: transformacoes aplicadas (antes/depois), valores missing, agregacao semanal, grafico de serie temporal de notificacoes
- [ ] Secao SINAN Gold: features engineered, descricao de cada grupo de features (lags, rolling, sazonais), heatmap de correlacao com target
- [ ] Secao INMET Bronze: dados brutos da estacao A001, cobertura temporal, variaveis meteorologicas
- [ ] Secao INMET Silver: tratamento de missing, interpolacao, agregacao semanal
- [ ] Secao INMET Gold: features derivadas, correlacao com dengue
- [ ] Secao Integracao: secao dedicada e visual mostrando o join SINAN+INMET
  - [ ] Diagrama/tabela mostrando como as tabelas se conectam (chaves de join)
  - [ ] Heatmap de missing por variavel climatica ao longo dos anos
  - [ ] Grafico de cobertura temporal: barras empilhadas mostrando semanas com/sem dados INMET por ano
  - [ ] Tabela resumo: total rows, colunas SINAN vs INMET, % missing por grupo
  - [ ] Exemplo visual de um registro integrado (1 linha com todas as colunas, destacando origem SINAN vs INMET)
  - [ ] Narrativa explicando a estrategia de integracao e impacto dos dados faltantes
- [ ] Secao Model-Ready: splits train/val/test com distribuicao temporal, histograma do target, estatisticas por split
- [ ] Graficos de distribuicao antes/depois de cada transformacao relevante
- [ ] Narrativa em markdown explicando o "por que" de cada etapa

### US-002: Notebook 2 — Treinamento dos Modelos

**Description:** Como pesquisador do TCC, quero treinar e otimizar modelos XGBoost para regressao (t+4) e classificacao (risco de surto), comparando diferentes abordagens.

**Acceptance Criteria:**
- [ ] Notebook `2_treino_df.ipynb` criado e executavel no Kaggle (GPU T4)
- [ ] Carrega splits model-ready do HuggingFace
- [ ] Aplica log-transform (log1p) no target de regressao
- [ ] Combina train+val para treino final (1.211 rows)
- [ ] **Experimento 1 — Baseline regressao**: XGBoost sem tuning, metricas R2/MAE/RMSE
- [ ] **Experimento 2 — Optuna regressao**: 50+ trials, search space documentado, melhores hiperparametros salvos
- [ ] **Experimento 3 — Modelo tuned regressao**: treino com best params, metricas no test
- [ ] **Experimento 4 — Comparacao SINAN-only vs SINAN+INMET**: mesmos hiperparametros, tabela comparativa
- [ ] **Experimento 5 — Baseline classificacao**: XGBClassifier para risco_surto_t4 (baixo/medio/alto/surto)
- [ ] **Experimento 6 — Optuna classificacao**: tuning com AUC como metrica
- [ ] **Experimento 7 — Modelo tuned classificacao**: best params, confusion matrix, classification report
- [ ] Walk-forward CV temporal (expanding window por ano) para regressao e classificacao
- [ ] Cada experimento documentado com markdown: hipotese, configuracao, resultado
- [ ] Modelos salvos em formato `.ubj` (XGBoost nativo)
- [ ] Metricas salvas em CSV/JSON para uso no notebook 3
- [ ] Tempo de execucao registrado para cada experimento

### US-003: Notebook 3 — Resultados e Analise

**Description:** Como apresentador na banca, quero um notebook com todos os resultados organizados, graficos comparativos e analise SHAP, pronto para gerar figuras do TCC.

**Acceptance Criteria:**
- [ ] Notebook `3_resultados_df.ipynb` criado e executavel no Kaggle
- [ ] Carrega modelos salvos (.ubj) e metricas (CSV/JSON) do notebook 2
- [ ] Tabela comparativa geral: todos os experimentos lado a lado (MAE, RMSE, R2, R2_log, AUC, F1)
- [ ] Grafico scatter pred vs real (escala log e original)
- [ ] Grafico serie temporal: real vs previsto no periodo de teste (2024)
- [ ] SHAP summary plot (top 20 features) para regressao
- [ ] SHAP summary plot para classificacao
- [ ] SHAP dependence plots para top 5 features
- [ ] Feature importance comparativa: XGBoost nativo vs SHAP
- [ ] Resultados walk-forward CV: grafico R2/MAE por ano
- [ ] Analise do distribution shift 2024: grafico mostrando o surto historico
- [ ] Confusion matrix heatmap para classificacao
- [ ] Precision/recall por classe de risco
- [ ] Secao "Discussao": markdown interpretando os resultados para o TCC
- [ ] Todas as figuras salvas em PNG 300dpi para inclusao no documento do TCC

## Functional Requirements

- FR-1: Todos os notebooks devem detectar o ambiente (Kaggle) e configurar paths automaticamente
- FR-2: Dados carregados exclusivamente do HuggingFace Datasets (repo `thiagorfreitas/dengue-tcc2-data`)
- FR-3: Filtrar apenas Distrito Federal (ibge_municipio == 5300108) em todas as etapas
- FR-4: Log-transform (log1p/expm1) aplicado no target de regressao para lidar com distribution shift
- FR-5: Train+Val combinados para treino (1.211 rows), test separado (180 rows, 2024+)
- FR-6: Optuna com no minimo 50 trials para regressao e 50 para classificacao
- FR-7: Walk-forward CV com janela expandindo (minimo 5 anos de treino)
- FR-8: SHAP calculado com TreeExplainer para todos os modelos finais
- FR-9: Metricas de regressao: MAE, RMSE, R2 (escala original), R2 (escala log)
- FR-10: Metricas de classificacao: AUC macro, F1 macro, precision/recall por classe
- FR-11: Modelos salvos em formato XGBoost nativo (.ubj)
- FR-12: Graficos com `plt.savefig()` em 300dpi para uso no documento do TCC
- FR-13: Reproducibilidade: `random_state=42` em todos os modelos e splits

## Non-Goals

- Nao incluir dados de outros municipios (apenas DF)
- Nao rodar no Google Colab (apenas Kaggle)
- Nao testar outros algoritmos alem de XGBoost
- Nao fazer deploy ou API do modelo
- Nao fazer feature selection automatica (manter todas as features do pipeline existente)
- Nao criar o documento do TCC em si (apenas gerar as figuras e tabelas)

## Technical Considerations

- **RAM**: dataset DF e pequeno (~1.200 rows), nao ha risco de OOM no Kaggle (30GB)
- **GPU**: XGBoost `tree_method="hist"` com `device="cuda"` para GPU T4
- **HuggingFace**: dados no repo `thiagorfreitas/dengue-tcc2-data`, precisa de HF_TOKEN configurado como Kaggle Secret
- **Camadas de dados**: Bronze (parquet bruto), Silver (limpo/agregado), Gold (features engineered) nos paths `data/{sinan,inmet}/{bronze,silver,gold}/`
- **Dependencia entre notebooks**: Notebook 2 gera artefatos (modelos .ubj, metricas .csv) que o Notebook 3 consome. No Kaggle, salvar em `/kaggle/working/` e usar como dataset de input no notebook seguinte
- **Versoes**: xgboost>=2.1, optuna>=3.6, shap>=0.46, scikit-learn>=1.4

## Success Metrics

- 3 notebooks executam do inicio ao fim no Kaggle sem erros
- R2_log >= 0.40 na regressao (benchmark atual: 0.46)
- AUC >= 0.75 na classificacao (benchmark atual: 0.76)
- Todas as figuras geradas em qualidade para o documento do TCC (300dpi)
- Banca consegue entender o pipeline completo lendo os notebooks em sequencia
- Walk-forward CV mostra estabilidade do modelo ao longo dos anos

## Resolved Questions

- **Dados locais**: Todas as camadas existem localmente — SINAN (bronze vazio, silver 57MB, gold 336MB), INMET (bronze 1GB, silver 17MB, gold 146MB), integrated, model_ready. O repo HF e privado; os notebooks devem carregar de la. Se faltar alguma camada no HF, subir do local.
- **Thresholds de risco**: Classes definidas por percentis historicos do municipio: 0=baixo (<=p50), 1=medio (p50-p75), 2=alto (p75-p90), 3=surto (>p90). DF train: {0:683, 1:236, 2:83, 3:52}.
- **Graficos**: Em ingles.
- **Optuna**: Usar MedianPruner para acelerar (dataset pequeno, nao precisa rodar trials ruins ate o fim).

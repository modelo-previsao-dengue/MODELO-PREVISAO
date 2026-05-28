# Pipeline XGBoost - Previsao de Dengue: Documentacao Completa

## Visao Geral

Pipeline completa de previsao de surtos de dengue em municipios brasileiros, utilizando dados epidemiologicos do SINAN e dados climaticos do INMET, com modelos XGBoost para regressao (contagem de notificacoes) e classificacao (nivel de risco).

O pipeline foi implementado em 4 fases com 13 user stories (US-001 a US-013), cobrindo desde a ingestao de dados brutos ate a validacao temporal e analise de explicabilidade.

---

## Fase 1: Pipeline INMET (US-001 a US-004)

### US-001 - Extracao e Padronizacao (scripts/inmet_extract_standardize.py)

Extrai e padroniza CSVs meteorologicos dos ZIPs anuais do INMET (2000-2024).

- **Entrada**: ZIPs do INMET em `data/inmet/bronze/` (formato CSV com header de 8 linhas, separador `;`, decimal `,`, encoding latin1)
- **Saida**: `data/inmet/bronze/hourly/*.parquet`, `estacoes_inmet.csv` (688 estacoes), `inventory.csv`
- **Volume**: 24 anos, ~73M registros horarios, ~1.1GB em parquet
- **Detalhe tecnico**: Usa `os.listdir()` em vez de `glob.glob()` para contornar normalizacao NFD do macOS em nomes de arquivo com acentos

### US-002 - Agregacao Semanal (scripts/inmet_weekly_aggregate.py)

Agrega dados horarios em semanas epidemiologicas por estacao.

- **Saida**: `data/inmet/silver/weekly_stations_YYYY.parquet`
- **Features geradas**: rain_sum_mm, rain_days, rain_heavy_days, temp_mean/min/max/range, humidity_mean, pressure_mean, wind_speed_mean, radiation_mean
- **Cobertura**: 442.168 estacao-semanas
- **Flag de qualidade**: `low_coverage` (< 50% das horas disponiveis)

### US-003 - Mapeamento Estacao-Municipio (scripts/inmet_station_municipality.py)

Mapeia 688 estacoes INMET para 5.571 municipios IBGE via distancia haversine vetorizada.

- **Saida**: `municipio_estacao_mapping.csv`
- **Cobertura**: 79.1% dos municipios com estacao a <= 50km, 99.0% a <= 100km
- **Metodo**: Nearest-neighbor (estacao mais proxima para cada municipio)

### US-004 - Features Climaticas Municipais (scripts/inmet_municipal_features.py)

Junta dados de estacoes aos municipios e calcula features temporais.

- **Saida**: `data/inmet/gold/weekly_municipal_climate_YYYY.parquet`, `inmet_feature_catalog.csv`
- **Features**: 34 features incluindo lags (1, 2, 4, 8 semanas) e medias moveis (4 semanas) para chuva, temperatura e umidade
- **Volume**: 4.008.405 registros

---

## Fase 2: Integracao e Preparacao (US-005 e US-006)

### US-005 - Integracao SINAN + INMET (scripts/integrate_sinan_inmet.py)

LEFT JOIN dos dados epidemiologicos SINAN Gold com features climaticas INMET Gold.

- **Chave de juncao**: (ibge_municipio, ano, semana_epidemiologica)
- **Saida**: `data/integrated/sinan_inmet_municipal_weekly.parquet`
- **Volume**: 7.665.428 linhas, 172 colunas
- **Cobertura INMET**: 51.4% dos registros com dados climaticos (restante tem NaN para features climaticas)

### US-006 - Dataset para Modelagem (scripts/prepare_model_dataset.py)

Cria targets, classificacao de risco, e split temporal.

- **Target regressao**: `notificacoes_t4` (notificacoes 4 semanas a frente por municipio)
- **Target classificacao**: `risco_surto_t4` (4 classes: baixo, medio, alto, surto - baseado em percentis p50/p75/p90)
- **Split temporal**:
  - Train: 2000-2019 (5.822.243 linhas)
  - Validation: 2020-2022 (873.731 linhas)
  - Test: 2023-2026 (947.194 linhas)
- **Features**: 161 features apos exclusao de colunas com >99% missing e colunas FHD
- **Saida**: `data/model_ready/train.parquet`, `val.parquet`, `test.parquet`, `feature_schema.csv`

---

## Fase 3: XGBoost MVP (US-007 a US-009)

### US-007 - Regressao MVP (scripts/train_xgb_regression.py)

XGBRegressor com parametros padrao para previsao t+4.

| Metrica | Valor |
|---------|-------|
| RMSE | 166.72 |
| MAE | 7.47 |
| R2 | 0.31 |
| MAPE | 88.81% |

- **Parametros**: n_estimators=500, max_depth=6, lr=0.1, subsample=0.8, colsample=0.8, tree_method=hist
- **Early stopping**: Parou na epoca 59 (de 500)
- **Saida**: `models/xgb_regression_mvp/`

### US-008 - Classificacao MVP (scripts/train_xgb_classification.py)

XGBClassifier multi-class (4 classes de risco) com multi:softprob.

| Metrica | Valor |
|---------|-------|
| AUC macro | 0.84 |
| F1 macro | 0.48 |
| F1 baixo | 0.88 |
| F1 medio | 0.20 |
| F1 alto | 0.22 |
| F1 surto | 0.63 |

- **Nota**: AUC macro (0.84) superou meta do PRD (0.70)
- **Desbalanceamento**: Classe "baixo" domina (~69% dos dados)
- **Early stopping**: Parou na epoca 283 (~2h de treino)
- **Saida**: `models/xgb_classification_mvp/`

### US-009 - Baseline SINAN-only vs SINAN+INMET (scripts/train_xgb_baseline.py)

Compara modelo apenas com features epidemiologicas vs modelo completo com clima.

| Modelo | Features | RMSE | R2 |
|--------|----------|------|-----|
| SINAN-only | 129 | 152.15 | 0.42 |
| SINAN+INMET | 159 | 166.72 | 0.31 |

**Achado importante**: O modelo SINAN-only e MELHOR que o SINAN+INMET. O INMET piorou o desempenho em +9.58% no RMSE. Hipotese: com apenas 51.4% de cobertura INMET, o modelo aprende a separar dados com/sem clima (splits em NaN) em vez de extrair sinal climatico real.

---

## Fase 4: Refinamento (US-010 a US-013)

### US-010 - Tuning com Optuna (scripts/tune_xgb_optuna.py)

Otimizacao Bayesiana de hiperparametros com 20 trials.

- **Subsample tuning**: 500K linhas (de 5.8M)
- **Subsample retrain**: 2M linhas
- **Espaco de busca**: max_depth [3,8], lr [0.01,0.3], n_estimators [100,300], subsample, colsample, reg_alpha, reg_lambda, min_child_weight

**Melhores hiperparametros encontrados**:

| Parametro | Valor |
|-----------|-------|
| max_depth | 8 |
| learning_rate | 0.0213 |
| n_estimators | 101 |
| subsample | 0.731 |
| colsample_bytree | 0.606 |
| min_child_weight | 20 |
| reg_alpha | 1.2e-7 |
| reg_lambda | 0.00137 |

**Resultado**: RMSE 166.76 no teste - essencialmente identico ao MVP (166.72). O tuning nao trouxe melhoria significativa, sugerindo que o gargalo e a qualidade/cobertura dos dados, nao os hiperparametros.

**Importancia dos hiperparametros** (Optuna):
1. learning_rate (36.1%)
2. colsample_bytree (22.8%)
3. min_child_weight (16.5%)
4. subsample (9.7%)

### US-011 - Explicabilidade SHAP (scripts/explain_shap.py)

TreeExplainer em amostra de 50K linhas do teste.

**Top-10 features por SHAP (mean |SHAP value|)**:

| Rank | Feature | Mean |SHAP| | Climatica? |
|------|---------|----------------|------------|
| 1 | notificacoes_lag_1 | 2.12 | Nao |
| 2 | notificacoes_media_movel_3 | 1.66 | Nao |
| 3 | notificacoes_diff_4 | 1.45 | Nao |
| 4 | qt_confirmados_provaveis | 1.00 | Nao |
| 5 | week_of_year_cos | 0.73 | Nao |
| 6 | notificacoes_razao_media_4 | 0.59 | Nao |
| 7 | notificacoes_min_movel_3 | 0.43 | Nao |
| 8 | notificacoes_razao_media_8 | 0.38 | Nao |
| 9 | notificacoes_media_movel_4 | 0.30 | Nao |
| 10 | notificacoes_diff_1 | 0.26 | Nao |

**Clima no ranking**: Apenas 1 feature climatica no top-20 (`rain_sum_mm_mm4`, rank 20, SHAP=0.09). Features climaticas com lag ficaram nas posicoes 143-153 de 161. Isso confirma quantitativamente que o sinal climatico tem impacto minimo no modelo atual, provavelmente devido a baixa cobertura INMET (51.4%).

**Artefatos**: `shap_summary_beeswarm.png`, `shap_dependence_climate_top5.png`, `shap_feature_importance.csv`

### US-012 - Validacao Walk-Forward (scripts/validate_walk_forward.py)

Validacao temporal com janela expansiva, testando a cada 3 anos.

- **Folds**: 9 (anos de teste: 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025, 2026)
- **Amostragem**: 30K linhas por ano (~815K total) para viabilidade em 16GB RAM

| Ano Teste | RMSE | MAE | R2 |
|-----------|------|-----|-----|
| 2004 | 5.04 | 0.99 | 0.23 |
| 2007 | 14.66 | 1.95 | 0.64 |
| 2010 | 27.74 | 3.10 | 0.63 |
| 2013 | 65.39 | 4.51 | 0.49 |
| 2016 | 71.68 | 4.78 | 0.52 |
| 2019 | 80.68 | 5.24 | 0.56 |
| 2022 | 32.48 | 4.00 | 0.54 |
| 2025 | 32.00 | 3.45 | 0.66 |
| 2026 | 16.37 | 3.82 | 0.48 |

- **Media**: RMSE 38.47 +/- 27.30, R2 0.56
- **CV do RMSE**: 71% (nao estavel, CV > 30%)
- **Interpretacao**: A alta variabilidade e esperada - dengue e episodica, com grandes surtos em anos especificos (2013, 2016, 2019) que elevam o RMSE. O R2 medio de 0.56 e superior ao R2 de 0.31 do split fixo, sugerindo que o modelo generaliza razoavelmente ao longo do tempo.

### US-013 - Multi-Horizonte (scripts/train_multi_horizon.py)

Modelos separados para horizontes t+1, t+2, t+4 e t+8 semanas.

| Horizonte | RMSE | MAE | R2 | MAPE |
|-----------|------|-----|-----|------|
| t+1 | 157.20 | 5.29 | 0.38 | 63.2% |
| t+2 | 160.95 | 5.82 | 0.35 | 71.5% |
| t+4 | 167.07 | 7.03 | 0.31 | 91.7% |
| t+8 | 185.54 | 9.52 | 0.16 | 132.5% |

- **Razao RMSE(t+8)/RMSE(t+1)**: 1.18 (aceitavel, < 2x)
- **Degradacao**: Graceful - R2 cai de 0.38 (t+1) para 0.16 (t+8), acompanhando a crescente incerteza em horizontes mais longos
- **Melhor horizonte**: t+1 oferece o melhor equilibrio (R2=0.38, MAPE=63%)

---

## Estrutura de Diretorios

```
MODELO-PREVISAO/
├── scripts/
│   ├── inmet_extract_standardize.py    # US-001
│   ├── inmet_weekly_aggregate.py       # US-002
│   ├── inmet_station_municipality.py   # US-003
│   ├── inmet_municipal_features.py     # US-004
│   ├── integrate_sinan_inmet.py        # US-005
│   ├── prepare_model_dataset.py        # US-006
│   ├── train_xgb_regression.py         # US-007
│   ├── train_xgb_classification.py     # US-008
│   ├── train_xgb_baseline.py           # US-009
│   ├── tune_xgb_optuna.py             # US-010
│   ├── explain_shap.py                # US-011
│   ├── validate_walk_forward.py        # US-012
│   └── train_multi_horizon.py          # US-013
├── data/
│   ├── inmet/bronze/hourly/            # Dados horarios padronizados
│   ├── inmet/silver/                   # Agregacao semanal por estacao
│   ├── inmet/gold/                     # Features climaticas municipais
│   ├── integrated/                     # SINAN + INMET integrado
│   └── model_ready/                    # Splits train/val/test
├── models/
│   ├── xgb_regression_mvp/             # Modelo regressao baseline
│   ├── xgb_classification_mvp/         # Modelo classificacao baseline
│   ├── xgb_baseline_sinan_only/        # Comparacao SINAN-only vs completo
│   ├── xgb_regression_tuned/           # Modelo com Optuna tuning
│   ├── shap_analysis/                  # Beeswarm, dependence plots, rankings
│   ├── walk_forward_results/           # Resultados validacao temporal
│   └── multi_horizon/                  # Modelos t+1, t+2, t+4, t+8
└── docs/
    └── xgboost_pipeline_results.md     # Este documento
```

---

## Achados Principais e Discussao

### 1. Features epidemiologicas dominam a previsao

A analise SHAP demonstra inequivocamente que as features derivadas da serie historica de notificacoes (lags, medias moveis, diferencas) sao os preditores mais fortes. O lag de 1 semana (`notificacoes_lag_1`) tem SHAP value 23x maior que a melhor feature climatica.

### 2. Dados climaticos INMET nao agregam valor ao modelo

Com apenas 51.4% de cobertura, as features climaticas introduzem mais ruido do que sinal. O modelo SINAN-only (R2=0.42) supera o SINAN+INMET (R2=0.31). Possibilidades de melhoria:
- Usar dados de satelite (CHIRPS, ERA5) com cobertura completa
- Implementar IDW (Inverse Distance Weighting) com multiplas estacoes em vez de nearest-neighbor
- Imputar dados climaticos faltantes usando interpolacao espacial

### 3. Tuning de hiperparametros tem retorno marginal

O Optuna nao conseguiu melhorar o modelo MVP, indicando que o gargalo esta nos dados e na formulacao do problema, nao nos parametros do modelo.

### 4. Modelo generaliza razoavelmente ao longo do tempo

O walk-forward mostra R2 medio de 0.56 (vs 0.31 no split fixo), com variabilidade alta (CV=71%) explicada pela natureza episodica dos surtos de dengue.

### 5. Classificacao de risco e viavel

AUC macro de 0.84 demonstra que o modelo consegue distinguir niveis de risco com boa discriminacao, especialmente para classes extremas (baixo F1=0.88, surto F1=0.63).

### 6. Degradacao multi-horizonte e aceitavel

A razao RMSE(t+8)/RMSE(t+1) de 1.18 indica que previsoes ate 8 semanas sao viaveis, com degradacao controlada.

---

## Reproducibilidade

### Dependencias

```
pandas, numpy, pyarrow, xgboost, scikit-learn, optuna, shap, matplotlib
```

### Execucao Sequencial

```bash
# Fase 1: Pipeline INMET
python3 scripts/inmet_extract_standardize.py
python3 scripts/inmet_weekly_aggregate.py
python3 scripts/inmet_station_municipality.py
python3 scripts/inmet_municipal_features.py

# Fase 2: Integracao
python3 scripts/integrate_sinan_inmet.py
python3 scripts/prepare_model_dataset.py

# Fase 3: XGBoost MVP
python3 scripts/train_xgb_regression.py
python3 scripts/train_xgb_classification.py
python3 scripts/train_xgb_baseline.py

# Fase 4: Refinamento
python3 scripts/tune_xgb_optuna.py
python3 scripts/explain_shap.py
python3 scripts/validate_walk_forward.py
python3 scripts/train_multi_horizon.py
```

### Requisitos de Hardware

- **RAM minima**: 16GB (scripts adaptados com subsampling para esta configuracao)
- **CPU**: Multi-core recomendado (XGBoost usa n_jobs=-1)
- **Disco**: ~5GB para dados intermediarios
- **Tempo total estimado**: ~12-15h para pipeline completa

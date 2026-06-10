# Modelo de Previsao de Dengue - Dados e Pipelines

Repositorio de **material tecnico** (dados, scripts e modelos) do TCC2 de Engenharia de Software da UnB.
A documentacao textual e o Overleaf ficam no repositorio [TCC2-DOCS](https://github.com/modelo-previsao-dengue/TCC2-DOCS).

**Titulo do TCC:** Desenvolvimento de um Modelo de Previsao para Surtos de Dengue em Municipios Brasileiros utilizando Series Temporais e Dados Climaticos

**Autores:** Pedro Lucas Santana e Thiago Ribeiro Freitas

---

## Estrutura do Repositorio

```
MODELO-PREVISAO/
├── data/
│   ├── raw/              # Microdados brutos SINAN (1.8 GB)
│   ├── sinan/            # Pipeline SINAN: bronze/silver/gold/serving/governance (1.1 GB)
│   ├── inmet/            # Pipeline INMET: bronze/silver/gold (1.2 GB)
│   ├── integrated/       # Join SINAN+INMET municipal semanal (270 MB)
│   ├── model_ready/      # Splits train/val/test prontos para XGBoost (304 MB)
│   ├── processed/        # Dados intermediarios legados (230 MB)
│   └── reference/        # Tabela IBGE municipios (4 MB)
│
├── scripts/
│   ├── sinan_tcc2_pipeline.py          # Pipeline SINAN nacional completa
│   ├── inmet_extract_standardize.py    # INMET: extracao e padronizacao horaria
│   ├── inmet_weekly_aggregate.py       # INMET: agregacao semanal por estacao
│   ├── inmet_station_municipality.py   # INMET: mapeamento estacao -> municipio
│   ├── inmet_municipal_features.py     # INMET: features climaticas municipais
│   ├── integrate_sinan_inmet.py        # Integracao SINAN + INMET
│   ├── prepare_model_dataset.py        # Feature engineering e split temporal
│   ├── train_xgb_regression.py         # XGBoost regressao MVP
│   ├── train_xgb_classification.py     # XGBoost classificacao de risco
│   ├── train_xgb_baseline.py           # Baseline SINAN-only
│   ├── tune_xgb_optuna.py             # Hyperparameter tuning com Optuna
│   ├── explain_shap.py                 # Interpretabilidade SHAP
│   ├── validate_walk_forward.py        # Validacao walk-forward temporal
│   └── train_multi_horizon.py          # Multi-horizonte (t+1, t+2, t+4, t+8)
│
├── models/
│   ├── xgb_regression_mvp/            # Modelo regressao + metricas + graficos
│   ├── xgb_classification_mvp/        # Modelo classificacao + confusion matrix
│   ├── xgb_baseline_sinan_only/       # Baseline sem clima + comparacao
│   ├── xgb_regression_tuned/          # Optuna trials + modelo otimizado
│   ├── shap_analysis/                 # SHAP beeswarm + dependencia + report
│   ├── walk_forward_results/          # Resultados validacao temporal
│   └── multi_horizon/                 # 4 modelos + degradacao
│
└── docs/                              # Documentacao tecnica das pipelines
```

## Fontes de Dados

| Fonte | Descricao | Periodo | Volume |
|-------|-----------|---------|--------|
| **SINAN** (OpenDataSUS) | Notificacoes de dengue por municipio e semana epidemiologica | 2000-2026 | 5.565 municipios, 27 anos |
| **INMET** (BDMEP) | Estacoes meteorologicas automaticas (temp, chuva, umidade, pressao, vento) | 2000-2026 | ~700 estacoes |
| **IBGE** | Codigos e coordenadas dos municipios brasileiros | - | 5.571 municipios |

## Arquitetura de Dados

Os dados passam por uma arquitetura **Medallion** (Bronze -> Silver -> Gold):

- **Bronze**: dados brutos extraidos das fontes oficiais
- **Silver**: dados limpos, padronizados, com granularidade municipio x semana
- **Gold**: dados enriquecidos com features prontas para modelagem

O dataset final integrado tem **7.6 milhoes de linhas** e **164 features**, dividido temporalmente:

| Split | Periodo | Linhas |
|-------|---------|--------|
| Train | < 2022 | 5.822.243 |
| Val | 2022-2023 | 873.731 |
| Test | 2024+ | 947.194 |

## Resultados Principais

| Experimento | Metrica | Valor |
|-------------|---------|-------|
| Regressao SINAN+INMET | R2 | 0.31 |
| Regressao SINAN-only | R2 | **0.42** |
| Classificacao de risco (4 classes) | AUC | **0.84** |
| Multi-horizonte t+1 | R2 | 0.38 |
| Multi-horizonte t+8 | R2 | 0.16 |

Achado principal: o modelo **sem dados climaticos** (SINAN-only) supera o modelo com clima (SINAN+INMET), devido a cobertura irregular do INMET (media 51.4%). A analise SHAP confirma que features epidemiologicas (lags de notificacoes) dominam a previsao.

## Como Reproduzir

### Pre-requisitos

```bash
python >= 3.9
pip install pandas pyarrow xgboost scikit-learn optuna shap matplotlib
```

### Pipeline completa

```bash
# 1. Pipeline SINAN
python3 scripts/sinan_tcc2_pipeline.py --start-year 2000 --end-year 2026 --version sinan_tcc2_v2

# 2. Pipeline INMET (4 etapas)
python3 scripts/inmet_extract_standardize.py
python3 scripts/inmet_weekly_aggregate.py
python3 scripts/inmet_station_municipality.py
python3 scripts/inmet_municipal_features.py

# 3. Integracao e feature engineering
python3 scripts/integrate_sinan_inmet.py
python3 scripts/prepare_model_dataset.py

# 4. Treinamento e avaliacao
python3 scripts/train_xgb_regression.py
python3 scripts/train_xgb_classification.py
python3 scripts/train_xgb_baseline.py
python3 scripts/tune_xgb_optuna.py
python3 scripts/explain_shap.py
python3 scripts/validate_walk_forward.py
python3 scripts/train_multi_horizon.py
```

## Dados Pesados

Os arquivos Parquet (~4.8 GB) nao estao versionados no GitHub (excluidos pelo `.gitignore`).
Para obter os dados, execute as pipelines acima a partir das fontes oficiais ou entre em contato com os autores.

## Stack

Python 3.9+ | pandas | pyarrow | XGBoost | scikit-learn | Optuna | SHAP | matplotlib

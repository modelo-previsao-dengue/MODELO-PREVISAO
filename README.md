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

### Recorte espaco-temporal (branch `feat/retreino-recorte-2019-2023`)

O dataset de modelagem deixou de ser o Brasil inteiro de 2000 a 2024. O clima
so existe onde ha estacao automatica do INMET funcionando, e a cobertura real
so ficou visivel depois de corrigir os zeros falsos de chuva. O recorte troca
abrangencia por qualidade, e o preco esta dito por extenso abaixo.

**E1 — recorte principal**, 6 UFs (DF, ES, GO, MG, RJ, SP) x 2019-2023, so
municipios com pelo menos 80% das semanas com as tres variaveis climaticas
presentes:

| Split | Periodo | Linhas |
|-------|---------|--------|
| Train | 2019-2021 | 214.933 |
| Val | 2022 | 71.188 |
| Test | 2023 | 72.557 |

1.369 municipios, 178 features, cobertura climatica de 94,26%. Ha ainda uma
janela de warm-up (2018) e cool-down (2024) que e processada mas nao
entregue, para que os lags de 12 semanas e o alvo t+4 das bordas existam.

**E2 — referencia de producao**, SINAN-only sobre 2000-2023, com o **mesmo ano
de teste (2023)**: treino ate 2020, validacao 2021-2022, 6.110.370 linhas de
treino.

**O custo, dito com todas as letras:** o recorte troca 19 anos de historico
epidemiologico por cobertura climatica. Isso penaliza o braco SINAN-only da
ablacao, que teria muito mais dados se pudesse usar o historico inteiro. O E2
existe justamente para medir esse efeito separadamente, e por isso usa o mesmo
ano de teste do E1 — assim a diferenca entre os dois nao mistura efeito de
recorte com efeito de periodo de teste.

### Features climaticas

As features do INMET passaram a espelhar as epidemiologicas do SINAN: lags de
1, 2, 4, 8 e 12 semanas e medias moveis de 4, 8 e 12, sobre chuva,
temperatura, umidade relativa e ponto de orvalho. Antes iam so ate 8 semanas,
mas o pico da correlacao clima-casos esta em 9-12 semanas em 11 de 12 UFs
auditadas — o modelo anterior nunca viu a janela relevante. Foram de 27 para
49 features climaticas.

Os lags sao calculados sobre uma grade semanal completa: sem isso,
`shift(12)` em uma serie com buracos anda 12 linhas e nao 12 semanas, e o
"lag de 12 semanas" podia ser de 15.

## Resultados Principais

Tabela completa em
[`docs/resultados_modelos_e_baselines.md`](docs/resultados_modelos_e_baselines.md).
Nenhum numero de XGBoost e reportado sem a linha de baseline ao lado.

MAE em casos por municipio-semana. `R2_orig` e sobre a contagem, `R2_log`
sobre log1p, que e o que o treino otimiza — os dois sao publicados porque sao
numeros diferentes e o relatorio anterior dizia so "R2".

### E1 — recorte 2019-2023, 1.369 municipios, teste em 2023

| Modelo | MAE | R2_orig | R2_log | F1_macro | AUC |
|---|---|---|---|---|---|
| baseline: persistencia | 9,449 | **0,4003** | 0,5700 | **0,4842** | — |
| baseline: media movel 4 | 10,891 | 0,3817 | 0,5689 | 0,4583 | — |
| baseline: sazonal | 14,046 | -0,1038 | 0,0473 | 0,3236 | — |
| XGBoost SINAN-only | 8,060 | 0,3611 | 0,7057 | 0,4011 | 0,7775 |
| XGBoost SINAN+INMET | **7,935** | 0,3900 | **0,7068** | 0,3992 | **0,7859** |

### E2 — historico completo 2000-2023, SINAN-only, mesmo teste (2023)

| Modelo | MAE | R2_orig | R2_log | F1_macro | AUC |
|---|---|---|---|---|---|
| baseline: persistencia | 4,654 | 0,4352 | 0,5845 | **0,4969** | — |
| XGBoost SINAN-only | **3,744** | **0,4897** | **0,7335** | 0,4514 | **0,8402** |

### US-009 — o mesmo recorte agregado por mesorregiao (43 series)

| Modelo | MAE | R2_orig | R2_log | F1_macro | AUC |
|---|---|---|---|---|---|
| baseline: persistencia | 252,10 | 0,3971 | 0,5322 | **0,5675** | — |
| XGBoost SINAN-only | 224,02 | 0,3814 | 0,6924 | 0,4092 | 0,7244 |
| XGBoost SINAN+INMET | **195,65** | **0,4874** | **0,7031** | 0,4336 | **0,7553** |

### Achados

1. **O clima contribui.** O braco SINAN+INMET supera o SINAN-only em todas as
   metricas. Isso **inverte a conclusao anterior deste repositorio**, de que o
   clima nao ajudava: aquele resultado se apoiava em ponto de orvalho rotulado
   como umidade, chuva ausente virando zero e lags que paravam antes da janela
   em que a correlacao existe. Ver
   [`docs/GUIA_TECNICO_RETREINO.md`](docs/GUIA_TECNICO_RETREINO.md).

2. **A escala espacial importa mais que o modelo, e tem um otimo.** O ganho
   do clima vai de +0,029 de R2_orig no municipio para +0,106 na mesorregiao,
   e **inverte para -0,097 na UF**. Agregar demais destroi o sinal: na UF sao
   17 estacoes numa media so, e a media resultante nao descreve o clima de
   lugar nenhum. Existe uma granularidade otima, e neste recorte ela e a
   mesorregiao — o que e contribuicao de metodo, nao resultado negativo.

3. **No recorte, o XGBoost nao bate a persistencia em R2_orig** (0,390
   contra 0,400). Ele ganha em MAE, ou seja, acerta melhor a semana tipica, e
   perde nos picos, que e o que domina o R2 — e o pico e justamente o que a
   vigilancia precisa prever. Com o historico inteiro (E2) o modelo supera a
   persistencia com folga, 0,490 contra 0,435, e na mesorregiao tambem, 0,487
   contra 0,397. O limitante e o tamanho do treino que o recorte deixa: 215
   mil linhas contra 6,1 milhoes.

4. **A classificacao ainda perde para o baseline em F1_macro.** O XGBoost
   minimiza logloss e, sem tratamento de desbalanceamento, escorrega para a
   classe majoritaria. Correcao conhecida e ainda nao aplicada, para nao
   misturar o efeito com o da ablacao.

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

---
language:
  - pt
license: cc-by-4.0
task_categories:
  - time-series-forecasting
  - tabular-classification
tags:
  - dengue
  - epidemiologia
  - xgboost
  - brazil
  - public-health
  - climate
  - sinan
  - inmet
  - mesoregion
  - outbreak-prediction
size_categories:
  - 100K<n<1M
---

# DengueCast — Sistema Preditivo de Surtos de Dengue

Dados, modelos treinados e reports de pipeline do **DengueCast v4**: sistema de previsao de surtos de dengue para **137 mesorregioes brasileiras** (2001–2026), usando XGBoost com dados epidemiologicos (SINAN) e climaticos (INMET).

## Links Rapidos

| Recurso | Link |
|---------|------|
| Dashboard interativa | [modelo-previsao-ffqc3ugltpbxxeaa4mygvw.streamlit.app](https://modelo-previsao-ffqc3ugltpbxxeaa4mygvw.streamlit.app) |
| Notebook: Exploracao | [Kaggle — v4_01_exploracao](https://www.kaggle.com/pedrolucassantanaf/denguecast-v4-01-exploracao) |
| Notebook: Treino | [Kaggle — v4_02_treino](https://www.kaggle.com/pedrolucassantanaf/denguecast-v4-02-treino) |
| Notebook: Resultados | [Kaggle — v4_03_resultados](https://www.kaggle.com/pedrolucassantanaf/denguecast-v4-03-resultados) |
| Repositorio GitHub | [github.com/pedrolucassantanaf/MODELO-PREVISAO](https://github.com/pedrolucassantanaf/MODELO-PREVISAO) |

## Sobre o Projeto

Trabalho de Conclusao de Curso (TCC) em Engenharia de Software — Universidade de Brasilia (UnB), Faculdade do Gama (FGA), 2024–2025.

**Autores:** Pedro Lucas F. Santana e Thiago Ribeiro Freitas

O sistema preve o numero de notificacoes de dengue 4 semanas a frente (`notificacoes_t4`) e classifica o nivel de risco em 4 categorias (Baixo, Moderado, Alto, Surto) para cada mesorregiao brasileira.

## Dados

### Pipeline v4 — Mesorregioes (`data/model_ready_v4/`)

O dataset principal do TCC. Cobertura:
- **137 mesorregioes** (todas as 27 UFs)
- **2001–2026** (semanas epidemiologicas)
- **266 features** (110 SINAN + 156 INMET)
- **~180.000 linhas** (train + val + test)

Organizado em 3 variantes experimentais:

| Variante | Treino | Validacao | Teste | Descricao |
|----------|--------|-----------|-------|-----------|
| **A** (baseline) | 2019–2022 (28.633 linhas) | 2023 (7.261) | 2024–2026 (16.029) | Serie curta, 4 anos |
| **B** (principal) | 2001–2022 (157.413 linhas) | 2023 (7.261) | 2024–2026 (16.029) | Serie completa, 22 anos |
| **C** (decay) | 2001–2022 (157.413 linhas) | 2023 (7.261) | 2024–2026 (16.029) | = B + pesos exponenciais (lambda=0.1) |

Cada variante tem 3 splits: `train.parquet`, `val.parquet`, `test.parquet` em `splits_{a,b,c}/`.

### Dados Municipais Antigos (`data/model_ready/`)

Pipeline anterior (pre-v4) com dados no nivel municipal. Mantido para referencia e comparacao historica.

## Modelos

18 modelos XGBoost treinados: 3 variantes (A/B/C) × 3 bracos de ablacao × 2 tarefas (regressao + classificacao).

### Regressao — R² no teste (log1p)

| Modelo | SINAN-only | INMET bruto | INMET enriquecido |
|--------|-----------|-------------|-------------------|
| **Var A** (4 anos) | 0.8690 | 0.8692 | 0.8723 |
| **Var B** (22 anos) | 0.8821 | 0.8848 | **0.8878** |
| **Var C** (decay) | 0.8789 | 0.8835 | 0.8870 |

**Melhor modelo:** `B_inmet_enriquecido` — R²_log = 0.8878 (IC 95%: 0.882–0.893), 266 features, 157.413 linhas de treino.

### Classificacao de Risco (4 classes)

Os 9 modelos de classificacao (`models/classification_v4/`) preveem o nivel de risco de surto em 4 categorias:
- **0 — Baixo**: abaixo do canal endemico
- **1 — Moderado**: dentro do canal endemico
- **2 — Alto**: acima do canal endemico
- **3 — Surto**: acima do limiar epidemico

## Como Usar

### Baixar dados e treinar modelo

```python
from huggingface_hub import hf_hub_download
import pandas as pd
import xgboost as xgb
import numpy as np

REPO = "pedrolucassantanaf/dengue-tcc2-data"

# Baixar splits da variante B (principal)
train = pd.read_parquet(hf_hub_download(REPO, "data/model_ready_v4/splits_b/train.parquet", repo_type="dataset"))
val = pd.read_parquet(hf_hub_download(REPO, "data/model_ready_v4/splits_b/val.parquet", repo_type="dataset"))
test = pd.read_parquet(hf_hub_download(REPO, "data/model_ready_v4/splits_b/test.parquet", repo_type="dataset"))

# Baixar schema de features
schema = pd.read_csv(hf_hub_download(REPO, "data/model_ready_v4/feature_schema_v4.csv", repo_type="dataset"))

# Selecionar features (inmet_enriquecido = todas)
features = schema["feature"].tolist()
target = "notificacoes_t4"

X_train, y_train = train[features], np.log1p(train[target])
X_val, y_val = val[features], np.log1p(val[target])
X_test, y_test = test[features], np.log1p(test[target])

# Treinar XGBoost
model = xgb.XGBRegressor(
    n_estimators=2000, tree_method="hist",
    early_stopping_rounds=50, eval_metric="rmse"
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Avaliar
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print(f"R² (log): {r2_score(y_test, y_pred):.4f}")
```

### Carregar modelo pre-treinado

```python
from huggingface_hub import hf_hub_download
import xgboost as xgb

REPO = "pedrolucassantanaf/dengue-tcc2-data"

# Baixar modelo principal
path = hf_hub_download(REPO, "models/regression_v4/model_B_inmet_enriquecido.ubj", repo_type="dataset")
model = xgb.XGBRegressor()
model.load_model(path)
```

## Estrutura de Pastas

```
├── README.md                              ← Este arquivo
├── config/
│   └── recorte.json                       ← Configuracao do experimento v4
├── data/
│   ├── geojson/
│   │   └── mesorregioes.geojson           ← GeoJSON das 137 mesorregioes
│   ├── reference/
│   │   └── municipio_mesorregiao.csv      ← Mapeamento codigo → nome/UF
│   ├── model_ready_v4/                    ← PIPELINE V4 (mesorregioes)
│   │   ├── splits_{a,b,c}/               ← 3 × {train, val, test}.parquet
│   │   ├── *_full.parquet                 ← Dados intermediarios
│   │   ├── feature_schema_v4.csv          ← Mapa feature → origem
│   │   └── 40–54_*.json                   ← Reports de cada etapa
│   └── model_ready/                       ← Pipeline antigo (municipal)
├── models/
│   ├── regression_v4/                     ← 9 modelos XGBoost (.ubj) + metricas
│   └── classification_v4/                 ← 9 modelos classificacao (.ubj)
└── dashboard/
    └── data/                              ← 9 previsoes pre-computadas
```

## Variantes Experimentais

| | Variante A | Variante B | Variante C |
|---|---|---|---|
| **Hipotese** | Serie curta basta? | Serie longa melhora? | Decay temporal ajuda? |
| **Treino** | 2019–2022 | 2001–2022 | 2001–2022 |
| **Linhas treino** | 28.633 | 157.413 | 157.413 |
| **Pesos** | Uniformes | Uniformes | Exponencial (λ=0.1) |
| **R²_log melhor** | 0.8723 | **0.8878** | 0.8870 |
| **Conclusao** | Baseline razoavel | **Melhor desempenho** | Marginal vs B |

A **Variante B** (serie completa, sem decay) apresentou o melhor desempenho, especialmente com features INMET enriquecidas (anomalias, medias moveis, bioclimaticas).

## Fontes de Dados

### SINAN — Sistema de Informacao de Agravos de Notificacao
- **Fonte:** Ministerio da Saude (DataSUS)
- **Conteudo:** Notificacoes semanais de dengue por municipio
- **Periodo:** 2001–2026
- **Variaveis:** notificacoes, confirmados, obitos, hospitalizacoes, taxas de incidencia

### INMET — Instituto Nacional de Meteorologia
- **Fonte:** Portal INMET
- **Conteudo:** Dados horarios de estacoes meteorologicas automaticas
- **Periodo:** 2000–2026
- **Variaveis:** temperatura, umidade, precipitacao, vento, radiacao, pressao atmosferica
- **Processamento:** Agregacao semanal por mesorregiao, features derivadas (anomalias, medias moveis, indices bioclimaticos)

## Citacao

```bibtex
@misc{santana2025denguecast,
  title={DengueCast: Sistema Preditivo de Surtos de Dengue para Mesorregioes Brasileiras},
  author={Santana, Pedro Lucas F. and Freitas, Thiago Ribeiro},
  year={2025},
  institution={Universidade de Brasilia -- Faculdade do Gama (FGA)},
  note={Trabalho de Conclusao de Curso em Engenharia de Software}
}
```

## Licenca

- **Dados:** CC-BY-4.0 (dados publicos governamentais brasileiros, processados e agregados)
- **Modelos:** MIT
- **Codigo:** MIT (ver repositorio GitHub)

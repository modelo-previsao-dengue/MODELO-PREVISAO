# PRD: Publicacao Completa — HuggingFace + Kaggle

## 1. Introduction

O DengueCast v4 — sistema preditivo de surtos de dengue para 137 mesorregioes brasileiras, 2001–2026, com 3 variantes experimentais (A/B/C) e 3 sub-modelos de features (SINAN-only, INMET bruto, INMET enriquecido) — existe inteiramente na maquina local do autor. Nenhum dos 18 modelos treinados, nenhum dos 9 splits de dados, nenhum dos 14 reports de pipeline e nenhuma das 9 caches de previsao esta publicado. O co-autor (Thiago) nao consegue reproduzir, validar ou apresentar os resultados do TCC.

Este PRD define a publicacao completa e permanente de todos os artefatos no HuggingFace, a criacao de notebooks reproduziveis no Kaggle para o pipeline v4, e a automacao para que nenhum artefato futuro fique apenas local.

### 1.1 Problema

| Situacao | Impacto |
|----------|---------|
| Dados v4 (571 MB) existem apenas em `~/TCC/MODELO-PREVISAO/data/model_ready_v4/` | Thiago nao pode treinar, validar ou apresentar resultados |
| 18 modelos XGBoost (215 MB) existem apenas em `models/regression_v4/` e `models/classification_v4/` | Banca nao pode verificar reproducibilidade |
| Notebooks Kaggle existentes filtram apenas o DF (157 linhas de treino) | Resultados do notebook divergem do TCC |
| Repo HF `pedrolucassantanaf/dengue-tcc2-data` contem apenas dados municipais antigos (pre-v4) | Link compartilhado aponta para dados obsoletos |
| Dashboard no Streamlit Cloud depende de `dashboard/data/` commitado no Git | Se Git LFS nao estiver configurado, previsoes pre-computadas podem nao estar no clone |
| Nenhum CI/CD para publicacao | Cada mudanca exige upload manual que e esquecido |

### 1.2 Escopo do projeto local (inventario completo)

**Dados v4 — `data/model_ready_v4/` (571 MB):**

| Arquivo | Tamanho | Descricao |
|---------|---------|-----------|
| `splits_a/{train,val,test}.parquet` | 44 MB | Variante A: treino 2019–2022, val 2023, test 2024–2026 |
| `splits_b/{train,val,test}.parquet` | 125 MB | Variante B: treino 2001–2022, val 2023, test 2024–2026 |
| `splits_c/{train,val,test}.parquet` | 125 MB | Variante C: = B + sample_weight decay (lambda=0.1) |
| `sinan_mesorregiao_full.parquet` | 20 MB | SINAN agregado por mesorregiao-semana |
| `inmet_mesorregiao_full.parquet` | 6.7 MB | INMET agregado por mesorregiao-semana |
| `integrated_mesorregiao_full.parquet` | 27 MB | SINAN + INMET integrado |
| `integrated_mesorregiao_v4.parquet` | 123 MB | Integrado com anomalias |
| `integrated_mesorregiao_v4_lags.parquet` | 100 MB | Integrado com lags e features derivadas |
| `feature_schema_v4.csv` | 16 KB | Mapa feature → origem (sinan/inmet) — 266 features |
| `40–54_*.json` (14 reports) | 180 KB | Reports de cada etapa do pipeline |
| `53_tcc_outputs_summary.md` | 4 KB | Resumo dos outputs para o documento do TCC |

**Modelos v4 — `models/regression_v4/` (37 MB) + `models/classification_v4/` (178 MB):**

| Tipo | Arquivos | Tamanho total |
|------|----------|---------------|
| Regressao XGBoost (.ubj) | 9 modelos: {A,B,C} × {sinan_only, inmet_bruto, inmet_enriquecido} | 37 MB |
| Classificacao XGBoost (.ubj) | 9 modelos: {A,B,C} × {sinan_only, inmet_bruto, inmet_enriquecido}_cls | 178 MB |
| Metricas CSV | `metrics_por_{ano,uf,mesorregiao,flag_inmet}_v4.csv` | 1.2 MB |
| SHAP | `shap_importance_v4.csv` | 12 KB |
| Data Drift | `data_drift_{psi,ks}_v4.csv` | 40 KB |

**Dados municipais antigos — `data/model_ready/` (305 MB):**

| Subdiretorio | Descricao |
|--------------|-----------|
| `{train,val,test}.parquet` + metadados | E1: 6 UFs, 2019–2023, municipal |
| `e2/` | E2: historico completo, SINAN-only |
| `mesorregiao/` | Agregado por mesorregiao (pre-v4) |
| `mesorregiao_surto/` | Canal endemico de surto |
| `uf/` | Agregado por UF |

**Dashboard — `dashboard/data/` (42 MB):**

| Arquivo | Descricao |
|---------|-----------|
| `predictions_{A,B,C}_{inmet_enriquecido,inmet_bruto,sinan_only}.parquet` (9 arquivos) | Previsoes pre-computadas usadas pela dashboard em producao |

**Outros artefatos necessarios:**

| Arquivo | Tamanho | Descricao |
|---------|---------|-----------|
| `data/geojson/mesorregioes.geojson` | 596 KB | GeoJSON das 137 mesorregioes (necessario para mapa) |
| `data/model_ready_v3/municipio_mesorregiao.csv` | 500 KB | Mapeamento municipio → mesorregiao + UF |
| `config/recorte.json` | 4 KB | Configuracao do experimento (variantes A/B/C, anos, UFs) |

**Total a publicar: ~1.1 GB**

## 2. Goals

- **G-1**: Publicar 100% dos artefatos listados no inventario (secao 1.2) no HuggingFace em `pedrolucassantanaf/dengue-tcc2-data`
- **G-2**: Qualquer pessoa com acesso ao repo HF consegue reproduzir todos os resultados do TCC sem nenhum arquivo local
- **G-3**: Publicar 3 notebooks no Kaggle que treinam, avaliam e explicam o modelo v4 de mesorregioes usando dados do HF
- **G-4**: Um unico comando (`make publish`) sobe tudo para HF e Kaggle
- **G-5**: Push em `main` com mudancas em dados/modelos dispara sync automatico para HF via GitHub Actions
- **G-6**: Dataset card no HF serve como ponto de entrada para o projeto: links para dashboard, Kaggle, Overleaf, e instrucoes de uso

## 3. User Stories

### US-001: Reestruturar script de upload para v4 completo
**Description:** Como desenvolvedor, quero que o script de upload publique todos os artefatos v4 (dados, modelos, reports, config, dashboard cache, geojson) no HuggingFace para que nada do projeto fique apenas local.

**Acceptance Criteria:**
- [ ] `scripts/upload_data_to_hf.py` reescrito para suportar a estrutura v4 completa
- [ ] O script sobe as seguintes arvores para o HuggingFace:

| Origem local | Destino no HF | Conteudo |
|-------------|---------------|----------|
| `data/model_ready_v4/` | `data/model_ready_v4/` | Todos os 30 arquivos (splits, intermediarios, reports, schema) |
| `models/regression_v4/` | `models/regression_v4/` | 9 .ubj + 7 CSVs |
| `models/classification_v4/` | `models/classification_v4/` | 9 .ubj |
| `data/model_ready/` | `data/model_ready/` | Dados municipais antigos (e1, e2, meso, uf) |
| `dashboard/data/` | `dashboard/data/` | 9 parquets de previsoes pre-computadas |
| `data/geojson/` | `data/geojson/` | `mesorregioes.geojson` |
| `data/model_ready_v3/municipio_mesorregiao.csv` | `data/reference/municipio_mesorregiao.csv` | Mapeamento municipio → mesorregiao |
| `config/recorte.json` | `config/recorte.json` | Config do experimento |

- [ ] Flags de escopo:
  - `--v4` sobe apenas `data/model_ready_v4/` + `models/*_v4/` (~786 MB)
  - `--dashboard` sobe apenas `dashboard/data/` + `data/geojson/` (~43 MB)
  - `--tudo` sobe tudo listado na tabela acima (~1.1 GB)
  - `--dry-run` lista arquivos e tamanhos sem enviar
- [ ] Repo destino configuravel via `HF_REPO_ID` (padrao: `pedrolucassantanaf/dengue-tcc2-data`)
- [ ] Valida que todos os parquets existem e nao estao vazios antes de iniciar upload
- [ ] Imprime progresso durante upload e URL do dataset ao final
- [ ] Upload usa `upload_large_folder` da `huggingface_hub` para eficiencia com LFS e uploads incrementais
- [ ] Se `HF_TOKEN` nao estiver configurado, exibe mensagem clara com instrucoes

### US-002: Estrutura e organizacao no HuggingFace
**Description:** Como pesquisador ou membro da banca, quero que o dataset no HF tenha uma estrutura intuitiva para encontrar qualquer artefato em segundos.

**Acceptance Criteria:**
- [ ] Arvore final no HuggingFace:
```
pedrolucassantanaf/dengue-tcc2-data/
├── README.md                              ← Dataset card (US-007)
├── config/
│   └── recorte.json                       ← Config do experimento v4
├── data/
│   ├── geojson/
│   │   └── mesorregioes.geojson           ← 137 mesorregioes
│   ├── reference/
│   │   └── municipio_mesorregiao.csv      ← Mapeamento cod → nome/UF
│   ├── model_ready_v4/                    ← PIPELINE V4 (mesorregioes)
│   │   ├── splits_a/
│   │   │   ├── train.parquet              ← Var A treino (2019–2022)
│   │   │   ├── val.parquet                ← Var A validacao (2023)
│   │   │   └── test.parquet               ← Var A teste (2024–2026)
│   │   ├── splits_b/
│   │   │   ├── train.parquet              ← Var B treino (2001–2022)
│   │   │   ├── val.parquet
│   │   │   └── test.parquet
│   │   ├── splits_c/
│   │   │   ├── train.parquet              ← Var C treino (2001–2022 + decay)
│   │   │   ├── val.parquet
│   │   │   └── test.parquet
│   │   ├── sinan_mesorregiao_full.parquet  ← Intermediario: SINAN agregado
│   │   ├── inmet_mesorregiao_full.parquet  ← Intermediario: INMET agregado
│   │   ├── integrated_mesorregiao_full.parquet
│   │   ├── integrated_mesorregiao_v4.parquet
│   │   ├── integrated_mesorregiao_v4_lags.parquet
│   │   ├── feature_schema_v4.csv           ← 266 features: origem sinan/inmet
│   │   ├── 40_sinan_meso_report.json       ← Report etapa 40
│   │   ├── 41_inmet_meso_report.json
│   │   ├── 42_integration_report.json
│   │   ├── 43_lags_report.json
│   │   ├── 44_anomalias_report.json
│   │   ├── 45_splits_report.json
│   │   ├── 46_regression_v4_report.json    ← Metricas finais regressao
│   │   ├── 47_walkforward_v4_report.json   ← Walk-forward por ano
│   │   ├── 48_shap_report.json
│   │   ├── 49_optuna_report.json
│   │   ├── 50_drift_report.json
│   │   ├── 51_learning_curve_report.json
│   │   ├── 52_final_results_v4.json        ← Resultados consolidados
│   │   ├── 53_tcc_outputs_summary.md
│   │   └── 54_classification_v4_report.json
│   └── model_ready/                        ← PIPELINE ANTIGO (municipal)
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── feature_schema.csv
│       ├── risk_thresholds.csv
│       ├── resumo_dataset.json
│       ├── cobertura_municipal.csv
│       ├── e2/
│       ├── mesorregiao/
│       ├── mesorregiao_surto/
│       └── uf/
├── models/
│   ├── regression_v4/
│   │   ├── model_A_sinan_only.ubj          ← 1.6 MB
│   │   ├── model_A_inmet_bruto.ubj         ← 3.5 MB
│   │   ├── model_A_inmet_enriquecido.ubj   ← 2.7 MB
│   │   ├── model_B_sinan_only.ubj          ← 4.5 MB
│   │   ├── model_B_inmet_bruto.ubj         ← 5.7 MB
│   │   ├── model_B_inmet_enriquecido.ubj   ← 4.9 MB (PRINCIPAL)
│   │   ├── model_C_sinan_only.ubj          ← 3.8 MB
│   │   ├── model_C_inmet_bruto.ubj         ← 3.3 MB
│   │   ├── model_C_inmet_enriquecido.ubj   ← 7.1 MB
│   │   ├── shap_importance_v4.csv
│   │   ├── metrics_por_ano_v4.csv
│   │   ├── metrics_por_uf_v4.csv
│   │   ├── metrics_por_mesorregiao_v4.csv
│   │   ├── metrics_por_flag_inmet_v4.csv
│   │   ├── data_drift_psi_v4.csv
│   │   └── data_drift_ks_v4.csv
│   └── classification_v4/
│       ├── model_A_sinan_only_cls.ubj       ← 5.3 MB
│       ├── model_A_inmet_bruto_cls.ubj      ← 8.7 MB
│       ├── model_A_inmet_enriquecido_cls.ubj ← 6.8 MB
│       ├── model_B_sinan_only_cls.ubj       ← 28 MB
│       ├── model_B_inmet_bruto_cls.ubj      ← 30 MB
│       ├── model_B_inmet_enriquecido_cls.ubj ← 29 MB (PRINCIPAL)
│       ├── model_C_sinan_only_cls.ubj       ← 22 MB
│       ├── model_C_inmet_bruto_cls.ubj      ← 24 MB
│       └── model_C_inmet_enriquecido_cls.ubj ← 24 MB
└── dashboard/
    └── data/
        ├── predictions_A_inmet_bruto.parquet
        ├── predictions_A_inmet_enriquecido.parquet
        ├── predictions_A_sinan_only.parquet
        ├── predictions_B_inmet_bruto.parquet
        ├── predictions_B_inmet_enriquecido.parquet
        ├── predictions_B_sinan_only.parquet
        ├── predictions_C_inmet_bruto.parquet
        ├── predictions_C_inmet_enriquecido.parquet
        └── predictions_C_sinan_only.parquet
```
- [ ] Nenhum arquivo fica fora dessa arvore
- [ ] Todos os `.parquet` e `.ubj` sao armazenados automaticamente em LFS pelo HuggingFace

### US-003: Criar notebooks Kaggle para pipeline v4 mesorregioes
**Description:** Como pesquisador ou membro da banca, quero 3 notebooks publicos no Kaggle que reproduzam o pipeline v4 completo usando apenas dados do HuggingFace.

**Notebook 1 — `kaggle/v4_01_exploracao.ipynb`: Exploracao dos dados**

**Acceptance Criteria:**
- [ ] Celula de setup: instala `huggingface_hub`, configura `HF_TOKEN` via Kaggle Secrets
- [ ] Baixa splits_b (variante principal) do HF via `hf_hub_download`
- [ ] Mostra volumetria: linhas por split, numero de mesorregioes (137), numero de UFs (27), cobertura temporal (2001–2026)
- [ ] Distribuicao do alvo `notificacoes_t4`: histograma em escala log, percentis, zeros
- [ ] Distribuicao dos 4 niveis de risco (`risco_surto_t4`): contagem e proporcao
- [ ] Mapa de calor: correlacao top-20 features (usando `feature_schema_v4.csv` para separar SINAN/INMET)
- [ ] Serie temporal agregada: total de notificacoes por ano, com anotacao train/val/test
- [ ] Cobertura INMET: % de semanas-mesorregiao com dados climaticos presentes
- [ ] Celula de verificacao de vazamento: confirma que `risco_surto_t4_com_vazamento` e `threshold_source` NAO estao nas features
- [ ] Markdown explicativo em portugues entre cada celula
- [ ] Roda sem erro no Kaggle (CPU, 30GB RAM) em < 5 min

**Notebook 2 — `kaggle/v4_02_treino.ipynb`: Treinamento e avaliacao**

**Acceptance Criteria:**
- [ ] Baixa splits_a, splits_b, splits_c do HF
- [ ] Baixa `feature_schema_v4.csv` para separar features SINAN de INMET
- [ ] Treina XGBoost regressao com `log1p` no alvo, `tree_method="hist"`, `n_estimators=2000`, `early_stopping_rounds=50`
- [ ] Treina os 3 bracos de ablacao para variante B: sinan_only, inmet_bruto, inmet_enriquecido
- [ ] Opcionalmente treina variantes A e C para comparacao (celula separada, marcada como "longo ~20min")
- [ ] Calcula baselines obrigatorios: persistencia (notificacoes atual como previsao t+4) e sazonal (mesmo periodo ano anterior)
- [ ] Reporta metricas no teste: R²_log, R²_original, RMSE, MAE — para cada braco e cada baseline
- [ ] Tabela comparativa: baselines vs XGBoost, mostrando ganho relativo
- [ ] Treina classificacao de risco (4 classes) para variante B inmet_enriquecido
- [ ] Reporta metricas de classificacao: F1_macro, AUC_macro_OvR, F1 por classe (especialmente F1_surto)
- [ ] Salva modelos treinados localmente para uso no notebook 3
- [ ] Celula de verificacao: compara metricas com os valores de referencia (`46_regression_v4_report.json`)
- [ ] Roda no Kaggle (CPU, 30GB RAM) em < 30 min para variante B completa

**Notebook 3 — `kaggle/v4_03_resultados.ipynb`: Resultados e visualizacoes**

**Acceptance Criteria:**
- [ ] Baixa modelos treinados do HF (`models/regression_v4/`, `models/classification_v4/`)
- [ ] OU usa modelos treinados no notebook 2 (detecta automaticamente)
- [ ] Scatter plot: previsao vs real em escala log, colorido por nivel de risco, com linha ideal tracejada
- [ ] Serie temporal nacional: soma semanal real vs prevista (todas as mesorregioes)
- [ ] R² por ano: bar chart colorido por split (treino/validacao/teste)
- [ ] Metricas por UF: tabela ou heatmap mostrando R² e MAE por UF
- [ ] Walk-forward: treino expandido ano a ano, R² por fold, grafico de linhas por variante
- [ ] SHAP: carrega `shap_importance_v4.csv` do HF, bar chart horizontal top-20 features com nomes legíveis em portugues
- [ ] Comparativo de variantes: bar chart A vs B vs C com intervalos de confianca
- [ ] Matriz de confusao: classificacao de risco (4x4) para variante B
- [ ] Todas as figuras usam estilo consistente (Plotly, font DM Sans, cores do design system)
- [ ] Roda no Kaggle (CPU, 30GB RAM) em < 10 min

**Requisitos comuns aos 3 notebooks:**
- [ ] Detectam ambiente automaticamente: `IS_KAGGLE = "KAGGLE_DATA_DIR" in os.environ`
- [ ] No Kaggle: usam `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")`
- [ ] Local: usam `os.environ.get("HF_TOKEN")` ou token salvo via `huggingface-cli login`
- [ ] Repo HF: `pedrolucassantanaf/dengue-tcc2-data` (hardcoded com variavel para override)
- [ ] Nenhum filtra por municipio ou DF — usam todas as 137 mesorregioes
- [ ] Colunas proibidas como features: `notificacoes_t4`, `risco_surto_t4`, `risco_surto_t4_com_vazamento`, `threshold_source`, `cod_mesorregiao`, `ano`, `semana_epidemiologica`
- [ ] Headers markdown com badges: versao do dataset, data de execucao, link para dashboard
- [ ] Cada notebook comeca com `%%capture` para instalar dependencias silenciosamente

### US-004: Script de publicacao no Kaggle via API
**Description:** Como desenvolvedor, quero um script que publique e atualize os 3 notebooks no Kaggle automaticamente.

**Acceptance Criteria:**
- [ ] `scripts/publish_kaggle.py` criado
- [ ] Cada notebook tem `kernel-metadata.json` em `kaggle/`:
  ```json
  {
    "id": "pedrolucassantanaf/denguecast-v4-01-exploracao",
    "title": "DengueCast v4 — Exploracao dos Dados",
    "code_file": "v4_01_exploracao.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": false,
    "enable_gpu": false,
    "enable_internet": true,
    "keywords": ["dengue", "xgboost", "epidemiologia", "brasil", "time-series"],
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
  }
  ```
- [ ] `python3 scripts/publish_kaggle.py` publica/atualiza os 3 notebooks de uma vez
- [ ] `python3 scripts/publish_kaggle.py --notebook v4_01` publica apenas um
- [ ] `python3 scripts/publish_kaggle.py --dry-run` mostra o que seria publicado
- [ ] Usa `kaggle kernels push` internamente
- [ ] Verifica que `~/.kaggle/kaggle.json` existe e tem credenciais validas
- [ ] Imprime URLs dos notebooks publicados ao final

### US-005: Makefile unificado de publicacao
**Description:** Como desenvolvedor, quero um unico ponto de entrada para publicar tudo, com validacao pre-upload.

**Acceptance Criteria:**
- [ ] `Makefile` na raiz do projeto com os seguintes targets:

| Target | Acao |
|--------|------|
| `make publish` | Valida → sobe HF (`--tudo`) → sobe Kaggle → imprime resumo |
| `make publish-hf` | Sobe apenas HuggingFace |
| `make publish-hf-v4` | Sobe apenas dados e modelos v4 |
| `make publish-hf-dashboard` | Sobe apenas cache da dashboard |
| `make publish-kaggle` | Publica notebooks no Kaggle |
| `make publish-dry` | Dry-run de tudo (HF + Kaggle) |
| `make validate` | Valida integridade dos artefatos locais sem subir |
| `make status` | Mostra o que esta local vs publicado |

- [ ] `make validate` verifica:
  - Todos os 9 splits existem e nao estao vazios
  - Todos os 18 modelos `.ubj` existem
  - Todos os 14 reports JSON existem e sao JSON valido
  - `feature_schema_v4.csv` existe e tem 266+ linhas
  - Todos os 9 predictions da dashboard existem
  - `HF_TOKEN` esta configurado (para targets de upload)
  - `~/.kaggle/kaggle.json` existe (para target Kaggle)
- [ ] `make status` compara arquivos locais com o que esta no HF (via `huggingface_hub.list_repo_tree`)
- [ ] Cada target imprime URLs finais ao concluir com sucesso
- [ ] Se alguma validacao falha, o target aborta com mensagem clara

### US-006: GitHub Action para sync automatico com HuggingFace
**Description:** Como desenvolvedor, quero que pushes em `main` que alterem dados ou modelos disparem upload automatico para o HuggingFace.

**Acceptance Criteria:**
- [ ] `.github/workflows/sync-hf.yml` criado com:
  ```yaml
  name: Sync to HuggingFace
  on:
    push:
      branches: [main]
      paths:
        - 'data/model_ready_v4/**'
        - 'models/regression_v4/**'
        - 'models/classification_v4/**'
        - 'dashboard/data/**'
        - 'data/geojson/**'
        - 'config/recorte.json'
    workflow_dispatch:  # trigger manual
  ```
- [ ] Usa `HF_TOKEN` como GitHub Repository Secret
- [ ] Instala `huggingface_hub` e roda `scripts/upload_data_to_hf.py --tudo`
- [ ] Idempotente: nao falha se nenhum arquivo mudou (upload incremental via `upload_large_folder`)
- [ ] Timeout: 30 minutos (upload de 1.1 GB pode demorar)
- [ ] Job separado para Kaggle: trigger apenas por `workflow_dispatch` (manual)
  ```yaml
  name: Publish Kaggle Notebooks
  on:
    workflow_dispatch:
  ```
- [ ] Usa `KAGGLE_USERNAME` e `KAGGLE_KEY` como GitHub Secrets
- [ ] Badge no `README.md` do projeto: `![HF Sync](https://github.com/.../workflows/sync-hf/badge.svg)`

### US-007: Dataset card (README) no HuggingFace
**Description:** Como visitante do HuggingFace, quero um README completo que sirva como ponto de entrada para todo o projeto.

**Acceptance Criteria:**
- [ ] Arquivo `hf_dataset_card/README.md` criado localmente (subido junto com os dados)
- [ ] Header YAML com metadados:
  ```yaml
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
  size_categories:
    - 100K<n<1M
  ---
  ```
- [ ] Secoes do README:
  1. **Titulo e descricao**: DengueCast — Sistema Preditivo de Surtos de Dengue
  2. **Links rapidos**: Dashboard (Streamlit Cloud URL), Notebooks Kaggle (3 URLs), Repositorio GitHub
  3. **Sobre o projeto**: TCC em Engenharia de Software, UnB FGA, 2024–2025. Autores: Pedro Lucas F. Santana e Thiago Ribeiro Freitas
  4. **Dados**: descricao de cada camada (v4 mesorregioes, municipal antigo), volumetria, cobertura temporal e espacial
  5. **Modelos**: 18 XGBoost (3 variantes × 3 sub-modelos × 2 tarefas), metricas principais (R²_log = 0.8878 para B_inmet_enriquecido)
  6. **Como usar**: snippet Python completo para carregar dados e modelos via `huggingface_hub`
  7. **Estrutura de pastas**: arvore completa com descricao de cada diretorio
  8. **Variantes experimentais**: tabela A vs B vs C com descricao, janela de treino e resultados
  9. **Fontes de dados**: SINAN (notificacoes epidemiologicas, Ministerio da Saude) e INMET (estacoes meteorologicas)
  10. **Citacao**: formato BibTeX para citar o TCC
  11. **Licenca**: CC-BY-4.0 para dados, MIT para codigo
- [ ] Upload do README como arquivo raiz do dataset no HF

### US-008: Validacao pos-publicacao
**Description:** Como desenvolvedor, quero verificar que tudo foi publicado corretamente e e acessivel.

**Acceptance Criteria:**
- [ ] `scripts/validate_hf_publication.py` criado
- [ ] Conecta ao repo HF e lista todos os arquivos publicados
- [ ] Compara com inventario local: reporta arquivos faltantes ou com tamanho divergente
- [ ] Tenta baixar um arquivo de cada categoria (split, modelo, report, prediction) e verifica integridade
- [ ] Para splits: carrega o parquet e verifica numero de linhas vs `45_splits_report.json`
- [ ] Para modelos: carrega o `.ubj` com `xgb.Booster()` e verifica que nao esta corrompido
- [ ] Imprime relatorio final: total de arquivos, tamanho total, status (OK / FALHA)
- [ ] `make validate-remote` roda este script

## 4. Functional Requirements

- FR-1: O script de upload deve suportar toda a arvore `data/model_ready_v4/` incluindo subdiretorios `splits_a/`, `splits_b/`, `splits_c/`
- FR-2: Modelos `.ubj` devem ser subidos para `models/` no HF, separados de `data/`, mantendo a mesma hierarquia local
- FR-3: CSVs de metricas, SHAP e data drift devem acompanhar os modelos no mesmo diretorio
- FR-4: Notebooks Kaggle devem usar `hf_hub_download` (nao `load_dataset`) para eficiencia de memoria — `load_dataset().to_pandas()` duplica a memoria via Arrow
- FR-5: Notebooks devem treinar em `log1p(notificacoes_t4)` e reportar metricas em ambas as escalas (R²_log e R²_original)
- FR-6: Notebooks devem incluir baselines (persistencia e sazonal) como referencia obrigatoria antes de qualquer metrica de XGBoost
- FR-7: A GitHub Action deve usar `upload_large_folder` para uploads incrementais (nao re-sobe arquivos que nao mudaram)
- FR-8: Todos os 14 reports JSON de `data/model_ready_v4/` devem ser subidos sem excecao
- FR-9: O `feature_schema_v4.csv` deve ser subido e usado pelos notebooks para separar features SINAN de INMET — nunca separar por nome de coluna
- FR-10: O script de upload deve validar integridade dos arquivos antes de subir: parquets legiveis, JSONs validos, UBJs nao vazios
- FR-11: As 9 previsoes pre-computadas da dashboard (`dashboard/data/predictions_*.parquet`) devem ser publicadas para que a dashboard possa ser reconstruida sem rodar inferencia
- FR-12: O `config/recorte.json` deve ser publicado para documentar as decisoes experimentais (anos, variantes, decay lambda)
- FR-13: O GeoJSON deve ser publicado para que a dashboard ou notebooks possam renderizar mapas sem dependencia local
- FR-14: Notebooks nao devem conter paths absolutos — apenas caminhos relativos ao HuggingFace ou ao ambiente Kaggle

## 5. Non-Goals (Out of Scope)

- **NAO** subir dados brutos Bronze (SINAN DBF/DBASE, INMET ZIP horarios) — sao 2+ GB de dados governamentais que podem ser baixados das fontes originais
- **NAO** criar interface no HuggingFace Spaces — a dashboard ja existe no Streamlit Cloud (`modelo-previsao-ffqc3ugltpbxxeaa4mygvw.streamlit.app`)
- **NAO** migrar a dashboard para o Kaggle ou HF Spaces
- **NAO** subir para o repo do Thiago (`thiagorfreitas/dengue-tcc2-data`) — ele pode ter acesso de leitura ou escrita ao repo principal
- **NAO** automatizar retreino de modelos — este PRD cobre apenas publicacao de artefatos ja existentes
- **NAO** subir os scripts de pipeline (`scripts/40–55_*.py`) para o HF — esses ficam no GitHub
- **NAO** subir dados intermediarios v2/v3 (`data/model_ready_v2/`, `data/model_ready_v3/`) — sao versoes obsoletas

## 6. Technical Considerations

### Dependencias
| Pacote | Versao | Uso |
|--------|--------|-----|
| `huggingface_hub` | >=0.20 | Upload/download de arquivos para HuggingFace |
| `kaggle` | >=1.6 | Publicacao de notebooks no Kaggle |
| `xgboost` | >=2.0 | Validacao de integridade dos modelos |
| `pandas` | >=2.0 | Validacao de integridade dos parquets |

### Limites e restricoes
- **HuggingFace**: limite de 300 GB por dataset (estamos em ~1.1 GB). LFS automatico para arquivos > 10 MB
- **Kaggle**: notebooks tem 30 GB RAM e 12h de execucao. E1 cabe com folga. Internet deve estar habilitada para `hf_hub_download`
- **GitHub Actions**: 6h de timeout maximo. Upload de 1.1 GB leva ~5-15 min dependendo da conexao do runner
- **Git LFS**: o repo GitHub ja tem `.parquet` e `.ubj` no Git. Esses mesmos arquivos vao para o HuggingFace LFS separadamente — sao coisas independentes
- **Token HF**: precisa de permissao de escrita (`write`) no repo `pedrolucassantanaf/dengue-tcc2-data`

### Seguranca
- `HF_TOKEN` e `KAGGLE_KEY` nunca devem ser commitados no codigo — usar variavel de ambiente ou GitHub Secrets
- Notebooks publicados no Kaggle nao devem conter tokens — o token vem de Kaggle Secrets

## 7. Success Metrics

| Metrica | Criterio |
|---------|----------|
| Cobertura | 100% dos arquivos do inventario (secao 1.2) publicados no HF |
| Reproducibilidade | Os 3 notebooks Kaggle rodam sem erro e produzem R²_log dentro de ±0.005 do valor de referencia (0.8878) |
| Tempo de publicacao | `make publish` completa em < 15 min (primeira vez) e < 5 min (incremental) |
| Automatizacao | GitHub Action roda sem falha em pushes que alteram dados/modelos |
| Acesso | Thiago baixa dados e modelos em < 5 min e consegue treinar um modelo |
| Integridade | `make validate-remote` retorna 100% OK para todos os artefatos |

## 8. Ordem de Implementacao

| Fase | Stories | Descricao | Estimativa |
|------|---------|-----------|------------|
| 1 | US-001, US-002 | Reescrever script de upload, subir tudo para HF | 2h |
| 2 | US-007 | Dataset card no HF | 30min |
| 3 | US-003 | Criar 3 notebooks Kaggle v4 | 4h |
| 4 | US-004 | Script de publicacao Kaggle | 1h |
| 5 | US-005 | Makefile unificado | 1h |
| 6 | US-006 | GitHub Action CI/CD | 1h |
| 7 | US-008 | Validacao pos-publicacao | 1h |
| **Total** | | | **~10h** |

A fase 1 e a mais urgente: Thiago precisa dos dados agora. As fases 3-7 podem ser feitas depois.

## 9. Open Questions

1. Thiago precisa de acesso de **escrita** ao repo HF, ou apenas **leitura** (download)?
2. Os notebooks Kaggle devem ficar no perfil `pedrolucassantanaf` ou outro?
3. Licenca dos dados: **CC-BY-4.0** (recomendado para dados academicos) ou outra?
4. Incluir os dados INMET Silver (~200 MB) no HF para reprodutibilidade do pipeline de integracao, ou apenas os dados model-ready?
5. O HF_TOKEN do Pedro ja tem permissao de escrita no repo? Ja foi feito `huggingface-cli login`?

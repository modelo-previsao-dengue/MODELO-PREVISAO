# PRD: Treinamento XGBoost com Série Histórica Completa

## 1. Resumo Executivo

A baseline v3 do TCC — XGBoost mesorregional com SINAN+INMET, treinada em **4 anos** (2019-2022, com warm-up 2018) — atingiu R²_log = 0.8698 (modelo C, INMET enriquecido). Este PRD especifica o experimento que treina o mesmo modelo com **toda a série histórica disponível (2000-2026)**, compara estatisticamente com a baseline e produz tabelas/figuras prontos para o texto final do TCC, cuja entrega é ~30/11/2026.

O experimento responde à pergunta central: **"Incluir mais de duas décadas de dados epidemiológicos e climáticos melhora a previsão de dengue a 4 semanas?"**

---

## 2. Contexto e Problema

### O que a baseline v3 mostrou

A pipeline v3 (`scripts/20_*` a `scripts/33_*`) agrega dados de 5.570 municípios em 137 mesorregiões, integra 12 variáveis climáticas do INMET via média por mesorregião, e treina 3 variantes de XGBoost com horizonte de previsão t+4 semanas:

| Modelo | Features | R²_log | RMSE | MAE | p-value (vs A) |
|--------|----------|--------|------|-----|----------------|
| A: SINAN-only | 110 | 0.8588 | 2203.72 | 271.75 | — |
| B: INMET bruto | 122 | 0.8676 | 2121.34 | 252.59 | 1.55e-44 |
| C: INMET enriquecido | 266 | 0.8698 | 2103.17 | 252.17 | 2.41e-33 |

Baselines ingênuos no mesmo teste: persistência R²_log = 0.8078, média móvel 4w = 0.7573, sazonal = 0.3564.

Walk-forward (4 folds expansivos, 2023-2026): modelo C supera A em 4/4 folds (média R²_log 0.8154 vs 0.7994).

**Recorte temporal v3**: warm-up=2018, treino=[2019,2020,2021,2022], val=[2023], teste=[2024,2025,2026]. Definido em `config/recorte.json` → `split_v3`.

### Por que testar a série completa

1. **Mais evidências sazonais/epidêmicas**: anos como 2002, 2008, 2010, 2013, 2015-2016 tiveram grandes surtos (>800k notificações/ano) que o treino atual não vê.
2. **Argumento acadêmico**: o TCC precisa demonstrar, com evidência, se dados históricos adicionais melhoram ou não a previsão — não basta assumir.
3. **Robustez do modelo**: treinar só em 4 anos (2019-2022) limita a diversidade de regimes climáticos e epidemiológicos vistos pelo modelo.

### Estado inconsistente do pipeline v3 em disco

> **ALERTA**: Os arquivos Parquet em disco e os relatórios JSON não são da mesma execução.

**Parquets (gerados em 08/09/2026)** — execução parcial com INMET Silver incompleto (anos 2009, 2020 e 2022 ausentes no disco):

| Arquivo | Linhas | Anos presentes |
|---------|--------|---------------|
| `sinan_mesorregiao_weekly.parquet` | 38.086 | 2019, 2021, 2023-2026 (6 anos) |
| `integrated_mesorregiao_v3.parquet` | 38.086 | idem |
| `train_v3.parquet` | 14.248 (272 cols) | 2019, 2021 (apenas 2 anos) |
| `val_v3.parquet` | 7.261 | 2023 |
| `test_v3.parquet` | 16.029 | 2024-2026 |

**Relatórios JSON (gerados em 28/09/2026)** — compilação que preservou métricas de uma execução anterior completa:

| Relatório | Valor registrado | Observação |
|-----------|-----------------|------------|
| `23_integration_report.json` | 60.554 linhas, 9 anos (2018-2026) | Cobertura INMET: 2020=89.3%, 2022=87.3% |
| `26_splits_v3_report.json` | train=28.633, anos=[2019-2022] | 4 anos completos |
| `27_regression_v3_report.json` | R²_log C=0.8698 | Métricas da execução completa |

**Causa raiz**: Os INMET Silver para 2020 e 2022 existiam quando o pipeline v3 rodou pela primeira vez (o integration report confirma cobertura de 89.3% e 87.3%). O `config/recorte.json` nota: *"a versão anterior pulava 2020 e 2022 (ZIPs do INMET fora do inventário)"*. Esses silver files foram **apagados do disco** posteriormente. A re-execução em 08/09 gerou parquets degradados (sem 2018, 2020, 2022 no SINAN mesorregional).

**Implicação**: Antes de iniciar qualquer trabalho v4, é obrigatório recuperar o estado consistente do pipeline v3 (ver seção 6.4 — Pré-requisitos).

---

## 3. Objetivos e Não-Objetivos

### Objetivos

- **O1**: Restaurar o pipeline v3 a um estado consistente e reproduzir a baseline (R²_log=0.8698 para modelo C).
- **O2**: Treinar e avaliar variantes com a série completa e com peso por recência.
- **O3**: Produzir comparação estatística (IC bootstrap, t-test pareado) entre variantes.
- **O4**: Gerar tabelas LaTeX, figuras e resumo Markdown prontos para o Overleaf.
- **O5**: Responder se a série completa melhora, piora ou não altera o modelo.

### Não-Objetivos

- **NG1**: Trocar o algoritmo (e.g., LightGBM, redes neurais). O modelo é XGBoost.
- **NG2**: Alterar a granularidade (mesorregião × semana) ou o target (`notificacoes_t4`).
- **NG3**: Recalcular o pipeline SINAN Bronze→Silver→Gold.
- **NG4**: Re-tunar hiperparâmetros extensivamente. Optuna limitado a orçamento de N trials (ver RF-09).

---

## 4. Hipóteses

- **H1**: A série completa (2001-2022) reduz o erro em mesorregiões com pouco histórico recente, pois o modelo vê mais regimes epidêmicos.
- **H2**: Dados antigos (pré-2010) com cobertura INMET esparsa adicionam ruído que pode neutralizar o ganho de H1.
- **H3**: Ponderar por recência (decaimento exponencial) mitiga o data drift de anos antigos e supera a série completa sem peso.
- **H4**: As features INMET se tornam mais úteis com a série completa porque a variabilidade climática aumenta.

---

## 5. Estado Atual do Pipeline

### 5.1 Mapeamento do pipeline v3

**Fato verificado no código:**

| # | Script | Função principal | Entrada | Saída |
|---|--------|-----------------|---------|-------|
| 20 | `20_build_mesorregiao_mapping.py` | `build_municipio_mesorregiao()`, `build_estacao_mesorregiao()` | `data/reference/ibge_municipios_api.json`, `data/inmet/bronze/estacoes_inmet.csv` | `municipio_mesorregiao.csv` (5.570 municípios→137 mesos), `estacao_mesorregiao.csv` (~700 estações) |
| 21 | `21_aggregate_sinan_mesorregiao.py` | `main()` | SINAN Gold Dense (particionado `year=*/*.parquet`) | `sinan_mesorregiao_weekly.parquet` |
| 22 | `22_aggregate_inmet_mesorregiao.py` | `main()` | INMET Silver (`weekly_stations_*.parquet`) | `inmet_mesorregiao_weekly.parquet` |
| 23 | `23_integrate_mesorregiao.py` | `main()` | sinan_meso + inmet_meso (LEFT JOIN) | `integrated_mesorregiao.parquet` |
| 24 | `24_feature_eng_lags_v3.py` | | integrated_mesorregiao | `integrated_mesorregiao_v3.parquet` (252 cols) |
| 25 | `25_feature_eng_anomalias_v3.py` | | integrated_v3 + climatologia(train-only) | `integrated_mesorregiao_v3.parquet` (276 cols) |
| 26 | `26_prepare_splits_v3.py` | `main()`, `compute_risk_class()` | integrated_v3 + `config/recorte.json` | `train_v3.parquet`, `val_v3.parquet`, `test_v3.parquet`, `feature_schema_v3.csv` |
| 27 | `27_train_regression_v3.py` | `train_and_eval()` | splits v3 | 3 modelos `.ubj` + `27_regression_v3_report.json` |
| 28 | `28_train_blindness_v3.py` | | splits v3 | report JSON (data blindness test) |
| 29 | `29_train_classification_v3.py` | | splits v3 | modelos classificação + report JSON |
| 30 | `30_explain_shap_v3.py` | | modelo C + test | SHAP report + figuras PNG |
| 31 | `31_validate_walkforward_v3.py` | `build_folds()` | integrated_v3 (completo) | walk-forward report + figuras |
| 32 | `32_compile_final_v3.py` | | todos os reports | `32_final_results_v3.json` |
| 33 | `33_baselines_v3.py` | | test split | baselines report (persistência, média móvel, sazonal) |

**Execução**: `scripts/rodar_v3.sh [primeiro_script]` roda scripts 20-33 em sequência, parando no primeiro erro (`set -euo pipefail`).

### 5.2 Onde o recorte temporal está definido

**Fato verificado:** O recorte está em `config/recorte.json`, chave `split_v3`:

```json
"split_v3": {
    "warmup_ano": 2018,
    "ano_fim": 2026,
    "train": [2019, 2020, 2021, 2022],
    "val": [2023],
    "test": [2024, 2025, 2026]
}
```

O helper `scripts/recorte_config.py` → `load_v3()` carrega isso e calcula `anos_pipeline = range(warmup_ano, ano_fim+1)` → [2018, 2019, ..., 2026].

**Dependências do recorte** (todos usam `recorte_config.load_v3()`):
- Script 21, linha 23: `VALID_YEARS = recorte_config.load_v3()["anos_pipeline"]` — filtra quais anos do SINAN Gold entram.
- Script 26, linhas 24-27: `TRAIN_YEARS`, `VAL_YEARS`, `TEST_YEARS` vêm do config.
- Script 25: climatologia calculada apenas nos `TRAIN_YEARS`.
- Script 31: `build_folds()` usa `split["train"]` + `split["val"]` + `split["test"]`.

### 5.3 Variável alvo, granularidade e métricas

**Fato verificado:**
- **Granularidade**: mesorregião × semana epidemiológica (137 mesorregiões × ~52 semanas)
- **Target de regressão**: `notificacoes_t4` = `notificacoes.shift(-4)` por mesorregião (previsão 4 semanas à frente)
- **Target de classificação**: `risco_surto_t4` — 4 classes (0=baixo, 1=moderado, 2=alto, 3=muito_alto) baseadas em percentis (p50, p75, p90) calculados **somente no treino**
- **Transformação**: `log1p(target)` para treino/avaliação
- **Métricas de regressão**: R²_log, R², RMSE, RMSE_log, MAE
- **Métricas de classificação**: accuracy, F1_macro, F1_weighted, AUC_macro_ovr, F1 por classe
- **Validação**: walk-forward com expanding window (folds anuais expansivos), early stopping com N_VAL_WEEKS=10 semanas do treino como val interna

### 5.4 Hiperparâmetros da baseline

**Fato verificado** (definidos inline em `scripts/27_train_regression_v3.py` e `scripts/31_validate_walkforward_v3.py`):

```python
XGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)
```

O script Optuna existente (`scripts/tune_xgb_optuna.py`, N_TRIALS=20) opera sobre o pipeline v1 (municipal, `data/model_ready/`), não sobre v3. Os hiperparâmetros de v3 foram definidos manualmente.

### 5.5 Mesorregião no modelo

**Fato verificado**: `cod_mesorregiao` é uma feature numérica (int) nos splits. Não é categórica nativa, não é one-hot, não é target-encoded.

**Decisão**: Manter essa abordagem em todas as variantes para comparabilidade.

### 5.6 Contagem de features por categoria

**Fato verificado** (`26_splits_v3_report.json` → `feature_schema_v3.csv`):

| Categoria | Qtd features | Exemplos |
|-----------|-------------|----------|
| SINAN (epidemiológicas) | 110 | `notificacoes`, `tx_incidencia`, lags `_lag1`..`_lag8` |
| INMET bruto | 12 | `temp_media`, `precipitacao`, `umidade_media`, etc. |
| INMET lags biológicos | 96 | `temp_media_lag1`..`_lag8` × 12 variáveis |
| INMET médias móveis | 24 | `temp_media_mm4`, `_mm8` × 12 variáveis |
| INMET anomalias | 24 | `temp_media_anom4`, `_anom8` × 12 variáveis |
| **Total** | **266** | |

Sub-modelos: A (SINAN-only)=110, B (SINAN+INMET bruto)=122, C (SINAN+INMET enriquecido)=266.

---

## 6. Inventário de Dados por Ano

### 6.1 SINAN Gold (municipal, formato denso)

**Fato verificado** (`data/sinan/gold/sinan_tcc2_v2/official_dense/year=*/`, 47 parquet files por partição):

| Ano | Linhas municipais (Gold Dense) | Linhas após agregação meso (estimada) | Municípios ativos | Notificações | Notas |
|-----|-------------------------------|---------------------------------------|-------------------|--------------|-------|
| 1999 | 5.565 | ~1.507 | 8 | 13 | Praticamente vazio |
| 2000 | 294.945 | ~7.124 | 2.087 | 171.654 | Cobertura parcial |
| 2001 | ~291.000 | ~7.124 | 3.512 | 485.989 | |
| 2002 | ~290.000 | ~7.124 | 4.303 | 885.534 | Grande surto |
| 2003 | ~290.000 | ~7.124 | 3.834 | 416.298 | |
| 2004 | 289.405 | ~7.124 | 3.079 | 136.524 | Ano com menor volume |
| 2005 | ~289.000 | ~7.124 | 3.404 | 260.443 | |
| 2006 | ~289.000 | ~7.124 | 3.788 | 410.039 | |
| 2007 | ~289.000 | ~7.124 | 4.324 | 714.498 | |
| 2008 | ~290.000 | ~7.124 | 4.414 | 914.831 | Grande surto |
| 2009 | ~289.000 | ~7.124 | 3.986 | 599.275 | |
| 2010 | 289.385 | ~7.124 | 4.572 | 1.371.679 | Megassurto |
| 2011 | ~289.000 | ~7.124 | 4.724 | 1.144.368 | |
| 2012 | ~289.000 | ~7.124 | 4.410 | 944.889 | |
| 2013 | 289.699 | ~7.124 | 4.799 | 2.022.238 | Megassurto |
| 2014 | ~290.000 | ~7.124 | 4.494 | 966.369 | |
| 2015 | ~289.000 | ~7.124 | 4.951 | 2.397.091 | Megassurto + Zika |
| 2016 | ~289.000 | ~7.124 | 5.132 | 2.296.878 | Megassurto |
| 2017 | ~289.000 | ~7.124 | 4.495 | 518.345 | |
| 2018 | ~289.000 | ~7.124 | 4.350 | 478.768 | Warm-up v3 |
| 2019 | 289.380 | ~7.124 | 4.991 | 2.261.441 | Train v3 |
| 2020 | ~290.000 | ~7.124 | 4.721 | 1.494.587 | Train v3 (COVID) |
| 2021 | ~289.000 | ~7.124 | 4.477 | 1.009.791 | Train v3 |
| 2022 | ~289.000 | ~7.124 | 5.018 | 1.404.535 | Train v3 |
| 2023 | ~290.000 | ~7.124 | 4.992 | 1.644.709 | Val v3 |
| 2024 | 289.464 | ~7.124 | 5.485 | 6.560.720 | Test v3 (recorde) |
| 2025 | ~290.000 | ~7.124 | 5.156 | 1.642.302 | Test v3 |
| 2026 | 90.026 | ~2.329 | 4.388 | 258.079 | Test v3 (parcial, até SE16) |

**Notas sobre a tabela**:
- "Linhas municipais" = total real de rows no SINAN Gold Dense (formato denso: ~5.565 municípios × 52-53 semanas).
- "Linhas após agregação meso" = estimativa: 137 mesorregiões × semanas do ano.
- Valores marcados com `~` são interpolados dos anos verificados (2000, 2004, 2010, 2013, 2019, 2024); todos seguem o padrão 289k-295k.
- Cada ano tem 47 parquet files e **142 colunas** (141 features + partição `year`).
- Fonte de notificações: `data/sinan/governance/sinan_tcc2_v2/coverage_temporal.json`.

### 6.2 INMET Silver (estação-semana)

**Fato verificado** (`data/inmet/silver/weekly_stations_*.parquet` + `coverage_report.csv`):

| Ano | Estações | Registros | % baixa cobertura | Silver em disco? | Notas |
|-----|----------|-----------|-------------------|-----------------|-------|
| 2000 | 5 | 156 | 21.8% | ✅ | Quase inexistente |
| 2001 | 16 | 431 | 13.9% | ✅ | |
| 2002 | 33 | 998 | 15.9% | ✅ | |
| 2003 | 55 | 2.722 | 26.8% | ✅ | |
| 2004 | 62 | 2.949 | 32.1% | ✅ | |
| 2005 | 64 | 3.346 | 22.6% | ✅ | |
| 2006 | 154 | 4.672 | 17.1% | ✅ | Salto: estações automáticas |
| 2007 | 264 | 10.923 | 10.2% | ✅ | |
| 2008 | 417 | 19.016 | 5.2% | ✅ | |
| **2009** | **—** | **—** | **—** | **❌** | **Silver apagado do disco** |
| 2010 | 446 | 23.265 | 9.9% | ✅ | |
| 2011 | 456 | 23.884 | 9.7% | ✅ | |
| 2012 | 468 | 24.351 | 6.6% | ✅ | |
| 2013 | 473 | 24.886 | 6.6% | ✅ | |
| 2014 | 475 | 25.092 | 6.8% | ✅ | |
| 2015 | 484 | 25.366 | 10.6% | ✅ | |
| 2016 | 529 | 26.395 | 10.4% | ✅ | |
| 2017 | 563 | 28.928 | 11.9% | ✅ | |
| 2018 | 596 | 31.049 | 11.3% | ✅ | |
| 2019 | 589 | 30.973 | 10.9% | ✅ | |
| **2020** | **—** | **—** | **—** | **❌** | **Silver apagado** (existia: integ report 89.3%) |
| 2021 | 588 | 31.164 | 44.8% | ✅ | Alta % baixa cobertura |
| **2022** | **—** | **—** | **—** | **❌** | **Silver apagado** (existia: integ report 87.3%) |
| 2023 | 567 | 30.051 | 16.7% | ✅ | |
| 2024 | 565 | 29.945 | 19.7% | ✅ | |
| 2025 | 594 | 30.392 | 30.6% | ✅ | |
| 2026 | 633 | 11.214 | 38.5% | ✅ | Parcial |

### 6.3 Cobertura INMET por mesorregião — dados verificados

**Fato verificado** (`data/model_ready_v3/inmet_mesorregiao_weekly.parquet`, 123.551 linhas, 26 anos):

| Ano | Mesorregiões com dados | % de 137 | Semanas | Classificação |
|-----|----------------------|----------|---------|---------------|
| 2000 | 5 | 4% | 35 | Muito esparso |
| 2001 | 16 | 12% | 52 | Muito esparso |
| 2002 | 27 | 20% | 52 | Esparso |
| 2003 | 43 | 31% | 52 | Esparso |
| 2004 | 49 | 36% | 52 | Parcial |
| 2005 | 50 | 36% | 52 | Parcial |
| 2006 | 84 | 61% | 53 | Razoável |
| 2007 | 111 | 81% | 52 | Bom |
| 2008 | 128 | 93% | 52 | Bom |
| 2009 | 129 | 94% | **1** | ⚠️ Esparso temporal (1 semana apenas) |
| 2010 | 131 | 96% | 52 | Excelente |
| 2011 | 131 | 96% | 52 | Excelente |
| 2012 | 132 | 96% | 53 | Excelente |
| 2013 | 133 | 97% | 52 | Excelente |
| 2014 | 133 | 97% | 52 | Excelente |
| 2015 | 133 | 97% | 52 | Excelente |
| 2016 | 136 | 99% | 52 | Excelente |
| 2017 | 137 | 100% | 53 | Completo |
| 2018 | 137 | 100% | 52 | Completo |
| 2019 | 136 | 99% | 52 | Completo |
| 2020 | 136 | 99% | **1** | ⚠️ Esparso temporal (1 semana apenas) |
| 2021 | 136 | 99% | 52 | Completo |
| 2023 | 136 | 99% | 53 | Completo |
| 2024 | 135 | 99% | 52 | Completo |
| 2025 | 135 | 99% | 52 | Completo |
| 2026 | 135 | 99% | 17 | Parcial (ano em curso) |

**Observações importantes**:
1. **2009 e 2020**: Apesar de alta cobertura espacial (94-99% das mesorregiões), possuem **apenas 1 semana de dados**. Para o modelo semanal, são efetivamente sem cobertura climática (51/52 semanas serão NaN).
2. **2022**: Não aparece no INMET mesorregião porque o silver foi apagado. O integration report confirma que existia com 87.3% de cobertura.
3. **2000-2005**: Cobertura espacial muito baixa (4-36% das mesorregiões). A maioria das mesorregiões nesses anos terá NaN em features INMET.
4. **2006-2008**: Transição — a rede de estações automáticas do INMET expandiu de 154 para 417 estações.

### 6.4 Pré-requisitos obrigatórios (antes de iniciar a implementação v4)

A implementação v4 **depende** de um estado consistente do pipeline v3. Os pré-requisitos abaixo devem ser cumpridos antes dos RFs:

#### PRE-01: Recuperar INMET Silver para 2020 e 2022

**Status**: OBRIGATÓRIO — sem eles, CA-01 (reproduzir baseline) é impossível.

- Re-baixar os ZIPs do INMET para 2020 e 2022 do portal do INMET
- Rodar o pipeline Bronze→Silver para gerar `weekly_stations_2020.parquet` e `weekly_stations_2022.parquet`
- Validar que os arquivos regenerados têm estações e registros compatíveis com os coverage_report valores históricos (2020: ~589 estações, 2022: ~567 estações)

**Nota sobre 2009**: O silver de 2009 também está ausente, mas como o mesorregião file mostra apenas 1 semana de dados (mesmo quando existia), sua recuperação é **opcional**. Se recuperado, terá impacto mínimo (1 semana para 129 mesorregiões).

#### PRE-02: Re-executar pipeline v3 end-to-end

- Após recuperar os silver files, rodar `scripts/rodar_v3.sh` do início
- Validar que os novos parquets correspondem aos valores dos JSON reports:
  - `sinan_mesorregiao_weekly.parquet`: deve ter 9 anos (2018-2026), ~60.554 linhas
  - `train_v3.parquet`: deve ter 4 anos [2019-2022], ~28.633 linhas
  - `test_v3.parquet`: ~16.988 linhas
- Validar que `27_regression_v3_report.json` → R²_log C=0.8698 é reproduzido

#### PRE-03: Confirmar métricas v3 reproduzidas

- Comparar métricas re-geradas com as do `32_final_results_v3.json`
- Tolerância: |ΔR²_log| < 0.001
- Se não reproduzir, investigar antes de prosseguir

**Estimativa de tempo**: 1-2 dias (download INMET + Bronze→Silver + re-run pipeline completa).

---

## 7. Requisitos Funcionais

### Configuração e dados

- **RF-01**: Criar configuração para as variantes experimentais. Como um único `split_v4` em `recorte.json` **não pode representar 3 variantes com treinos diferentes**, a solução é:
  - Adicionar em `config/recorte.json` uma chave `experiment_v4` com 3 sub-chaves (`variante_a`, `variante_b`, `variante_c`), cada uma com seus `warmup_ano`, `train`, `val` e `test`.
  - Criar `load_v4(variante)` em `recorte_config.py` que retorna o split da variante solicitada.
  - **Val e test são idênticos** nas 3 variantes: `val=[2023]`, `test=[2024, 2025, 2026]`.
  - Variante A: `warmup_ano=2018`, `train=[2019,2020,2021,2022]` — idêntica a v3.
  - Variante B: `warmup_ano=2000`, `train=[2001,...,2022]` (22 anos).
  - Variante C: idêntica a B (a diferença é o `sample_weight`, não o recorte).

- **RF-02**: Criar `scripts/40_aggregate_sinan_full.py` (baseado em `scripts/21_*`) para agregar SINAN Gold de **todos os anos** (2000-2026, excluindo 1999). Diferenças em relação ao script 21:
  - Lê `anos_pipeline` do config v4 variante B (2000-2026).
  - Produz `data/model_ready_v4/sinan_mesorregiao_full.parquet`.
  - O script 21 original **não é alterado** — v3 permanece intacto.

- **RF-03**: Criar `scripts/41_aggregate_inmet_full.py` (baseado em `scripts/22_*`) para processar todos os anos disponíveis no INMET Silver (2000-2008, 2010-2021, 2023-2026). Anos sem silver em disco (2009, 2022; e 2020 se não recuperado) ficam ausentes — tratados como NaN na integração.
  - Produz `data/model_ready_v4/inmet_mesorregiao_full.parquet`.
  - O script 22 original **não é alterado**.

- **RF-04**: Criar `scripts/42_integrate_full.py` (baseado em `scripts/23_*`). Na integração (LEFT JOIN do SINAN mesorregião + INMET mesorregião), adicionar coluna `flag_fonte_inmet` (int):
  - 1 = clima presente (pelo menos 1 estação contribuiu naquela mesorregião-semana)
  - 0 = sem dado INMET (NaN em todas as 12 variáveis climáticas)
  - **Esta coluna é estritamente metadata para análise posterior (RF-10). Ela NÃO deve entrar como feature no modelo** — deve ser excluída da lista de features antes do treino.

### Variantes experimentais

- **RF-05**: Implementar 3 variantes com **mesmo teste** (2024-2026) e **mesma validação** (2023):

  | Variante | Warm-up | Train | # anos treino | Linhas estimadas (treino) | Descrição |
  |----------|---------|-------|---------------|--------------------------|-----------|
  | **A — Baseline** | 2018 | 2019-2022 | 4 | ~28.633 | Reprodução exata da v3 |
  | **B — Série completa** | 2000 | 2001-2022 | 22 | ~152.000 | Todos os anos disponíveis |
  | **C — Série + peso** | 2000 | 2001-2022 | 22 | ~152.000 | Decaimento exponencial |

  **Nota**: O warm-up de 2000 serve exclusivamente para que os lags (até 8 semanas) do início de 2001 tenham valores não-nulos. O ano 2000 **não entra no treino**. O treino começa em 2001 para evitar ruído de um ano com apenas 2.087 municípios ativos e 4% de cobertura INMET.

- **RF-06**: Para variante C, implementar `sample_weight` com decaimento exponencial:

  ```
  weight(ano) = exp(-λ × (ano_ref - ano))
  ```

  Onde `ano_ref` = último ano de treino (2022) e `λ = 0.1` (meia-vida ≈ 6.93 anos).

  Pesos resultantes:
  | Ano | Peso | Ano | Peso |
  |-----|------|-----|------|
  | 2001 | 0.12 | 2012 | 0.37 |
  | 2005 | 0.18 | 2015 | 0.50 |
  | 2008 | 0.25 | 2019 | 0.74 |
  | 2010 | 0.30 | 2022 | 1.00 |

  Passar como `sample_weight` ao `model.fit()` do XGBoost. **Cada linha** de treino recebe o peso do seu ano.

- **RF-07**: Todas as variantes usam os mesmos `XGB_PARAMS` da baseline (ver seção 5.4). Manter `tree_method="hist"`, `random_state=42`, `early_stopping_rounds=50` na validação temporal.

- **RF-08**: Para cada variante, treinar 3 sub-modelos (SINAN-only, INMET bruto, INMET enriquecido). Total: **9 modelos** (3 variantes × 3 configs de features). O modelo principal para comparação entre variantes é o **INMET enriquecido** (266 features).

### Tuning (opcional, com orçamento)

- **RF-09**: Se e somente se o prazo permitir (antes de 15/11): rodar Optuna com **N=30 trials** sobre variante B (série completa, INMET enriquecido) usando validação temporal. Espaço de busca: `max_depth ∈ [4,10]`, `learning_rate ∈ [0.01, 0.3]`, `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`, `min_child_weight ∈ [1, 30]`, `reg_alpha ∈ [1e-3, 10]`, `reg_lambda ∈ [1e-3, 10]`. Se o melhor trial superar a baseline em ≥0.005 R²_log, usar os hiperparâmetros otimizados; caso contrário, manter os originais.

### Métricas e análises

- **RF-10**: Calcular para cada variante/sub-modelo:
  - Métricas globais: R²_log, R², RMSE, RMSE_log, MAE
  - Métricas **por ano de teste** (2024, 2025, 2026 separadamente)
  - Métricas **por mesorregião** (137 valores)
  - Métricas separadas para linhas com `flag_fonte_inmet == 1` vs `== 0` (nas variantes B e C, para quantificar o impacto da cobertura INMET esparsa nos anos antigos)

- **RF-11**: Intervalo de confiança 95% via **bootstrap** (n=1000 reamostras) para R²_log de cada variante. Comparação A vs B e A vs C: diferença + IC da diferença.

- **RF-12**: Teste t pareado (como já feito em v3, `scripts/27_train_regression_v3.py`) para as comparações A vs B, A vs C, B vs C.

- **RF-13**: Importância de features (gain do XGBoost + SHAP values) para as variantes A e B, comparadas lado a lado. Amostrar **10.000 linhas do teste** para SHAP (TreeExplainer é exato mas com ~150k linhas de treino a inferência pode ser lenta).

### Walk-forward

- **RF-14**: Walk-forward com expanding window adaptado à série completa. 4 folds anuais expansivos (mesmo design que `scripts/31_validate_walkforward_v3.py`):
  - Variante A (baseline): folds idênticos aos de v3 (train começa em 2019)
  - Variantes B e C: folds começando em 2001, expandindo até cada ano de teste

  Total de modelos walk-forward: 3 variantes × 3 sub-modelos × 4 folds = **36 modelos**.

### Curva de aprendizado (opcional)

- **RF-15**: Se o prazo permitir, gerar curva de aprendizado: treinar com janelas decrescentes — [2001-2022 (todos), 2007-2022 (16a), 2012-2022 (11a), 2017-2022 (6a), 2019-2022 (baseline)]. Plotar R²_log vs tamanho da janela de treino. Usa INMET enriquecido apenas.

### Saídas para o TCC

- **RF-16**: Gerar automaticamente:
  - Tabela comparativa A × B × C (métricas + IC 95%) em LaTeX
  - Gráfico de R²_log por ano de teste (barras agrupadas A/B/C)
  - Gráfico de R²_log por mesorregião (top/bottom 20 mesorregiões, delta B vs A)
  - Gráfico de cobertura INMET por ano (barras empilhadas: com dado vs NaN, por mesorregião)
  - Curva de aprendizado (se feita) — R²_log vs janela temporal
  - Comparativo de importância de features (SHAP top-30 A vs B, lado a lado)
  - Resumo em Markdown com decisões, limitações e ameaças à validade

---

## 8. Requisitos Não Funcionais

- **RNF-01**: O pipeline completo (pré-requisitos + variantes A+B+C) deve ser viável no hardware disponível: **16 GB RAM, 8 cores Apple M-series, sem GPU dedicada**.

  **Estimativa de tempo realista:**
  | Etapa | Modelos | Estimativa |
  |-------|---------|-----------|
  | Agregação + integração + feature eng | — | ~30 min |
  | Treino regressão (9 modelos) | 3 var × 3 sub | ~45 min (5 min/modelo) |
  | Walk-forward (36 modelos) | 3 var × 3 sub × 4 folds | ~3h (5 min/modelo) |
  | SHAP (2 variantes × 10k amostras) | 2 | ~30 min |
  | Bootstrap (1000 reamostras × 3 var) | 3 | ~20 min |
  | Data drift + figuras + compilação | — | ~15 min |
  | **Total estimado** | | **~5-6 horas** |

  Se exceder 8 horas: reduzir `n_estimators` para 500, limitar SHAP a 5.000 amostras, ou pular curva de aprendizado.

- **RNF-02**: Pico de memória < 12 GB (para não causar swap excessivo com 16 GB total).

  **Estimativa**: Dataset completo (2000-2026) ≈ 171.000 linhas × 276 colunas × 8 bytes ≈ **378 MB** cru. Com overhead Pandas (~2×), splits carregados simultaneamente e XGBoost em memória: pico estimado ~3-4 GB. Confortável.

- **RNF-03**: Dados intermediários salvos em Parquet com cache por etapa. Se um Parquet já existe e o config não mudou (hash do config), pular a etapa.

- **RNF-04**: Seed fixa (42) para reprodutibilidade completa.

- **RNF-05**: Cada etapa registra em JSON: versão do config, hash dos dados de entrada, métricas, tempo de execução.

- **RNF-06**: Um único comando para rodar o experimento: `scripts/rodar_v4.sh [primeiro_script]`, seguindo o padrão de `rodar_v3.sh` (`set -euo pipefail`, sequência numerada).

---

## 9. Desenho do Experimento

### 9.1 Variantes

```
Variante A (Baseline — reprodução exata de v3):
  Warm-up: 2018
  Train:   2019-2022 (4 anos, ~28.633 linhas mesorregionais)
  Val:     2023 (~7.261 linhas)
  Test:    2024-2026 (~16.988 linhas)
  Peso:    uniforme (todos = 1.0)

Variante B (Série completa):
  Warm-up: 2000
  Train:   2001-2022 (22 anos, ~152.000 linhas estimadas)
  Val:     2023 (IDÊNTICO a A)
  Test:    2024-2026 (IDÊNTICO a A)
  Peso:    uniforme (todos = 1.0)

Variante C (Série completa + peso por recência):
  Warm-up: 2000
  Train:   2001-2022 (idêntico a B)
  Val:     2023 (IDÊNTICO a A)
  Test:    2024-2026 (IDÊNTICO a A)
  Peso:    sample_weight = exp(-0.1 × (2022 - ano))
    2001: 0.12    2005: 0.18    2010: 0.30
    2015: 0.50    2019: 0.74    2022: 1.00
```

### 9.2 Split temporal

Validação **estritamente temporal** — nunca k-fold aleatório. O conjunto de val/test é idêntico entre variantes para comparação justa. Os percentis de classificação (`risco_surto_t4`) são calculados **apenas** no treino de cada variante.

A climatologia para anomalias (script 44) é recalculada para cada variante usando seus próprios anos de treino.

### 9.3 Walk-forward

4 folds expansivos (mesmo design de `scripts/31_validate_walkforward_v3.py`):

```
Fold 1: Train=[warm-up...2022] → Test=[2023]
Fold 2: Train=[warm-up...2023] → Test=[2024]
Fold 3: Train=[warm-up...2024] → Test=[2025]
Fold 4: Train=[warm-up...2025] → Test=[2026]
```

Para variante A, o warm-up é 2018 e o train começa em 2019; para B/C, warm-up é 2000 e train começa em 2001.

Validação interna (early stopping): últimas N_VAL_WEEKS=10 semanas epidemiológicas do conjunto de treino de cada fold.

### 9.4 Métricas e teste estatístico

| Métrica | Tipo | Uso |
|---------|------|-----|
| R²_log | Principal | Comparação entre variantes (métrica do TCC) |
| R², RMSE, MAE | Complementar | Interpretação na escala original |
| IC 95% (bootstrap, n=1000) | Incerteza | Barras de erro nas tabelas e figuras |
| t-test pareado (por mesorregião-semana) | Significância | p-value para A vs B, A vs C, B vs C |
| F1_macro, AUC_macro_ovr | Classificação | Se tempo permitir (RF-09) |

---

## 10. Tratamento dos Riscos de Dados

### R-01: Dados SINAN 1999 praticamente vazios

**Fato verificado**: 1999 tem 5.565 rows no Gold Dense mas apenas 8 municípios ativos e 13 notificações.

**Decisão**: Excluir 1999 completamente. O ano mais antigo no pipeline é **2000** (warm-up). O treino começa em **2001**. O warm-up de 2000 fornece valores não-nulos para os lags (até 8 semanas) do início de 2001.

### R-02: Mudança de divisão territorial IBGE (2017)

**Fato verificado**: O mapeamento em `scripts/20_build_mesorregiao_mapping.py` usa `data/reference/ibge_municipios_api.json` (API IBGE), que retorna microrregião/mesorregião legada para 5.570 dos 5.571 municípios (1 sem mesorregião: Boa Esperança do Norte, MT). O mapping é **estático** e aplica o **mesmo** crosswalk município→mesorregião a todos os anos.

**Decisão**: Manter o crosswalk atual. As mesorregiões foram substituídas pelas Regiões Geográficas Intermediárias em 2017, mas os **códigos de município** (ibge_municipio, 7 dígitos) não mudaram — o mesmo município pertence à mesma mesorregião independente do ano. Validar com assert (ver T-02).

### R-03: Municípios criados/desmembrados ao longo do tempo

**Fato verificado**: O SINAN Gold tem de 2.087 (ano 2000) a 5.485 (ano 2024) municípios ativos.

**Decisão**: O merge `inner` entre SINAN Gold e `municipio_mesorregiao.csv` descarta automaticamente qualquer município com código não mapeável. Registrar no report quantos municípios foram descartados por ano. Não inventar dados para municípios que não existiam — a agregação por mesorregião absorve isso naturalmente (menos municípios contribuindo naquele ano = soma menor, mas reflete a realidade).

### R-04: Cobertura do INMET muda drasticamente com o tempo

**Fato verificado**: INMET Silver vai de 5 estações/5 mesorregiões em 2000 até 633 estações/135 mesorregiões em 2026. Três anos não têm silver em disco: 2009 (recuperação opcional), 2020 e 2022 (recuperação obrigatória — PRE-01).

**Decisão**:
1. **Anos sem silver** (2009 se não recuperado): as 12 features climáticas ficam NaN para todas as mesorregiões naquele ano. O `flag_fonte_inmet = 0`. O XGBoost trata NaN nativamente (envia para o ramo que reduz o loss). **Não interpolar** entre anos.
2. **2009 e 2020** (1 semana de dados): Mesmo com silver recuperado, a cobertura temporal é insuficiente (1 semana em 52). As 51 semanas restantes terão NaN. Tratar igual aos anos sem silver.
3. **Anos com poucas estações** (2000-2005, 4-36% das mesorregiões): muitas mesorregiões sem estação → NaN. A análise de métricas com/sem INMET (RF-10) quantificará o impacto.
4. **Fallback intra-ano**: Se uma mesorregião tem estação mas com falhas em semanas específicas, os NaN ficam. Não preencher com estação vizinha (mantém simplicidade e reprodutibilidade).
5. O gráfico de cobertura INMET por ano (RF-16) mostrará visualmente a evolução.

### R-05: Mudanças de esquema/unidade nas fontes

**Fato verificado**: O SINAN Gold Dense tem **142 colunas** consistentes em todos os anos (1999-2026). O INMET Silver tem esquema consistente (`coverage_report.csv` mostra as mesmas colunas para todos os anos). A padronização foi feita na camada Bronze→Silver.

**Decisão**: Sem ação necessária. Se a pipeline detectar colunas faltantes em algum ano, logar WARNING e preencher com NaN.

### R-06: Data drift / concept drift

**Proposta**:
1. Calcular PSI (Population Stability Index) ou teste KS para as top-20 features (por SHAP) entre cada ano de treino e o conjunto de teste. Registrar no report.
2. A variante C (peso por recência) é a mitigação direta: anos antigos com distribuições diferentes contribuem menos.
3. Comparar importância de features entre A e B — se a série completa eleva features instáveis, isso indica drift.

### R-07: Vazamento temporal

**Fato verificado no código**: O pipeline v3 já trata isso:
- Target calculado com `shift(-4)` por grupo (script 26)
- Percentis de classificação calculados **somente no treino** (script 26, `compute_risk_class()`)
- Anomalias calculadas com climatologia **somente do treino** (script 25)
- Walk-forward usa janela expansiva, nunca olha o futuro

**Decisão**: Manter as mesmas garantias nos scripts v4. Adicionalmente:
- Normalizações (se adicionadas): fit somente no treino de cada fold
- O treino de 2001-2022 não pode usar nenhum dado de 2023-2026
- O warm-up (2000) é usado **apenas** para calcular lags; suas linhas não entram no treino

---

## 11. Arquitetura e Mudanças no Código

### 11.1 Princípio: v3 intacto

Os scripts v3 (20-33) **não são alterados**. Os scripts v4 (40-53) são **novos scripts baseados nos v3**, para preservar a reprodutibilidade da baseline. Ambos os conjuntos podem coexistir.

### 11.2 Arquivos a criar

| Arquivo | Baseado em | Descrição |
|---------|-----------|-----------|
| `config/recorte.json` (nova chave `experiment_v4`) | — | Config para variantes A, B, C com mesmo val/test |
| `scripts/recorte_config.py` (novo método `load_v4()`) | — | Carrega config da variante solicitada |
| `scripts/40_aggregate_sinan_full.py` | `21_*` | Agregação SINAN para todos os anos (2000-2026) |
| `scripts/41_aggregate_inmet_full.py` | `22_*` | Agregação INMET para todos os anos disponíveis |
| `scripts/42_integrate_full.py` | `23_*` | Integração com `flag_fonte_inmet` (metadata) |
| `scripts/43_feature_eng_lags_v4.py` | `24_*` | Lags sobre o dataset completo |
| `scripts/44_feature_eng_anomalias_v4.py` | `25_*` | Anomalias com climatologia por variante |
| `scripts/45_prepare_splits_v4.py` | `26_*` | Splits para variantes A, B, C simultaneamente |
| `scripts/46_train_regression_v4.py` | `27_*` | Treino 3 variantes × 3 sub-modelos = 9 modelos |
| `scripts/47_train_classification_v4.py` | `29_*` | Classificação (se tempo permitir) |
| `scripts/48_explain_shap_v4.py` | `30_*` | SHAP para variantes A e B (10k amostras) |
| `scripts/49_validate_walkforward_v4.py` | `31_*` | Walk-forward 3 variantes × 3 sub × 4 folds |
| `scripts/50_data_drift_analysis.py` | novo | PSI/KS por feature × ano |
| `scripts/51_learning_curve.py` | novo | Curva de aprendizado (opcional) |
| `scripts/52_compile_final_v4.py` | `32_*` | Compilação de todos os resultados |
| `scripts/53_generate_tcc_outputs.py` | novo | Tabelas LaTeX + figuras para o Overleaf |
| `scripts/rodar_v4.sh` | `rodar_v3.sh` | Execução completa (`set -euo pipefail`) |

### 11.3 Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `scripts/recorte_config.py` | Adicionar `load_v4(variante: str)` |
| `config/recorte.json` | Adicionar chave `experiment_v4` |

### 11.4 Diretórios de saída

```
data/model_ready_v4/            # Dados intermediários (parquets, reports JSON)
data/model_ready_v4/splits/     # train/val/test para cada variante
models/regression_v4/           # Modelos treinados (.ubj)
models/regression_v4/variante_a/
models/regression_v4/variante_b/
models/regression_v4/variante_c/
```

---

## 12. Plano de Testes

### Testes automatizados (assertions nos scripts)

| ID | Verificação | Script |
|----|------------|--------|
| T-01 | Schema: todos os anos do SINAN Gold têm as mesmas 142 colunas | 40 |
| T-02 | Crosswalk: 100% dos municípios do SINAN Gold são mapeáveis para mesorregião (assert no merge; logar com warning os descartados) | 40 |
| T-03 | `flag_fonte_inmet` preenchido em 100% das linhas do dataset integrado, com valor 0 ou 1 | 42 |
| T-04 | `flag_fonte_inmet` **não está** na lista de features de treino (é metadata) | 45, 46 |
| T-05 | Ausência de vazamento temporal: `set(train.anos) ∩ set(test.anos) == ∅` | 45 |
| T-06 | Target `notificacoes_t4` = `shift(-4)` correto por mesorregião (spot-check em 10 linhas aleatórias) | 45 |
| T-07 | Variante A reproduz métricas da baseline v3 com |ΔR²_log| < 0.001 | 46 |
| T-08 | Val e test são **idênticos** (mesmas linhas, mesmos valores) entre variantes A, B, C | 45 |
| T-09 | `sample_weight` da variante C é estritamente positivo e monotonicamente crescente com o ano | 46 |
| T-10 | Pico de memória < 12 GB (verificar via `tracemalloc` ou `psutil`) | rodar_v4.sh |

### Validações manuais

| ID | Verificação |
|----|------------|
| T-11 | Comparar gráfico de cobertura INMET com tabela 6.3 — consistência visual |
| T-12 | Verificar que SHAP top-10 features da variante A são consistentes com SHAP da baseline v3 original |
| T-13 | Inspecionar métricas por ano de teste: 2026 (parcial) deve ter IC mais largo que 2024/2025 |

---

## 13. Critérios de Aceite

| ID | Critério | Método de verificação |
|----|----------|----------------------|
| CA-01 | Variante A reproduz R²_log da baseline v3 com |diferença| < 0.001 (baseline: 0.8698 para modelo C) | T-07 |
| CA-02 | Variante B (série completa) tem métricas computadas e IC 95% reportados | Inspecionar JSON + tabela LaTeX |
| CA-03 | Variante C (com peso) tem métricas computadas e IC 95% reportados | idem |
| CA-04 | Teste t pareado entre A vs B e A vs C com p-value reportado | Inspecionar JSON |
| CA-05 | Tabela LaTeX comparativa A × B × C salva no diretório do Overleaf | Verificar arquivo .tex |
| CA-06 | Gráficos (métricas por ano, por mesorregião, cobertura INMET, SHAP) gerados como PNG 300dpi | Verificar PNGs |
| CA-07 | Resumo em Markdown com conclusão sobre H1-H4 salvo em `data/model_ready_v4/` | Verificar .md |
| CA-08 | Pipeline completa roda com um único comando e termina sem erro em < **8 horas** | Medir tempo real |
| CA-09 | Nenhum dado de 2023-2026 entra no treino de qualquer variante | T-05 |
| CA-10 | INMET Silver para 2020 e 2022 recuperados e pipeline v3 consistente (PRE-01 a PRE-03) | Validar parquets vs reports |

---

## 14. Artefatos Gerados para o TCC

| Artefato | Formato | Destino |
|----------|---------|---------|
| Tabela comparativa A × B × C (métricas + IC) | LaTeX (.tex) | `Overleaf/TCC2 Base FCTE UnB/tabelas/` |
| Gráfico R²_log por ano de teste (barras agrupadas) | PNG 300dpi | `Overleaf/TCC2 Base FCTE UnB/figuras/resultados/` |
| Gráfico R²_log por mesorregião (delta B vs A) | PNG 300dpi | idem |
| Gráfico cobertura INMET por ano (barras empilhadas) | PNG 300dpi | idem |
| Curva de aprendizado (se feita) | PNG 300dpi | idem |
| SHAP top-30 (A vs B, lado a lado) | PNG 300dpi | idem |
| Data drift PSI/KS por ano | PNG 300dpi | idem |
| Walk-forward R²_log por fold (3 variantes) | PNG 300dpi | idem |
| Resumo Markdown com decisões e limitações | .md | `data/model_ready_v4/` |
| Reports JSON de cada etapa | .json | `data/model_ready_v4/` |

---

## 15. Cronograma

Prazo final TCC: **~30/11/2026**. Treino e tabelas prontos até **~20/11/2026** para incorporar no texto.

| Fase | Período | Entregas | Dias |
|------|---------|----------|------|
| **F0: Pré-requisitos** | 03-04/11 | Recuperar INMET 2020/2022, re-run v3, validar CA-10 | 2 |
| **F1: Config + Agregação** | 05-07/11 | `experiment_v4` em recorte.json, scripts 40-42, dataset integrado full com `flag_fonte_inmet` | 3 |
| **F2: Feature Eng + Splits** | 08-09/11 | Scripts 43-45, splits A/B/C, validações T-01 a T-10 | 2 |
| **F3: Treino + Métricas** | 10-12/11 | Script 46, 9 modelos treinados, métricas + IC bootstrap + t-test, CA-01 validado | 3 |
| **F4: Walk-forward + SHAP** | 13-15/11 | Scripts 48-49, 36 modelos walk-forward, SHAP A vs B | 3 |
| **F5: Drift + Curva + Saídas** | 16-17/11 | Scripts 50-53, PSI/KS, curva de aprendizado (se viável), todas as figuras e tabelas LaTeX | 2 |
| **F6: Tuning Optuna** (opcional) | 15-16/11 | RF-09, só se F3 mostrar que B supera A e vale otimizar | 2 |
| **F7: Validação final** | 18-19/11 | Rodar pipeline v4 completa do zero, verificar CA-01 a CA-10, corrigir bugs | 2 |
| **Buffer** | 20-22/11 | Artefatos finais para Overleaf, revisão do orientador | 3 |
| **Escrita TCC** | 20-30/11 | Incorporar resultados no texto final | — |

**Total estimado**: ~19 dias (03-21/11), incluindo 2 dias de pré-requisitos + 3 dias de buffer.

---

## 16. Riscos e Plano B

| # | Risco | Probabilidade | Impacto | Mitigação / Plano B |
|---|-------|---------------|---------|----------------------|
| 1 | **Pipeline v3 em estado inconsistente** — parquets degradados, INMET silver ausente | Confirmado | **Crítico** | PRE-01 a PRE-03 tratam isso. Se não conseguir recuperar ZIPs do INMET: usar mesorregião file existente e aceitar baseline sem 2020/2022. |
| 2 | **Série completa não melhora** | Média | Alto (narrativa TCC) | Resultado negativo é igualmente publicável: "dados antigos não ajudam devido à baixa cobertura INMET e data drift". A variante C (peso) fornece o argumento de mitigação. |
| 3 | **Não cabe em memória** | Baixa | Médio | Dataset estimado ~378 MB cru. Se estourar, processar por chunks (ano a ano) e concatenar ao final. |
| 4 | **Tempo de execução > 8h** | Média | Médio | 36 modelos walk-forward são o gargalo. Reduzir `n_estimators` para 500, ou limitar SHAP a 5.000 amostras, ou pular curva de aprendizado. |
| 5 | **INMET antigo degrada modelo** | Média | Médio | Variante C (peso por recência) é a mitigação direta. Se C supera B, o argumento é claro. |
| 6 | **Optuna não converge a tempo** | Baixa | Baixo | Manter hiperparâmetros da baseline. Optuna é opcional (RF-09). |
| 7 | **Bug nos scripts v4** | Média | Alto | CA-01 (reprodução da baseline) é o primeiro teste. Se A não bate com v3, debugar antes de prosseguir. Os scripts v3 permanecem intactos como referência. |
| 8 | **Orientador pede mudanças** | Média | Médio | Buffer de 3 dias (20-22/11). Se necessário, entregar só A+B sem C ou sem curva de aprendizado. |
| 9 | **SINAN 2026 muito parcial** | Confirmado | Baixo | Apenas SE01-SE16 (90.026 linhas Gold). O teste inclui 2024+2025 completos. Reportar 2026 como parcial em todas as análises. |
| 10 | **INMET 2009 irrecuperável** | Alta | Baixo | Impacto mínimo: mesmo quando existia, tinha apenas 1 semana de dados. Tratar como NaN e registrar. |

---

## 17. Perguntas em Aberto

1. **Excluir 1999 e usar 2000 como warm-up?**
   1999 tem 8 municípios (13 notificações) — praticamente vazio. 2000 tem 2.087 (38% dos ativos) e 4% de cobertura INMET.
   **Proposta**: excluir 1999 completamente, usar 2000 como warm-up (lags), treinar a partir de 2001.
   Preciso da sua aprovação.

2. **Lambda do decaimento exponencial**:
   Proposto λ=0.1 (meia-vida ≈ 6.93 anos). Testar outros valores (0.05, 0.15, 0.2) ou só o proposto?

3. **Recuperar INMET Silver para 2009?**
   2020 e 2022 são obrigatórios (ver PRE-01). Mas 2009 é opcional — mesmo quando existia, tinha apenas 1 semana de dados (129 mesorregiões × 1 semana). Recuperar adiciona ~0.5 dia de trabalho para impacto mínimo. Vale o esforço?

4. **Classificação na série completa**:
   O pipeline v3 inclui classificação (4 classes). Repetir para as variantes B/C ou focar apenas em regressão (mais rápido e o argumento central do TCC é regressão)?

5. **Curva de aprendizado — janelas exatas**:
   Proposta: [2001-2022 (todos), 2007-2022 (16a), 2012-2022 (11a), 2017-2022 (6a), 2019-2022 (baseline)].
   Testar apenas com INMET enriquecido para economizar tempo. OK?

6. **SHAP: amostrar 10.000 linhas do teste?**
   SHAP TreeExplainer é exato para XGBoost, mas com ~150k linhas pode ser lento. Proposta: amostrar 10.000 linhas do teste. OK?

7. **Onde salvar tabelas LaTeX?**
   Direto em `Overleaf/TCC2 Base FCTE UnB/tabelas/` (como v3 faz com figuras) ou em `docs/`?

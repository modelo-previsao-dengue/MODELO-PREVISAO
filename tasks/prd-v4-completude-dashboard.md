# PRD: Completude v4 — INMET, Optuna, Classificação, Dashboard e Landing Page

## 1. Introdução

O pipeline v4 de treinamento XGBoost mesorregional está funcional: 13 scripts, 9 modelos de regressão (3 variantes × 3 sub-modelos), 27 anos de SINAN (2000-2026), 137 mesorregiões, walk-forward, SHAP, data drift, learning curve, e outputs para Overleaf (9 figuras + 5 tabelas LaTeX).

No entanto, restam **5 lacunas**:

1. **INMET Silver 2022 ausente** — o arquivo não está em disco, causando delta de 0.0025 no CA-01 e cobertura INMET de 48% na Var A (deveria ser ~90%).
2. **Optuna tuning** (RF-09 do PRD original) — não implementado; pode melhorar Var B.
3. **Classificação** (script 47 do PRD original) — não implementada; o dashboard Streamlit precisa de modelos de classificação para alertas de risco.
4. **Dashboard Streamlit** — `dashboard/app.py` aponta para modelos antigos (`models/nacional/`, dados `data/model_ready/`). Precisa ser atualizado para v4 mesorregional.
5. **Landing Page HTML** — `docs/dashboard.html` é baseada em v2. Precisa rebuild com resultados v4.

Este PRD especifica a implementação dessas 5 lacunas para completar o TCC.

---

## 2. Objetivos

- **O1**: Recuperar INMET Silver 2022 e re-rodar a pipeline v4 para obter CA-01 ≤ 0.001.
- **O2**: Otimizar hiperparâmetros da Var B com Optuna (30 trials) e avaliar se melhora ≥ 0.005 R²_log.
- **O3**: Treinar modelos de classificação de risco de surto (4 classes) para as 3 variantes v4.
- **O4**: Atualizar o dashboard Streamlit para usar os modelos v4 mesorregionais (regressão + classificação).
- **O5**: Rebuild da landing page HTML estática com todos os resultados v4.

---

## 3. Estado Atual (Pré-requisitos)

### 3.1 Pipeline v4 (completa)

| Item | Estado | Local |
|------|--------|-------|
| 13 scripts Python (40-53) | OK | `scripts/4[0-9]_*.py`, `scripts/5[0-3]_*.py` |
| `rodar_v4.sh` | OK | `scripts/rodar_v4.sh` (22.8 min, exit 0) |
| 9 modelos regressão `.ubj` | OK | `models/regression_v4/` |
| 12 JSON reports | OK | `data/model_ready_v4/` |
| 9 figuras PNG (Overleaf) | OK | `../Overleaf/TCC2 Base FCTE UnB/figuras/resultados_v4/` |
| 5 tabelas LaTeX (Overleaf) | OK | `../Overleaf/TCC2 Base FCTE UnB/tabelas/tab_v4_*.tex` |
| Resumo MD | OK | `data/model_ready_v4/53_tcc_outputs_summary.md` |

### 3.2 Dados

| Dataset | Linhas | Anos | Mesorregiões |
|---------|--------|------|--------------|
| SINAN Gold Dense | 7.665.428 | 28 (1999-2026) |  5.565 municípios |
| SINAN Mesorregião (script 40) | 188.512 | 27 (2000-2026) | 137 |
| INMET Mesorregião (script 41) | — | 26 (falta 2022) | variável |
| Integrado v4 (script 42) | 188.512 | 27 | 137 (65.3% cobertura INMET) |
| Features v4 (scripts 43-44) | 188.512 | 27 | 277 colunas |

### 3.3 Resultados principais

| Variante | R²_log | IC 95% | p-value vs A |
|----------|--------|--------|-------------|
| A (baseline 4yr) | 0.8723 | [0.8661, 0.8780] | — |
| **B (série 22yr)** | **0.8878** | [0.8820, 0.8929] | 8.29e-40 |
| C (série + peso) | 0.8870 | [0.8812, 0.8924] | 7.24e-56 |

### 3.4 Lacunas

| Item | Impacto |
|------|---------|
| INMET Silver 2022 ausente | CA-01 delta=0.0025 (>0.001), Var A cobertura 48% |
| Optuna não rodou | Var B pode melhorar com tuning |
| Classificação não implementada | Dashboard sem alertas de risco |
| Dashboard Streamlit desatualizado | Aponta para modelos antigos (nacional/v2) |
| Landing page HTML desatualizada | Baseada em v2, sem resultados v4 |

---

## 4. User Stories

### US-001: Recuperar INMET Silver 2022

**Descrição:** Como pesquisador, quero que o INMET Silver de 2022 esteja disponível para que a Var A tenha cobertura INMET correta (~90%) e o CA-01 seja atendido (|delta| < 0.001).

**Acceptance Criteria:**
- [ ] Arquivo INMET Silver 2022 existe em `data/inmet/silver/` (ou caminho equivalente)
- [ ] Script 41 processa 2022 sem erro
- [ ] Cobertura INMET da Var A no treino ≥ 80%
- [ ] Re-run completa do pipeline v4 (rodar_v4.sh) termina com exit 0
- [ ] CA-01: |Var A R²_log - 0.8698| < 0.001

**Nota:** Este passo requer **download manual** do portal INMET (https://portal.inmet.gov.br/dadoshistoricos). O script de ingestão Bronze→Silver já existe. O PRD cobre apenas a integração após o download.

---

### US-002: Optuna Tuning (RF-09)

**Descrição:** Como pesquisador, quero otimizar os hiperparâmetros do XGBoost na Var B (série completa, INMET enriquecido) para verificar se o tuning melhora o R²_log em ≥ 0.005.

**Acceptance Criteria:**
- [ ] Script `scripts/49_optuna_tuning_v4.py` criado
- [ ] 30 trials executados com validação temporal (val=2023)
- [ ] Espaço de busca conforme PRD original: `max_depth ∈ [4,10]`, `learning_rate ∈ [0.01, 0.3]`, `subsample ∈ [0.6, 1.0]`, `colsample_bytree ∈ [0.6, 1.0]`, `min_child_weight ∈ [1, 30]`, `reg_alpha ∈ [1e-3, 10]`, `reg_lambda ∈ [1e-3, 10]`
- [ ] Report JSON salvo em `data/model_ready_v4/49_optuna_report.json`
- [ ] Se melhor trial supera baseline em ≥ 0.005 R²_log: re-treinar Var B com params otimizados e salvar novo modelo
- [ ] Se não supera: manter params originais, registrar no report
- [ ] Tabela comparativa (baseline vs tuned) incluída nos outputs

---

### US-003: Classificação de Risco (3 variantes)

**Descrição:** Como pesquisador, quero modelos de classificação XGBoost (4 classes: baixo/moderado/alto/surto) para as 3 variantes v4, para que o dashboard mostre alertas de risco.

**Acceptance Criteria:**
- [ ] Script `scripts/54_train_classification_v4.py` criado (baseado em `scripts/29_train_classification_v3.py`)
- [ ] Target: `risco_surto_t4` (já existe nos splits, calculado com percentis do treino)
- [ ] 9 modelos de classificação (3 variantes × 3 sub-modelos) treinados
- [ ] Modelos salvos em `models/classification_v4/` como `.ubj`
- [ ] Métricas: F1_macro, AUC_macro_ovr, precision/recall por classe, confusion matrix
- [ ] Report JSON salvo em `data/model_ready_v4/54_classification_v4_report.json`
- [ ] Tabela LaTeX de classificação salva em `../Overleaf/TCC2 Base FCTE UnB/tabelas/tab_v4_classification.tex`

---

### US-004: Dashboard Streamlit v4

**Descrição:** Como pesquisador/apresentador do TCC, quero um dashboard Streamlit interativo com os modelos v4 mesorregionais para visualizar previsões, métricas e alertas de risco por mesorregião.

**Acceptance Criteria:**
- [ ] `dashboard/app.py` atualizado para carregar modelos de `models/regression_v4/` e `models/classification_v4/`
- [ ] Dados de teste carregados de `data/model_ready_v4/splits_b/test.parquet` (Var B = melhor modelo)
- [ ] Sidebar: filtro por UF, mesorregião, ano, variante (A/B/C)
- [ ] Painel de métricas: R²_log, RMSE, MAE para variante selecionada
- [ ] Alertas de surto: top mesorregiões com risco alto/surto previsto
- [ ] Série temporal: real vs previsto por mesorregião selecionada
- [ ] Mapa de calor: R²_log por mesorregião (usando `metrics_por_mesorregiao_v4.csv`)
- [ ] SHAP: top-20 features (carregado de `models/regression_v4/shap_importance_v4.csv`)
- [ ] Walk-forward: gráfico de R²_log por fold (carregado do report JSON)
- [ ] Comparação entre variantes: barras agrupadas A/B/C
- [ ] `streamlit run dashboard/app.py` funciona sem erro
- [ ] Verify in browser

---

### US-005: Landing Page HTML Estática v4

**Descrição:** Como pesquisador, quero uma landing page HTML estática com todos os resultados v4, sem servidor, para referência rápida e apresentação offline.

**Acceptance Criteria:**
- [ ] Script `scripts/55_build_dashboard_v4.py` criado (baseado em `scripts/build_dashboard.py`)
- [ ] Carrega dados de `data/model_ready_v4/52_final_results_v4.json` e CSVs
- [ ] Gera `docs/dashboard_v4.html` (arquivo único, auto-contido)
- [ ] Seções: resumo executivo, comparação de variantes, métricas por ano/UF/mesorregião, SHAP, walk-forward, learning curve, data drift, conclusões
- [ ] Gráficos interativos via Chart.js (embutido)
- [ ] Responsivo (mobile-friendly)
- [ ] Dark mode toggle
- [ ] Abre no browser sem servidor (`file://`)

---

### US-006: Integração no rodar_v4.sh

**Descrição:** Como pesquisador, quero que todos os novos scripts sejam integrados no pipeline de execução única.

**Acceptance Criteria:**
- [ ] `scripts/rodar_v4.sh` atualizado: glob `5[0-5]_*` para incluir scripts 49, 54 e 55
- [ ] Ordem: 40-48, **49** (optuna), 50-53, **54** (classificação), **55** (dashboard HTML)
- [ ] Pipeline completa roda com `bash scripts/rodar_v4.sh` sem erro
- [ ] Tempo total < 8 horas (CA-08)

---

## 5. Requisitos Funcionais

### INMET e Pipeline

- **RF-01**: Após download manual do INMET Silver 2022, o script 41 deve processá-lo automaticamente ao detectar o novo arquivo.
- **RF-02**: Re-run do pipeline v4 deve produzir CA-01 ≤ 0.001 com INMET 2022 presente.

### Optuna

- **RF-03**: Criar `scripts/49_optuna_tuning_v4.py` que:
  - Carrega splits Var B (treino e validação)
  - Usa `optuna.create_study(direction="maximize")` com `R²_log` no val como métrica
  - Treina XGBoost com INMET enriquecido (266 features)
  - 30 trials, timeout opcional de 2h
  - Salva `best_params`, `best_value`, trial history no JSON report
  - Se `best_value - baseline_R2_log ≥ 0.005`: salva modelo otimizado em `models/regression_v4/model_B_inmet_enriquecido_tuned.ubj`

- **RF-04**: Espaço de busca Optuna:
  ```python
  max_depth: int [4, 10]
  learning_rate: float [0.01, 0.3] (log)
  subsample: float [0.6, 1.0]
  colsample_bytree: float [0.6, 1.0]
  min_child_weight: int [1, 30]
  reg_alpha: float [1e-3, 10] (log)
  reg_lambda: float [1e-3, 10] (log)
  n_estimators: fixo 1000 (com early_stopping_rounds=50)
  ```

### Classificação

- **RF-05**: Criar `scripts/54_train_classification_v4.py` que:
  - Treina XGBClassifier para 4 classes de risco: `{0: "baixo", 1: "moderado", 2: "alto", 3: "surto"}`
  - Target: `risco_surto_t4` (já nos splits, calculado com percentis Q50/Q75/Q90 do treino, por mesorregião)
  - 9 modelos: 3 variantes × 3 sub-modelos (mesma estrutura do script 46)
  - Usa `multi:softprob` como objective, `eval_metric="mlogloss"`, `early_stopping_rounds=50`
  - Variante C usa `sample_weight` idêntico ao da regressão
  - Métricas: F1_macro, AUC_macro_ovr (one-vs-rest), precision/recall/f1 por classe, confusion matrix normalizada

- **RF-06**: Salvar modelos em `models/classification_v4/model_{A,B,C}_{sinan_only,inmet_bruto,inmet_enriquecido}_cls.ubj`

- **RF-07**: Salvar confusion matrix como figura PNG em `../Overleaf/TCC2 Base FCTE UnB/figuras/resultados_v4/fig_v4_confusion_matrix.png`

### Dashboard Streamlit

- **RF-08**: Atualizar `dashboard/app.py` para:
  - Granularidade mesorregional (não municipal como o antigo)
  - Carregar modelos de `models/regression_v4/` e `models/classification_v4/`
  - Dados de `data/model_ready_v4/splits_b/test.parquet` (Var B, melhor)
  - Mapping mesorregião→nome via `data/model_ready_v3/municipio_mesorregiao.csv`

- **RF-09**: Seções do dashboard:
  1. **Header**: título "DengueCast v4 — Previsão de Dengue por Mesorregião", seletor de variante
  2. **KPIs**: R²_log, RMSE, MAE, nº mesorregiões, período (cards)
  3. **Alertas**: tabela das top-20 mesorregiões com maior risco previsto (filtráveis por UF)
  4. **Série Temporal**: gráfico real vs previsto para mesorregião selecionada (Plotly)
  5. **Mapa de Calor por UF**: R²_log por UF (usando `metrics_por_uf_v4.csv`)
  6. **Importância de Features**: barplot horizontal dos top-20 SHAP values
  7. **Comparação de Variantes**: barras A/B/C com IC 95%
  8. **Walk-forward**: R²_log por fold, 3 variantes

- **RF-10**: Sidebar com filtros: UF, mesorregião, ano, variante (A/B/C), sub-modelo (SINAN/INMET bruto/enriquecido)

### Landing Page HTML

- **RF-11**: Criar `scripts/55_build_dashboard_v4.py` que gera `docs/dashboard_v4.html`

- **RF-12**: A landing page deve conter:
  1. **Hero**: título, resumo da pesquisa, KPIs principais
  2. **Comparação de Variantes**: tabela + gráfico (Chart.js)
  3. **Testes Estatísticos**: t-test, bootstrap IC
  4. **Métricas por Ano**: gráfico de barras agrupadas 2024/2025/2026
  5. **Métricas por UF**: tabela/heatmap
  6. **Walk-forward**: gráfico de linha por fold
  7. **SHAP**: top-20 features horizontal bar
  8. **Data Drift**: PSI por ano
  9. **Learning Curve**: R²_log vs tamanho do treino
  10. **Cobertura INMET**: barras empilhadas por ano
  11. **Conclusões**: resumo textual com H1-H4
  12. **Footer**: metadados (data de geração, versão, nº modelos)

- **RF-13**: Dados embutidos como JSON no HTML (lidos de `52_final_results_v4.json` e CSVs)
- **RF-14**: Chart.js para gráficos interativos (hover com valores)
- **RF-15**: Tailwind CSS para styling, dark mode toggle, responsivo

---

## 6. Não-Objetivos (Fora de Escopo)

- **NG-01**: Não trocar o algoritmo (permanece XGBoost).
- **NG-02**: Não alterar granularidade (mesorregião × semana) nem target (notificacoes_t4, risco_surto_t4).
- **NG-03**: Não refazer pipeline SINAN Bronze→Silver→Gold.
- **NG-04**: Não criar API REST ou deploy em nuvem. Dashboard é local (Streamlit) e landing page é estática.
- **NG-05**: Não alterar scripts v3 (20-33). Coexistência mantida.
- **NG-06**: Não implementar modelos de deep learning ou ensemble com outros algoritmos.

---

## 7. Considerações Técnicas

### Dependências existentes
- Python 3.9+, XGBoost, pandas, numpy, scikit-learn, matplotlib, SHAP
- **Novas dependências**: `optuna` (tuning), `streamlit`, `plotly` (dashboard)
- Dashboard HTML: Chart.js (CDN), Tailwind CSS (CDN) — sem dependências Python

### Hardware
- 16 GB RAM, 8 cores Apple M-series, sem GPU
- Optuna 30 trials × ~5 min/trial = ~2.5h (se paralelo com `n_jobs=-1`)
- Classificação 9 modelos: ~15-20 min
- Dashboard build: < 1 min

### Reprodutibilidade
- `random_state=42` em todos os modelos
- Optuna sampler: `TPESampler(seed=42)`
- Resultados salvos em JSON para auditoria

---

## 8. Arquivos a Criar/Alterar

### Novos scripts

| Arquivo | Baseado em | Descrição |
|---------|-----------|-----------|
| `scripts/49_optuna_tuning_v4.py` | novo | Optuna 30 trials sobre Var B INMET enriquecido |
| `scripts/54_train_classification_v4.py` | `scripts/29_train_classification_v3.py` | Classificação 3 var × 3 sub = 9 modelos |
| `scripts/55_build_dashboard_v4.py` | `scripts/build_dashboard.py` | Gera landing page HTML estática |

### Arquivos alterados

| Arquivo | Alteração |
|---------|-----------|
| `dashboard/app.py` | Reescrito para v4 mesorregional |
| `scripts/rodar_v4.sh` | Adicionar scripts 49, 54, 55 |
| `scripts/52_compile_final_v4.py` | Incluir report do Optuna e classificação na compilação final |
| `scripts/53_generate_tcc_outputs.py` | Adicionar tabela de classificação e confusion matrix |

### Novos diretórios

```
models/classification_v4/       # 9 modelos classificação .ubj
docs/dashboard_v4.html           # Landing page HTML estática
```

---

## 9. Métricas de Sucesso

| Métrica | Alvo |
|---------|------|
| CA-01 atingido | |Var A R²_log - 0.8698| < 0.001 (requer INMET 2022) |
| Optuna melhoria | Reportar se ≥ 0.005 R²_log acima da baseline B |
| Classificação F1_macro | ≥ 0.60 para Var B INMET enriquecido |
| Dashboard funcional | `streamlit run dashboard/app.py` sem erro, todas as seções renderizam |
| Landing page funcional | `docs/dashboard_v4.html` abre em browser, todos os gráficos renderizam |
| Pipeline completa | `bash scripts/rodar_v4.sh` exit 0 em < 8h |

---

## 10. Plano de Testes

| ID | Verificação | Script |
|----|------------|--------|
| T-14 | Optuna: todos os 30 trials completam sem erro | 49 |
| T-15 | Optuna: best_params estão dentro dos limites do espaço de busca | 49 |
| T-16 | Classificação: F1_macro > 0 para todas as 4 classes em todos os modelos | 54 |
| T-17 | Classificação: `risco_surto_t4` tem 4 classes (0,1,2,3) no teste | 54 |
| T-18 | Dashboard: `streamlit run dashboard/app.py` inicia sem erro | manual |
| T-19 | Dashboard: todos os filtros (UF, meso, ano, variante) funcionam | manual |
| T-20 | Landing page: abre em Chrome/Safari sem erro de console | manual |
| T-21 | Landing page: todos os gráficos Chart.js renderizam com dados | manual |
| T-22 | Landing page: dark mode toggle funciona | manual |
| T-23 | Pipeline: `rodar_v4.sh` inclui scripts 49, 54, 55 e termina com exit 0 | rodar_v4.sh |

---

## 11. Cronograma

Prazo TCC: **~30/11/2026**. Treino e artefatos prontos até **~20/11**.

| Fase | Período | Entregas | Dias |
|------|---------|----------|------|
| **F8: INMET 2022** | 03-04/11 | Download manual + re-run pipeline, CA-01 validado | 2 |
| **F9: Optuna** | 05-06/11 | Script 49, 30 trials, report JSON, decisão usar/não usar | 2 |
| **F10: Classificação** | 07-08/11 | Script 54, 9 modelos, métricas, tabela LaTeX, confusion matrix | 2 |
| **F11: Dashboard Streamlit** | 09-11/11 | `dashboard/app.py` v4, todas as seções, testado no browser | 3 |
| **F12: Landing Page HTML** | 12-13/11 | Script 55, `docs/dashboard_v4.html`, responsivo, dark mode | 2 |
| **F13: Integração + Validação** | 14-15/11 | rodar_v4.sh atualizado, run completa, CA verificados | 2 |
| **Buffer** | 16-18/11 | Ajustes finais, incorporar no Overleaf | 3 |

**Total: ~16 dias** (03-18/11)

---

## 12. Riscos

| # | Risco | Prob. | Impacto | Mitigação |
|---|-------|-------|---------|-----------|
| 1 | INMET 2022 não disponível no portal | Baixa | Alto | Aceitar CA-01 com nota e documentar no TCC |
| 2 | Optuna não melhora ≥ 0.005 | Média | Baixo | Resultado negativo é informação: manter params originais |
| 3 | Classificação com F1 baixo | Média | Médio | Classes desbalanceadas → usar class_weight="balanced" ou SMOTE |
| 4 | Streamlit não disponível no ambiente | Baixa | Médio | `pip install streamlit plotly` — sem outras deps |
| 5 | Tempo Optuna > 2.5h | Média | Baixo | Limitar timeout ou reduzir trials para 20 |

---

## 13. Open Questions

1. **Limiares de classificação**: Os percentis atuais (Q50/Q75/Q90 do treino, por mesorregião) são adequados para 4 classes? Ou prefere 3 classes (baixo/médio/alto)?
2. **Dashboard deploy**: O Streamlit será usado apenas localmente ou há planos de deploy (Streamlit Cloud, Heroku)?
3. **INMET 2009**: Vale recuperar também? Tem apenas 1 semana de dados no silver, impacto mínimo.
4. **Overleaf**: Quantas novas tabelas/figuras o texto final do TCC comporta? (Atualmente: 5 tabelas + 9 figuras. Este PRD adicionaria ~2 tabelas + 1-2 figuras.)

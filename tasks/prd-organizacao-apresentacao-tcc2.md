# PRD: Organizacao, Revisao e Preparacao para Apresentacao do TCC2

## Introduction

Organizar, revisar e validar TODOS os artefatos do TCC2 (dados, modelos, API, dashboard, Overleaf) para que estejam prontos para demonstracao na defesa perante banca academica de Eng. Software da UnB. O entregavel final e: Overleaf completo com texto + Dashboard Streamlit rodando + backup no Colab/Kaggle.

## Goals

- Validar que todos os scripts, modelos e dados estao funcionais e reproduziveis
- Garantir que o dashboard Streamlit roda localmente e exibe todas as secoes
- Garantir que a API FastAPI sobe e responde corretamente
- Confirmar que o Overleaf compila sem erros e todas as 13 figuras renderizam
- Preparar notebook de backup no Colab/Kaggle como plano B para demo
- Documentar comandos exatos para rodar cada componente na apresentacao

## User Stories

### US-001: Validar dados nas 3 camadas (Bronze, Silver, Gold)
**Description:** Como apresentador, quero confirmar que os dados existem e estao integros em cada camada, para poder mostrar a arquitetura medallion na defesa.

**Acceptance Criteria:**
- [x] Bronze SINAN: `data/sinan/bronze/sinan_tcc2_v2/` — OK (inventory presente)
- [x] Silver SINAN: `data/sinan/silver/sinan_tcc2_v2/` — OK (27 parquets anuais)
- [x] Gold SINAN: `data/sinan/gold/sinan_tcc2_v2/` — OK (3.966 arquivos + feature_catalog)
- [x] Bronze INMET: `data/inmet/bronze/hourly/` — OK (24 dirs, 689 estacoes)
- [x] Silver INMET: `data/inmet/silver/` — OK (24 parquets + coverage_report)
- [x] Gold INMET: `data/inmet/gold/` — OK (26 parquets + feature_catalog)
- [x] Integrado: 270 MB, 7.665.428 linhas x 172 cols, 5.565 municipios, 1999-2026
- [x] Model-ready: train 5.822.243 x 164, val 873.731 x 164, test 947.194 x 164, schema OK
- [x] Integridade confirmada via pd.read_parquet()

### US-002: Validar modelos treinados (DF + Nacional)
**Description:** Como apresentador, quero confirmar que todos os modelos estao salvos e carregam corretamente, para poder demonstrar predicoes ao vivo.

**Acceptance Criteria:**
- [x] DF Regression baseline: OK — carrega com XGBRegressor
- [x] DF Regression tuned: OK — carrega com XGBRegressor
- [x] DF Classification: OK — carrega com XGBClassifier
- [x] DF SINAN-only: OK — carrega com XGBRegressor
- [x] Nacional regression: OK — 2.3 MB, carrega corretamente
- [x] SHAP: OK — shap_report.json + shap_feature_importance.csv
- [x] Walk-forward: OK — walk_forward_results.csv
- [x] Metricas: R2_log=0.7592, R2_orig=0.3088, MAE_orig=6.2

### US-003: Testar Dashboard Streamlit
**Description:** Como apresentador, quero que o dashboard rode localmente sem erros e mostre todas as secoes, para fazer demo ao vivo na defesa.

**Acceptance Criteria:**
- [x] Executar `python3 -m streamlit run dashboard/app.py --server.port 8501` — sobe OK
- [x] Dashboard abre no browser em http://localhost:8501 — HTTP 200
- [ ] Secao "Metricas do Modelo Nacional" mostra R²=0.7592, R²_orig=0.3088 — **TESTAR NO BROWSER**
- [ ] Filtro de UF funciona (selecionar DF, SP, etc.) — **TESTAR NO BROWSER**
- [ ] Filtro de Ano funciona (selecionar 2024, 2025) — **TESTAR NO BROWSER**
- [ ] Secao "Alertas de Surto" mostra municipios com risco Alto/Surto — **TESTAR NO BROWSER**
- [ ] Secao "Serie Temporal por Municipio" mostra grafico Plotly interativo — **TESTAR NO BROWSER**
- [ ] Secao "Distribuicao de Risco" mostra pie chart com 4 niveis — **TESTAR NO BROWSER**
- [ ] Secao "Desempenho por UF" mostra bar chart colorido — **TESTAR NO BROWSER**
- [ ] Secao "Explorador de Dados" permite ver dados brutos — **TESTAR NO BROWSER**
- [ ] Dashboard nao crasha ao trocar filtros rapidamente — **TESTAR NO BROWSER**

### US-004: Testar API FastAPI
**Description:** Como apresentador, quero que a API responda corretamente para demonstrar o sistema de alertas.

**Acceptance Criteria:**
- [x] Executar `python3 -m uvicorn api.main:app --port 8000` — sobe OK
- [x] GET / — retorna models_loaded=["reg_nacional"], test_rows=947194
- [x] GET /health — retorna status=healthy
- [x] GET /predict/5300108?ano=2024 — 52 rows, DF com risco=surto (4220 notif previstas)
- [x] GET /predict/3550308?ano=2024 — 53 rows, SP com risco=surto (1318 notif)
- [x] GET /municipios?limit=5 — SP(49392), DF(22278), BH(20689), RJ(14477)
- [x] Swagger UI OK em /docs
- [x] GET /alerts?min_risk=3 — OK com fallback via thresholds de regressao (sem cls model)

### US-005: Corrigir dashboard para funcionar SEM modelo de classificacao
**Description:** Como desenvolvedor, preciso que o dashboard e API funcionem apenas com o modelo de regressao (sem xgb_cls_nacional.ubj), derivando risco via thresholds nas predicoes.

**Acceptance Criteria:**
- [x] Dashboard: fallback com thresholds fixos (<=1 baixo, <=5 medio, <=20 alto, >20 surto)
- [x] API /alerts: funciona com regressao only via mesmos thresholds
- [x] Metricas card: substituido F1 macro por MAE (orig) — sem crash com valor None
- [x] Dashboard e API sobem sem erros sem xgb_cls_nacional.ubj — testado

### US-006: Verificar Overleaf completo
**Description:** Como apresentador, quero confirmar que o Overleaf compila sem erros e contem todo o conteudo necessario.

**Acceptance Criteria:**
- [x] Todas as 13 figuras existem e tem tamanho > 100 KB
- [x] Todos os 13 \includegraphics apontam para arquivos existentes (0 MISSING)
- [x] Capitulo Execucao: 17 sections, menciona XGBoost + RF + SARIMA + Nacional + Dashboard
- [x] Capitulo Resultados: 14 sections, tabelas comparativas, R²=0.7592, tabela per-UF
- [x] Capitulo Conclusao: 4 sections, 3 algoritmos, contribuicoes, limitacoes, trabalhos futuros
- [x] Zero referencias a Claude/Co-Author (grep confirmado)
- [ ] Compilar no Overleaf e verificar PDF — **VOCE PRECISA FAZER ISSO MANUALMENTE**

### US-007: Preparar backup Colab/Kaggle
**Description:** Como apresentador, quero ter notebooks no Colab/Kaggle como plano B caso o Mac falhe na demo.

**Acceptance Criteria:**
- [x] 3 notebooks existem: 1_dados (54 cells), 2_treino (46 cells), 3_resultados (44 cells)
- [ ] Verificar dados no HuggingFace: `pedrolucassantanaf/dengue-tcc2-data` — **VOCE PRECISA VERIFICAR**
- [ ] Testar notebook 3_resultados no Kaggle — **VOCE PRECISA TESTAR**
- [ ] Documentar URL do Kaggle — **VOCE PRECISA ANOTAR A URL**

### US-008: Criar guia de comandos para a defesa
**Description:** Como apresentador, quero um guia rapido (cheat sheet) com todos os comandos para rodar cada componente durante a defesa.

**Acceptance Criteria:**
- [x] Criado `DEMO_GUIDE.md` com:
  - Comandos para dashboard e API
  - URLs (localhost:8501, localhost:8000/docs)
  - Exemplos de curl para todos os endpoints
  - Caminhos das camadas Bronze/Silver/Gold
  - Tabela de modelos salvos com metricas
  - Roteiro de 5 min para demo ao vivo (6 passos)
  - Numeros-chave para citar na defesa
- [x] Troubleshooting incluido: porta ocupada, modelo nao carrega, dashboard lento, streamlit not found

## Functional Requirements

- FR-1: Todos os arquivos Parquet de dados (Bronze/Silver/Gold/Integrado/Model-ready) devem abrir com `pd.read_parquet()` sem erros
- FR-2: Todos os modelos .json e .ubj devem carregar com `xgb.XGBRegressor().load_model()` ou `XGBClassifier().load_model()`
- FR-3: O dashboard Streamlit deve subir em < 30 segundos e renderizar todas as secoes
- FR-4: A API FastAPI deve responder em < 2 segundos para qualquer endpoint
- FR-5: O dashboard deve funcionar corretamente mesmo sem o modelo de classificacao nacional
- FR-6: O Overleaf deve compilar sem erros de `\includegraphics` (todas as 13 figuras presentes)
- FR-7: Nenhum artefato deve conter referencia a Claude ou Co-Author

## Non-Goals

- NAO treinar novos modelos ou refazer experimentos
- NAO alterar a arquitetura de dados ou pipelines
- NAO adicionar features novas ao dashboard
- NAO fazer deploy em servidor externo (apenas local)
- NAO modificar o conteudo academico do Overleaf (so verificar)

## Technical Considerations

- **Dependencias ja instaladas:** fastapi 0.117.1, streamlit 1.50.0, plotly 6.3.1, xgboost 2.1.4, shap 0.49.1
- **Modelo de classificacao nacional (xgb_cls_nacional.ubj) NAO existe** — o treinamento de 4 classes x 6.7M rows era muito lento. O dashboard e API precisam de fallback para funcionar so com regressao.
- **Metricas nacionais** (metrics_nacional.json) NAO contem F1_macro — o campo existe com valor null
- **Dados no Mac local:** ~600 MB em `data/`, projeto total ~1 GB
- **Portas:** dashboard usa 8501, API usa 8000 — verificar que nao estao ocupadas
- **Colab/Kaggle:** dados hospedados no HuggingFace (repo privado)

## Success Metrics

- Dashboard sobe e funciona em < 30 segundos sem erros
- API responde corretamente a todos os 5 endpoints
- Todas as 13 figuras aparecem no PDF do Overleaf
- Apresentador consegue fazer demo completa em < 5 minutos seguindo o guia
- Zero crashs durante demonstracao com trocas rapidas de filtros

## Open Questions

- O notebook `3_resultados_df.ipynb` no Kaggle ainda funciona ou precisa de atualizacao? (treinou RF e SARIMA?)
- Precisa de nomes de municipios (nao so codigos IBGE) no dashboard para a demo ficar mais intuitiva?
- Quer adicionar screenshots do dashboard no Overleaf como figuras?

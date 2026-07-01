# Guia de Demonstracao — TCC2 Dengue

## Comandos Rapidos

### Dashboard (principal para a defesa)
```bash
cd /Users/filippoferrari/Documents/UnB/TCC/MODELO-PREVISAO
python3 -m streamlit run dashboard/app.py --server.port 8501
```
Abrir: http://localhost:8501

### API REST
```bash
cd /Users/filippoferrari/Documents/UnB/TCC/MODELO-PREVISAO
python3 -m uvicorn api.main:app --port 8000
```
Swagger: http://localhost:8000/docs

### Testar API via curl
```bash
# Root
curl http://localhost:8000/

# Previsao para o DF (2024)
curl "http://localhost:8000/predict/5300108?ano=2024"

# Previsao para Sao Paulo
curl "http://localhost:8000/predict/3550308?ano=2024"

# Top municipios
curl "http://localhost:8000/municipios?limit=10"

# Alertas de surto
curl "http://localhost:8000/alerts?min_risk=3&limit=10"

# Alertas filtrados por UF
curl "http://localhost:8000/alerts?uf=MG&min_risk=2"
```

---

## Roteiro de Demo (5 min)

### 1. Dashboard — Visao Geral (1 min)
- Abrir http://localhost:8501
- Mostrar metricas: R² log = 0.7592, R² orig = 0.3088, MAE = 6.2
- Explicar: "Modelo treinado em 6.7M linhas, 5.565 municipios"

### 2. Dashboard — Filtros e Alertas (1 min)
- Selecionar UF = "DF" no sidebar
- Mostrar tabela de alertas de surto (risco Alto/Surto)
- Selecionar UF = "SP" para comparar

### 3. Dashboard — Serie Temporal (1 min)
- Selecionar municipio no dropdown (DF = 5300108)
- Mostrar grafico real vs previsto interativo
- Passar mouse sobre picos de 2024 ("distribution shift")

### 4. Dashboard — Desempenho por UF (30s)
- Scroll ate "Desempenho por UF"
- Mostrar bar chart: SC=0.82, SP=0.81, PR=0.79 (melhores)
- Destacar que TODAS as 27 UFs tem R² > 0.30

### 5. API — Swagger (30s)
- Abrir http://localhost:8000/docs
- Executar /predict/5300108 ao vivo
- Mostrar JSON com notificacoes_previstas e nivel de risco

### 6. Dados — Arquitetura Medallion (1 min)
- Explicar no Overleaf (slide/texto)
- Bronze: dados brutos SINAN (27 arquivos) + INMET (567 estacoes)
- Silver: limpeza, agregacao semanal, 33.4M notificacoes
- Gold: features derivadas, 7.6M linhas, 172 features
- Model-ready: splits temporais (train 5.8M, val 873K, test 947K)

---

## Dados — Caminhos das Camadas

| Camada | SINAN | INMET |
|--------|-------|-------|
| Bronze | `data/sinan/bronze/` | `data/inmet/bronze/hourly/` |
| Silver | `data/sinan/silver/` (27 parquets anuais) | `data/inmet/silver/` (24 parquets) |
| Gold | `data/sinan/gold/` (3.966 arquivos) | `data/inmet/gold/` (26 parquets) |
| Integrado | `data/integrated/sinan_inmet_municipal_weekly.parquet` (270 MB) | |
| Model-ready | `data/model_ready/{train,val,test}.parquet` | |

---

## Modelos Salvos

| Modelo | Caminho | Metricas |
|--------|---------|----------|
| DF Baseline | `models/xgb_regression_mvp/model.json` | R²_log=0.46 |
| DF Tuned | `models/xgb_regression_tuned/model.json` | R²_log=0.46 |
| DF SINAN-only | `models/xgb_baseline_sinan_only/model_sinan_only.json` | R²_log=0.49 |
| DF Classificacao | `models/xgb_classification_mvp/model.json` | F1=0.45 |
| Nacional Reg | `models/nacional/xgb_reg_nacional.ubj` | R²_log=0.76 |
| SHAP | `models/shap_analysis/` | top: confirmados, lag_1 |
| Walk-forward | `models/walk_forward_results/` | estavel por ano |

---

## Backup — Kaggle/Colab

Notebooks em `colab/`:
- `1_dados_df.ipynb` — exploracao de dados
- `2_treino_df.ipynb` — treinamento de modelos
- `3_resultados_df.ipynb` — resultados e figuras

Dados no HuggingFace: `pedrolucassantanaf/dengue-tcc2-data`

---

## Troubleshooting

**Porta ocupada:**
```bash
lsof -i :8501  # dashboard
lsof -i :8000  # API
kill -9 <PID>
```

**Modelo nao carrega:**
```bash
python3 -c "import xgboost as xgb; m=xgb.XGBRegressor(); m.load_model('models/nacional/xgb_reg_nacional.ubj'); print('OK')"
```

**Dashboard lento na primeira carga:**
Normal — carrega 56 MB de dados de teste + modelo. Segunda vez usa cache.

**streamlit command not found:**
Usar `python3 -m streamlit run ...` em vez de `streamlit run ...`

**API: endpoint /alerts retorna lista vazia:**
Verificar filtros. Sem filtro de ano, busca em todos os anos. Tentar: `curl "http://localhost:8000/alerts?min_risk=2&limit=5"`

---

## Numeros-Chave para Citar na Defesa

- **33.4 milhoes** de notificacoes SINAN processadas
- **567 estacoes** INMET, **5.565 municipios**
- **7.665.428 linhas** no dataset integrado, **172 features**
- **R² log = 0.7592** no modelo nacional (ganho de 64% vs DF-only)
- **27/27 UFs** com R² > 0.30
- **3 algoritmos** comparados: XGBoost, Random Forest, SARIMA
- **SARIMA falhou** (R² < 0): modelos univariados nao capturam dinâmica multivariada
- **Distribution shift 2024**: 5.6x o maximo historico

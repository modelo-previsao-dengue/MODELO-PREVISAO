# TCC - Previsão de Surtos de Dengue no Brasil - Modelo

## 📋 Sobre o Projeto

Este repositório contém o trabalho de conclusão de curso (TCC) do curso de Engenharia de Software da UnB, focado no desenvolvimento de um modelo de previsão para surtos de dengue em municípios brasileiros utilizando séries temporais e dados climáticos.

**Título:** *Desenvolvimento de um Modelo de Previsão para Surtos de Dengue em Municípios Brasileiros utilizando Séries Temporais e Dados Climáticos*

## 🎯 Objetivos

Este repositório serve para **duas grandes finalidades**:

1. **Documentação Completa**: Guardar toda a documentação do TCC, incluindo:
   - Informações sobre o tema
   - Papers e artigos científicos
   - TCCs anteriores relacionados ao tema
   - Referências bibliográficas
   - Estudos e análises

2. **Análise de Dados e Modelagem**: Desenvolvimento de:
   - Scripts para coleta e processamento de dados
   - Agentes de IA para análise dos dados
   - Modelos de previsão (SARIMA, XGBoost, LSTM, etc.)
   - Pipelines de ETL e análise

## 🔬 Tema do TCC

**Previsão de surtos de dengue utilizando séries temporais combinadas com dados climáticos**

Existe vasta literatura mostrando que variáveis climáticas (temperatura, precipitação, umidade) têm forte influência sobre a dinâmica da dengue. Modelos de previsão combinando séries temporais epidemiológicas (SINAN/DataSUS) com covariáveis climáticas (INMET, satélites) apresentam bons resultados. Técnicas modernas incluem LSTM (redes recorrentes), modelos híbridos espaço-temporais, e abordagens de ML com seleção de janelas temporais.

## 📂 Estrutura do Repositório

```
TCC1/
├── README.md                 # Este arquivo
├── docs/                     # Documentação completa do TCC
│   ├── 01-tema-e-motivacao.md
│   ├── 02-referencias-bibliograficas.md
│   ├── 03-bases-de-dados.md
│   ├── 04-metodologia.md
│   ├── 05-plano-de-trabalho.md
│   └── papers/               # Papers e TCCs em PDF
├── scripts/                  # Scripts para processamento de dados
│   └── data/                 # Scripts de coleta de dados
│       ├── download_sinan.py
│       └── download_climate.py
├── notebooks/                # Jupyter notebooks para análise
│   └── prototipo_inicial.ipynb
├── data/                     # Dados do projeto
│   ├── raw/                  # Dados brutos (não versionados)
│   └── processed/            # Dados processados
├── models/                   # Modelos treinados
└── agents/                   # Agentes de IA para análise
```

## 🗃️ Fontes de Dados

### Dados Epidemiológicos
- **OpenDataSUS / SINAN**: Casos de dengue por município e semana epidemiológica
- **DataSUS / TabNet**: Dados agregados de notificações de dengue

### Dados Climáticos
- **INMET / BDMEP**: Estações meteorológicas (temperatura, precipitação, umidade)
- **CHIRPS**: Dados de precipitação por satélite
- **ERA5**: Reanalysis climática (Copernicus)
- **NASA POWER**: API para dados meteorológicos

### Dados Auxiliares
- **IBGE**: Malhas municipais, população, indicadores socioeconômicos
- **InfoDengue (Fiocruz)**: Sistema de alerta e dados processados

## 🛠️ Tecnologias e Ferramentas

### Linguagens e Bibliotecas
- **Python**: Linguagem principal para análise e modelagem
  - `pandas`, `numpy`: Manipulação de dados
  - `geopandas`, `xarray`: Dados geoespaciais e climáticos
  - `scikit-learn`: Modelos de ML tradicionais
  - `xgboost`: Gradient boosting
  - `tensorflow`/`keras`: Redes neurais (LSTM)
  - `statsmodels`: Modelos estatísticos (SARIMA)
  - `prophet`: Previsão de séries temporais

### Infraestrutura
- **Armazenamento**: PostgreSQL + PostGIS para dados estruturados
- **Versionamento**: Git + GitHub
- **Notebooks**: Jupyter / Google Colab
- **Orquestração**: Prefect / Airflow (futuro)

## 📊 Modelos a Serem Testados

1. **Modelos Estatísticos (Baseline)**
   - ARIMA / SARIMA / SARIMAX
   - Prophet

2. **Machine Learning**
   - Random Forest
   - XGBoost
   - Gradient Boosting

3. **Deep Learning**
   - LSTM (Long Short-Term Memory)
   - BiLSTM
   - Temporal CNN
   - Transformers para séries temporais

## 📈 Métricas de Avaliação

- **Previsão de Contagem**: RMSE, MAE, MAPE
- **Classificação de Surtos**: AUC, Precision, Recall, F1-Score
- **Validação Temporal**: Rolling-window cross-validation

## 🚀 Como Começar

### Pré-requisitos
```bash
python >= 3.9
pip install -r requirements.txt
```

### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/TCC1.git
cd TCC1

# Instale as dependências
pip install -r requirements.txt

# Execute os notebooks
jupyter lab notebooks/
```

## 📚 Documentação

Toda a documentação detalhada está disponível na pasta [`docs/`](./docs/):

- [Tema e Motivação](./docs/01-tema-e-motivacao.md)
- [Referências Bibliográficas](./docs/02-referencias-bibliograficas.md)
- [Bases de Dados](./docs/03-bases-de-dados.md)
- [Metodologia](./docs/04-metodologia.md)
- [Plano de Trabalho](./docs/05-plano-de-trabalho.md)

## 👨‍💻 Autor

**Pedro Lucas e Thiago**
- Curso: Engenharia de Software - UnB
- Trabalho: TCC1 (Trabalho de Conclusão de Curso)

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuições

Este é um projeto acadêmico, mas sugestões e feedback são sempre bem-vindos! Sinta-se à vontade para abrir issues ou entrar em contato.

---

⚠️ **Nota**: Os dados brutos não são versionados no repositório devido ao tamanho. Utilize os scripts na pasta `scripts/data/` para fazer o download das fontes oficiais.


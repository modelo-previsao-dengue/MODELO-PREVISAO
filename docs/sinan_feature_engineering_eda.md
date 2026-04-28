# Engenharia de Atributos e EDA - SINAN

Este documento formaliza a etapa 3 do cronograma do TCC2 para o recorte SINAN.

## Contexto no cronograma

Esta entrega corresponde a:

- `3. Engenharia de Atributos e Análise Exploratória`
- `Construção dos Atributos`
- `Análise Exploratória Estatística`

Documento de fechamento da task:

- [Task Concluída - Engenharia de Atributos e EDA - SINAN](./task_engenharia_atributos_eda_sinan.md)

A etapa foi posicionada depois da consolidação da base e antes do treinamento dos modelos, para transformar o recorte semanal do SINAN em uma camada analítica pronta para modelagem preditiva.

## Fontes utilizadas

- Portal oficial do SINAN: https://portalsinan.saude.gov.br/
- Base dos Dados - SINAN: https://basedosdados.org/dataset/f51134c2-5ab9-4bbc-882f-f1034603147a
- Microdados oficiais processados pela pipeline do projeto em `scripts/sinan_pipeline.py`

## Entrada analítica

O script desta fase consome principalmente:

- `data/processed/sinan/weekly_features_brasilia_2022_2024.csv`
- `data/processed/sinan/cluster_assignments_brasilia_2022_2024.csv`
- `data/processed/sinan/cluster_assignments_profile_brasilia_2022_2024.csv`

No recorte atual:

- município: `Brasília/DF`
- código: `5300108`
- período: `2022-2024`
- semanas epidemiológicas: `156`
- notificações totais: `399.094`

## Objetivos da etapa

1. Expandir a base semanal com atributos temporais úteis para previsão.
2. Consolidar indicadores clínicos mais interpretáveis do que dezenas de flags isoladas.
3. Quantificar sazonalidade, variação, picos e associações estatísticas do recorte.
4. Produzir artefatos reprodutíveis para justificar a seleção de atributos na fase de modelagem.

## Script da etapa

Arquivo principal:

`scripts/sinan_feature_engineering_eda.py`

Execução:

```bash
cd MODELO-PREVISAO
python3 scripts/sinan_feature_engineering_eda.py
```

## Atributos construídos

Os atributos foram organizados em blocos:

### 1. Volume histórico

- `notificacoes_lag_1`, `lag_2`, `lag_3`, `lag_4`, `lag_8`, `lag_12`

Esses atributos capturam dependência temporal curta, média e sazonal do processo epidemiológico.

### 2. Janelas móveis

- médias móveis
- desvios móveis
- mínimos móveis
- máximos móveis

Foram usadas janelas de `3`, `4`, `8` e `12` semanas, sempre deslocadas para evitar vazamento temporal.

### 3. Variação temporal

- diferenças absolutas semanais
- variações percentuais
- aceleração do crescimento
- razão entre valor corrente e médias móveis

Esses atributos ajudam a capturar transições de regime e aceleração epidêmica.

### 4. Sazonalidade

- `semana_sin`
- `semana_cos`

Esse par representa a posição da semana epidemiológica ao longo do ciclo anual sem quebrar a circularidade entre semana `52` e semana `1`.

### 5. Indicadores compostos clínicos

- `indice_sintomas`
- `indice_comorbidades`
- `indice_alarme`
- `indice_gravidade`
- `indice_hemorragico`
- `indice_carga_clinica`
- `indice_desfecho_severo`
- `indice_confirmacao`

Esses índices resumem blocos clínicos em variáveis mais estáveis para análise e modelagem.

### 6. Rótulos auxiliares

- `alerta_notificacoes_q75`
- `alerta_notificacoes_q90`
- `alerta_crescimento`

Eles não substituem o alvo final da modelagem, mas já deixam disponível um ponto de partida para tarefas de classificação de alerta.

## Principais achados exploratórios do recorte atual

### Notificações

- média semanal: `2.558,29`
- mediana semanal: `721`
- desvio padrão: `4.923,99`
- coeficiente de variação: `1,9247`
- limiar do quartil superior: `1.783,75`
- limiar do decil superior: `6.614,5`

O comportamento semanal é fortemente assimétrico, com explosão epidêmica concentrada em 2024.

### Resumo anual

- `2022`: `73.027` notificações, média semanal `1.404,37`, pico `4.860`
- `2023`: `43.083` notificações, média semanal `828,52`, pico `3.279`
- `2024`: `282.984` notificações, média semanal `5.442,00`, pico `23.293`

### Semana de pico

- semana epidemiológica `202408`
- início aproximado: `2024-02-19`
- notificações: `23.293`

### Correlações de maior magnitude com `NOTIFICACOES`

As maiores correlações apareceram em:

- `notificacoes_lag_1`
- `notificacoes_max_movel_3`
- `notificacoes_media_movel_3`
- `notificacoes_lag_2`
- `notificacoes_max_movel_4`
- `indice_sintomas`
- `dor_costas_flag`
- `artrite_flag`
- `dor_retro_flag`

Isso sugere que a série mantém forte inércia temporal e que alguns sinais clínicos acompanham a intensificação semanal das notificações.

## Artefatos gerados

O script produz:

- `data/processed/sinan/weekly_model_features_brasilia_2022_2024.csv`
- `data/processed/sinan/feature_catalog_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_descriptive_stats_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_yearly_summary_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_peak_weeks_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_correlations_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_cluster_summary_brasilia_2022_2024.csv`
- `data/processed/sinan/eda_summary_brasilia_2022_2024.json`
- `data/processed/sinan/relatorio_eda_brasilia_2022_2024.md`

## Relação com a próxima fase

Esta etapa prepara diretamente a fase `4. Desenvolvimento de Modelos Preditivos`, porque entrega:

- atributos autoregressivos para modelos de séries temporais e gradient boosting
- indicadores clínicos resumidos para reduzir dimensionalidade
- rótulos auxiliares para classificação de semanas críticas
- diagnóstico estatístico do recorte para apoiar escolha de modelos e validação

## Execução nacional completa

Além do recorte de Brasília, o projeto passou a contar com uma execução nacional consolidada da série oficial:

- `data/processed/sinan/weekly_features_brasil_2000_2026.csv`
- `data/processed/sinan/weekly_model_features_brasil_2000_2026.csv`
- `data/processed/sinan/relatorio_sinan_brasil_2000_2026.md`
- `data/processed/sinan/relatorio_eda_brasil_2000_2026.md`

Essa versão cobre o Brasil inteiro no período `2000-2026` e usa diretamente os microdados anuais oficiais do SINAN/Dengue.

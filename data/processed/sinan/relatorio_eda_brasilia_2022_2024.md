# Engenharia de Atributos e Analise Exploratoria - SINAN

## 1. Escopo

- Recorte processado: `brasilia_2022_2024`
- Semanas epidemiologicas analisadas: `156`
- Cobertura temporal: `202201` ate `202452`
- Intervalo calendario aproximado: `2022-01-03` ate `2024-12-23`

## 2. Objetivo da etapa 3

- Transformar a agregacao semanal do SINAN em uma camada explicita de atributos para modelagem preditiva.
- Caracterizar o comportamento temporal, clinico e epidemiologico do recorte antes do treinamento dos modelos.
- Produzir artefatos reprodutiveis para justificar a escolha de atributos no TCC2.

## 3. Engenharia de atributos

- Blocos de atributos gerados: volume historico, janelas moveis, variacao temporal, sazonalidade, perfis clinicos compostos e rotulos auxiliares.
- Total de atributos catalogados na nova camada: `49`

Principais grupos:

- `identificacao`: 6 atributos
- `janela_movel`: 16 atributos
- `perfil_clinico`: 8 atributos
- `sazonalidade`: 2 atributos
- `targets_auxiliares`: 3 atributos
- `variacao`: 7 atributos
- `volume`: 7 atributos

## 4. Comportamento das notificacoes

- Total de notificacoes no periodo: `399094`
- Media semanal: `2558.2949`
- Mediana semanal: `721.0`
- Desvio padrao semanal: `4923.9882`
- Coeficiente de variacao: `1.9247`
- Limiar de semanas altas (Q75): `1783.75`
- Limiar de semanas extremas (Q90): `6614.5`
- Semana de pico: `202408` com `23293` notificacoes

## 5. Resumo anual

- 2022: 73027 notificacoes, media semanal 1404.37, pico 4860
- 2023: 43083 notificacoes, media semanal 828.52, pico 3279
- 2024: 282984 notificacoes, media semanal 5442.00, pico 23293

## 6. Correlacoes com notificacoes

Atributos com maior associacao linear absoluta com `NOTIFICACOES`:

- `notificacoes_lag_1`: correlacao de Pearson `0.9766`
- `notificacoes_max_movel_3`: correlacao de Pearson `0.9520`
- `notificacoes_media_movel_3`: correlacao de Pearson `0.9405`
- `notificacoes_lag_2`: correlacao de Pearson `0.9346`
- `notificacoes_max_movel_4`: correlacao de Pearson `0.9329`
- `notificacoes_media_movel_4`: correlacao de Pearson `0.9138`
- `notificacoes_min_movel_3`: correlacao de Pearson `0.9106`
- `alerta_notificacoes_q90`: correlacao de Pearson `0.9037`
- `notificacoes_lag_3`: correlacao de Pearson `0.8745`
- `notificacoes_desvio_movel_8`: correlacao de Pearson `0.8647`
- `notificacoes_min_movel_4`: correlacao de Pearson `0.8575`
- `notificacoes_max_movel_8`: correlacao de Pearson `0.8406`
- `indice_sintomas`: correlacao de Pearson `0.8349`
- `dor_costas_flag`: correlacao de Pearson `0.8305`
- `artrite_flag`: correlacao de Pearson `0.8213`
- `notificacoes_desvio_movel_4`: correlacao de Pearson `0.8068`
- `notificacoes_desvio_movel_12`: correlacao de Pearson `0.8016`
- `dor_retro_flag`: correlacao de Pearson `0.7998`
- `notificacoes_lag_4`: correlacao de Pearson `0.7933`
- `notificacoes_media_movel_8`: correlacao de Pearson `0.7775`
- `notificacoes_desvio_movel_3`: correlacao de Pearson `0.7577`
- `notificacoes_max_movel_12`: correlacao de Pearson `0.7429`
- `febre_flag`: correlacao de Pearson `0.7159`
- `laco_flag`: correlacao de Pearson `0.7152`
- `cefaleia_flag`: correlacao de Pearson `0.6963`

## 7. Semanas de pico

- `202408`: 23293 notificacoes, variacao semanal `0.0735`, indice de alarme `0.1863`
- `202405`: 22357 notificacoes, variacao semanal `0.1707`, indice de alarme `0.2017`
- `202409`: 21995 notificacoes, variacao semanal `-0.0557`, indice de alarme `0.1820`
- `202406`: 21921 notificacoes, variacao semanal `-0.0195`, indice de alarme `0.1937`
- `202407`: 21698 notificacoes, variacao semanal `-0.0102`, indice de alarme `0.1867`
- `202410`: 19210 notificacoes, variacao semanal `-0.1266`, indice de alarme `0.1807`
- `202404`: 19097 notificacoes, variacao semanal `0.4739`, indice de alarme `0.1961`
- `202411`: 17343 notificacoes, variacao semanal `-0.0972`, indice de alarme `0.1892`

## 8. Leitura dos clusters

- O cruzamento entre cluster de intensidade e cluster clinico ajuda a diferenciar semanas apenas volumosas de semanas volumosas com mudanca no perfil clinico.

- Intensidade `transicao` + perfil `perfil_raro`: 13 semanas, media `17390.2` notificacoes
- Intensidade `transicao` + perfil `perfil_extremo`: 6 semanas, media `6464.3` notificacoes
- Intensidade `transicao` + perfil `perfil_agudo`: 2 semanas, media `3392.5` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_agudo`: 23 semanas, media `2315.5` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_extremo`: 16 semanas, media `1545.8` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_alterado`: 48 semanas, media `676.8` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_intermediario`: 24 semanas, media `459.6` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_basal`: 24 semanas, media `247.7` notificacoes

## 9. Uso na proxima fase

- Os lags e janelas moveis permitem modelagem temporal autoregressiva.
- Os indicadores compostos clinicos agregam dezenas de sinais em variaveis de interpretacao mais simples.
- Os rotulos auxiliares podem apoiar tarefas de classificacao de alerta e comparacao com modelos de regressao.
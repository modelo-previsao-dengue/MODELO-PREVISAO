# Engenharia de Atributos e Analise Exploratoria - SINAN

## 1. Escopo

- Recorte processado: `brasil_2000_2026`
- Semanas epidemiologicas analisadas: `1375`
- Cobertura temporal: `200001` ate `202615`
- Intervalo calendario aproximado: `2000-01-03` ate `2026-04-06`

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

- Total de notificacoes no periodo: `33417771`
- Media semanal: `24303.8335`
- Mediana semanal: `9397.0`
- Desvio padrao semanal: `43998.4073`
- Coeficiente de variacao: `1.8103`
- Limiar de semanas altas (Q75): `23230.5`
- Limiar de semanas extremas (Q90): `59630.4`
- Semana de pico: `202412` com `433513` notificacoes

## 5. Resumo anual

- 2000: 174643 notificacoes, media semanal 3295.15, pico 7991
- 2001: 473332 notificacoes, media semanal 8930.79, pico 25884
- 2002: 880232 notificacoes, media semanal 16927.54, pico 77839
- 2003: 403566 notificacoes, media semanal 7614.45, pico 23848
- 2004: 133751 notificacoes, media semanal 2523.60, pico 6523
- 2005: 250857 notificacoes, media semanal 4824.17, pico 9876
- 2006: 390424 notificacoes, media semanal 7508.15, pico 21156
- 2007: 718996 notificacoes, media semanal 13826.85, pico 39505
- 2008: 918957 notificacoes, media semanal 17338.81, pico 88454
- 2009: 609750 notificacoes, media semanal 11725.96, pico 33357
- 2010: 1374384 notificacoes, media semanal 26430.46, pico 67794
- 2011: 1142350 notificacoes, media semanal 21968.27, pico 63626
- 2012: 948065 notificacoes, media semanal 18232.02, pico 47895
- 2013: 2017154 notificacoes, media semanal 38791.42, pico 137522
- 2014: 970810 notificacoes, media semanal 18317.17, pico 56012
- 2015: 2421177 notificacoes, media semanal 46561.10, pico 139437
- 2016: 2259456 notificacoes, media semanal 43451.08, pico 154114
- 2017: 514769 notificacoes, media semanal 9899.40, pico 19011
- 2018: 481391 notificacoes, media semanal 9257.52, pico 18941
- 2019: 2258032 notificacoes, media semanal 43423.69, pico 142065
- 2020: 1548867 notificacoes, media semanal 29223.91, pico 87353
- 2021: 1010359 notificacoes, media semanal 19429.98, pico 44526
- 2022: 1405095 notificacoes, media semanal 27021.06, pico 103841
- 2023: 1645956 notificacoes, media semanal 31653.00, pico 110655
- 2024: 6564924 notificacoes, media semanal 126248.54, pico 433513
- 2025: 1646533 notificacoes, media semanal 31066.66, pico 90350
- 2026: 253941 notificacoes, media semanal 16929.40, pico 26371

## 6. Correlacoes com notificacoes

Atributos com maior associacao linear absoluta com `NOTIFICACOES`:

- `notificacoes_lag_1`: correlacao de Pearson `0.9857`
- `notificacoes_media_movel_3`: correlacao de Pearson `0.9589`
- `notificacoes_max_movel_3`: correlacao de Pearson `0.9569`
- `notificacoes_lag_2`: correlacao de Pearson `0.9552`
- `notificacoes_min_movel_3`: correlacao de Pearson `0.9540`
- `notificacoes_media_movel_4`: correlacao de Pearson `0.9407`
- `notificacoes_max_movel_4`: correlacao de Pearson `0.9389`
- `notificacoes_min_movel_4`: correlacao de Pearson `0.9295`
- `notificacoes_lag_3`: correlacao de Pearson `0.9122`
- `notificacoes_lag_4`: correlacao de Pearson `0.8574`
- `notificacoes_max_movel_8`: correlacao de Pearson `0.8462`
- `notificacoes_media_movel_8`: correlacao de Pearson `0.8401`
- `notificacoes_min_movel_8`: correlacao de Pearson `0.7769`
- `notificacoes_max_movel_12`: correlacao de Pearson `0.7531`
- `alerta_notificacoes_q90`: correlacao de Pearson `0.7422`
- `notificacoes_desvio_movel_4`: correlacao de Pearson `0.7294`
- `notificacoes_desvio_movel_3`: correlacao de Pearson `0.7288`
- `notificacoes_media_movel_12`: correlacao de Pearson `0.7179`
- `notificacoes_desvio_movel_8`: correlacao de Pearson `0.7165`
- `notificacoes_desvio_movel_12`: correlacao de Pearson `0.6979`
- `alerta_notificacoes_q75`: correlacao de Pearson `0.6295`
- `notificacoes_min_movel_12`: correlacao de Pearson `0.5803`
- `notificacoes_lag_8`: correlacao de Pearson `0.5517`
- `semana_sin`: correlacao de Pearson `0.4592`
- `indice_confirmacao`: correlacao de Pearson `0.4088`

## 7. Semanas de pico

- `202412`: 433513 notificacoes, variacao semanal `0.1057`, indice de alarme `0.0802`
- `202415`: 419145 notificacoes, variacao semanal `0.0101`, indice de alarme `0.0940`
- `202414`: 414963 notificacoes, variacao semanal `0.1049`, indice de alarme `0.0896`
- `202411`: 392067 notificacoes, variacao semanal `0.0748`, indice de alarme `0.0737`
- `202416`: 383330 notificacoes, variacao semanal `-0.0854`, indice de alarme `0.0998`
- `202413`: 375565 notificacoes, variacao semanal `-0.1337`, indice de alarme `0.0868`
- `202410`: 364793 notificacoes, variacao semanal `0.0650`, indice de alarme `0.0759`
- `202409`: 342519 notificacoes, variacao semanal `0.1039`, indice de alarme `0.0756`

## 8. Leitura dos clusters

- O cruzamento entre cluster de intensidade e cluster clinico ajuda a diferenciar semanas apenas volumosas de semanas volumosas com mudanca no perfil clinico.

- Intensidade `transicao` + perfil `perfil_intermediario`: 331 semanas, media `28380.6` notificacoes
- Intensidade `baixa_transmissao` + perfil `perfil_basal`: 1044 semanas, media `23011.3` notificacoes

## 9. Uso na proxima fase

- Os lags e janelas moveis permitem modelagem temporal autoregressiva.
- Os indicadores compostos clinicos agregam dezenas de sinais em variaveis de interpretacao mais simples.
- Os rotulos auxiliares podem apoiar tarefas de classificacao de alerta e comparacao com modelos de regressao.
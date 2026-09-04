# Guia técnico do retreino (recorte 2019-2023)

Este documento explica o que mudou no pipeline, por que cada decisão foi
tomada, o que cada métrica significa e como ajustar o modelo. Serve como
referência para escrever o TCC e para retomar o trabalho depois de um tempo
sem mexer nele.

Branch: `feat/retreino-recorte-2019-2023` · Plano de origem:
[`tasks/prd-retreino-recorte-2019-2023.md`](../tasks/prd-retreino-recorte-2019-2023.md)

---

## 1. Resumo em uma página

O modelo anterior reportava R² de 0,42 e concluía que o clima não ajudava a
prever dengue. As duas afirmações estavam apoiadas em dados quebrados.

Foram encontrados **cinco defeitos**, três previstos no plano e dois
descobertos durante a implementação:

| # | Defeito | Efeito no resultado anterior |
|---|---|---|
| 1 | Colunas do INMET mapeadas por posição errada | A "umidade" do modelo era temperatura de ponto de orvalho |
| 2 | Chuva ausente virava zero | Semanas sem estação funcionando entravam como "não choveu" |
| 3 | Semanas de virada de ano partidas (INMET) | 2.878 semanas com agregação parcial |
| 4 | Semanas de virada de ano duplicadas (SINAN) | 1.393 semanas com contagem parcial, alvo contaminado |
| 5 | Limiares de risco calculados sobre o dataset inteiro | Rótulo de risco vazava validação e teste para dentro do treino |

Mais **três limitações de método**:

- Os *lags* climáticos iam só até 8 semanas, mas o pico da correlação
  clima-casos está em 9-12 semanas. O modelo nunca viu a janela relevante.
- Não havia nenhum baseline. Comparava-se XGBoost com XGBoost.
- O Optuna ajustava hiperparâmetros em 500 mil linhas e o modelo final
  treinava em 2 milhões; o SHAP explicava 50 mil linhas de um modelo
  treinado em 2 milhões.

Tudo isso foi corrigido. O resultado honesto, com baseline ao lado, está na
[seção 7](#7-resultados).

---

## 2. Glossário de siglas

### Métricas

| Sigla | Nome | Em uma frase |
|---|---|---|
| **MAE** | Mean Absolute Error | Erro médio em casos, sem punir erro grande além do proporcional |
| **RMSE** | Root Mean Squared Error | Como o MAE, mas eleva o erro ao quadrado antes de tirar a média: pune desproporcionalmente o erro grande |
| **R²** | Coeficiente de determinação | Fração da variação dos casos que o modelo explica. 1 é perfeito, 0 é "igual a chutar a média", negativo é pior que chutar a média |
| **MAPE** | Mean Absolute Percentage Error | Erro médio em porcentagem. Indefinido quando o valor real é zero |
| **F1** | F1-score | Média harmônica entre precisão e recall. Resume o acerto numa classe desbalanceada |
| **AUC** | Area Under the ROC Curve | Probabilidade de o modelo dar nota maior a um caso positivo do que a um negativo. 0,5 é sorteio, 1,0 é perfeito |
| **OvR** | One-vs-Rest | Como transformar um problema de 4 classes em 4 problemas de "esta classe contra todas as outras" |

### Dados e método

| Sigla | Significado |
|---|---|
| **SINAN** | Sistema de Informação de Agravos de Notificação — a fonte dos casos de dengue |
| **INMET** | Instituto Nacional de Meteorologia — a fonte dos dados climáticos |
| **IBGE** | Instituto Brasileiro de Geografia e Estatística — a fonte dos códigos de município |
| **UF** | Unidade Federativa (estado) |
| **WMO** | World Meteorological Organization — o código que identifica cada estação do INMET |
| **SE** | Semana epidemiológica — semana que começa no domingo, usada em vigilância |
| **t+4** | O alvo: número de casos daqui a 4 semanas |
| **lag k** | O valor da variável k semanas atrás |
| **MM4 / mm4** | Média móvel de 4 semanas |
| **E1 / E2** | Os dois experimentos: E1 é o recorte 2019-2023, E2 é o histórico completo |
| **SHAP** | SHapley Additive exPlanations — método para dizer quanto cada feature contribuiu para cada previsão |
| **TPE** | Tree-structured Parzen Estimator — o algoritmo que o Optuna usa para escolher o próximo conjunto de hiperparâmetros |
| **OOM** | Out Of Memory — quando o sistema mata um processo por falta de memória |

---

## 3. As métricas explicadas

### Por que toda métrica agora vem com a escala escrita

O modelo é treinado sobre `log1p(casos)`, isto é, `log(1 + casos)`, e não
sobre a contagem crua. Isso é padrão para dados de contagem com cauda longa:
sem a transformação, um município com 3.000 casos numa semana domina o
treino inteiro e o modelo vira um preditor de São Paulo.

O problema é que **o R² sobre log e o R² sobre contagem são números muito
diferentes**, e o relatório anterior publicava um só, sem dizer qual:

```
R2_log  = 0,7057   ← quanto o modelo explica da variação de log(1+casos)
R2_orig = 0,3611   ← quanto o modelo explica da variação dos casos
```

Os dois são verdadeiros e medem coisas diferentes. `R2_log` é a métrica que o
treino otimiza. `R2_orig` é a que responde à pergunta do TCC: "quanto dos
casos de dengue este modelo explica?". Publicar só o primeiro é otimista;
publicar só o segundo esconde o que o modelo está de fato aprendendo. Por
isso `scripts/metrics_common.py` devolve sempre os dois, com sufixo.

### MAE vs RMSE: quando um diz uma coisa e o outro diz outra

```
MAE  = média de |real - previsto|
RMSE = raiz da média de (real - previsto)²
```

O quadrado no RMSE faz um erro de 100 casos pesar 100 vezes mais que um erro
de 10, e não 10 vezes mais. Consequência prática: **o RMSE é dominado pelos
picos epidêmicos**, e o MAE representa a semana típica.

Se um modelo tem MAE melhor e RMSE pior que outro, ele acerta mais no dia a
dia e erra mais nos surtos. Para vigilância epidemiológica, errar no surto é
justamente o que importa — por isso as duas métricas aparecem juntas.

### R²: por que pode dar negativo

```
R² = 1 - (soma dos erros do modelo)² / (soma dos erros de chutar a média)²
```

Se o modelo erra mais do que erraria alguém que sempre chuta a média
histórica, o R² fica negativo. Foi o que aconteceu com o baseline sazonal
(prever 2023 repetindo 2022): **R²_orig = −0,10**. Isso não é um bug; é a
informação de que 2023 não se pareceu com 2022, e portanto que o ano de
teste é difícil de propósito.

### MAPE: por que virou `MAPE_pct_positivos`

O MAPE divide o erro pelo valor real. Numa semana com zero casos isso é
divisão por zero. Como mais de um terço das linhas do dataset tem zero
notificações, o MAPE "sobre tudo" ou explodia ou era calculado sobre um
subconjunto silencioso. Agora ele é declaradamente calculado só sobre as
linhas com casos, e o número de linhas usadas sai junto (`n_positivos`).

### F1_macro: por que macro e não simples

São quatro classes de risco (baixo, médio, alto, surto) e elas são muito
desbalanceadas — a classe "baixo" é a maioria. Um modelo que responde
"baixo" sempre teria acurácia alta e seria inútil.

- **F1_weighted** pondera cada classe pelo tamanho: ainda premia acertar a
  classe grande.
- **F1_macro** dá o mesmo peso às quatro: um modelo que ignora "surto" é
  punido de verdade.

O F1_macro é a métrica que interessa aqui, porque a classe rara é
exatamente a que a vigilância quer prever.

### AUC macro OvR

Para cada uma das 4 classes, mede-se a capacidade de separar "é esta classe"
de "não é", e tira-se a média. Diferente do F1, o AUC não depende do corte de
decisão — mede a qualidade do *ranking* das probabilidades. Um AUC alto com
F1 baixo significa que o modelo ordena bem mas o limiar de decisão está mal
escolhido.

---

## 4. As correções, uma a uma

### 4.1 Mapeamento das colunas do INMET (defeito mais grave)

O CSV horário do INMET tem 19 colunas em ordem fixa. O extrator antigo
atribuía os nomes **por posição**, e a lista divergia do arquivo real a
partir da coluna 8. O resultado:

| Nome no Bronze | Conteúdo real |
|---|---|
| `temp_max_c` | temperatura do ponto de orvalho |
| `temp_min_c` | temperatura máxima real |
| `temp_orvalho_c` | temperatura mínima real |
| `umidade_inst_pct` | ponto de orvalho mínimo |
| `vento_dir_graus` | umidade relativa |
| `vento_rajada_ms` | direção do vento |
| `vento_vel_ms` | rajada de vento |
| — | velocidade do vento: descartada |

A "correlação umidade-casos de 0,777" que o plano citava como evidência
estava medindo ponto de orvalho. E `temp_range_c`, calculado como
`temp_max - temp_min`, era a diferença entre ponto de orvalho e temperatura
máxima — às vezes negativa.

**Correção:** o mapeamento passou a ser por nome do cabeçalho, normalizado
(sem acento, maiúsculo, tolerante à mojibake `REGI?O` dos arquivos de 2019+),
com validação de faixa física. Bronze reextraído para 2018-2024: 35.450.088
registros, 689 estações, zero cabeçalhos não mapeados. Depois da correção
todas as relações físicas se sustentam: `temp_max > temp_inst > temp_min`,
`umidade_max > umidade_inst > umidade_min`, `rajada > velocidade`, direção
entre 1 e 360.

### 4.2 Zeros falsos de chuva

A agregação semanal fazia `sum()` sobre a chuva horária. Em pandas, a soma de
um grupo vazio é `0.0`, não nulo. Semana em que a estação não reportou nada
entrava no modelo como **"choveu 0 mm"**, indistinguível de uma semana seca
de verdade.

**Correção:** passou a contar as horas válidas de precipitação
(`n_valid_rain_hours`) e mascarar como nulo quando são zero. Há uma
verificação automática que aborta o processo se alguma linha tiver
`n_valid_rain_hours == 0` e `rain_sum_mm` não nulo.

**Efeito colateral honesto:** a cobertura climática real é bem pior do que
parecia. Com os zeros falsos, a chuva "nunca faltava". Sem eles, a cobertura
das três variáveis exigidas juntas vai de 89,2% em 2019 a 57,2% em 2021.

### 4.3 e 4.4 Semanas partidas na virada de ano

Os dados são armazenados em arquivos anuais, mas a semana epidemiológica que
cruza 31 de dezembro pertence a dois arquivos. Cada arquivo continha só a
parte da semana que caiu nele.

- **No INMET:** 8.379 estação-semanas existiam em duplicata, cada uma com uma
  agregação parcial. O código anterior resolvia com
  `drop_duplicates(keep="first")` — ou seja, ficava com um fragmento e jogava
  o outro fora. Exemplo real: A001/2020-W52 tinha um fragmento de 120 horas /
  0,2 mm e outro de 48 horas / 6,0 mm; o segundo era descartado.
- **No SINAN:** a mesma coisa, 1.393 semanas-município no histórico completo
  e 113 no recorte. Um lado trazia a contagem quase completa, o outro um
  resto.

**Correção:** agregação em duas etapas no INMET (somas e contagens parciais
por ano, combinadas por chave depois) e consolidação por soma no SINAN. No
SINAN isso recuperou **7.899 notificações** que a escolha de um fragmento
descartaria. Isso importa mais do que o volume sugere, porque um valor
parcial de `notificacoes` contamina também o alvo `notificacoes_t4` da linha
quatro semanas antes.

### 4.5 Vazamento temporal no rótulo de risco

O rótulo `risco_surto_t4` (baixo/médio/alto/surto) sai de percentis p50, p75 e
p90 das notificações de cada município. Esses percentis eram calculados
**sobre o dataset inteiro**, antes de separar treino, validação e teste.

Isso é vazamento: o rótulo de uma linha de 2023 dependia de quanto se
notificou em 2023, informação que o modelo não teria na hora da previsão.

**Correção:** os percentis saem só do treino. Município que o treino nunca
viu recebe o percentil da própria UF, e a origem de cada limiar fica
registrada em `threshold_source` para auditoria. Resultado: **27.842 rótulos
(7,76%) mudam** quando o vazamento sai.

O rótulo antigo foi mantido como `risco_surto_t4_com_vazamento` — não para
usar, mas para medir de quanto era a inflação. Ver seção 7.

### 4.6 Lags climáticos até 12 semanas

O ciclo é: chove → o mosquito se reproduz → as pessoas são picadas → adoecem
→ procuram o serviço → o caso é notificado. Isso leva mais de 8 semanas. A
auditoria mostrou o pico da correlação clima-casos em *lag* 9-12 em 11 de 12
UFs, fora do alcance dos *lags* `[1, 2, 4, 8]` que existiam.

**Correção:** *lags* `[1, 2, 4, 8, 12]` e médias móveis `[4, 8, 12]`, agora
espelhando o que as features epidemiológicas do SINAN já tinham. As features
climáticas foram de 27 para 49.

Duas sutilezas importantes:

- Os *lags* são calculados sobre uma **grade semanal completa**. Sem isso,
  `shift(12)` em uma série com buracos anda 12 *linhas*, não 12 semanas — e o
  "lag de 12 semanas" podia ser de 15.
- São calculados no nível da **estação**, não do município. O mapeamento
  estação-município é um para um, então o resultado é idêntico, mas o custo
  cai de 5.571 séries para ~600.

### 4.7 Correções de método

- **Baselines** (`scripts/baselines_naive.py`): persistência, sazonal e média
  móvel de 4 semanas, avaliados no mesmo teste, com as mesmas métricas e no
  mesmo formato, para entrarem na mesma tabela.
- **Sem subamostragem**: o Optuna agora tuna sobre o treino inteiro. Isso
  importa porque `max_depth` e `min_child_weight` ótimos dependem do tamanho
  da amostra — tunar em 500 mil e treinar em 2 milhões entrega
  hiperparâmetros escolhidos para outro problema.
- **`notificacoes` da semana atual virou feature.** O código antigo excluía
  `notificacoes` (semana t) mas mantinha `notificacoes_lag_1` (semana t−1),
  o que não faz sentido: se t−1 é conhecido, t também é. Pior, isso
  handicapava o modelo contra o baseline de persistência, que usa exatamente
  esse valor. Agora é uma chave de configuração, ligada por padrão.

---

## 5. O recorte: o que foi trocado por quê

O modelo antigo treinava em todo o Brasil, de 2000 a 2024. O clima só existe
onde há estação automática do INMET funcionando, o que é uma fração pequena
disso. O recorte troca **abrangência por qualidade**:

| Dimensão | Antes | Agora (E1) |
|---|---|---|
| UFs | 27 | 6 (DF, ES, GO, MG, RJ, SP) |
| Anos | 2000-2024 | 2019-2023 |
| Municípios | 5.571 | 1.369 |
| Linhas | ~2,0 milhões | 358.678 |
| Cobertura climática | não medida | 94,26% |

As 6 UFs foram escolhidas pela cobertura climática medida depois das
correções: DF 99,73%, MG 91,00%, RJ 90,86%, GO 85,06%, ES 84,92%. SP entrou
com 78,77% por peso epidemiológico. SC (80,64%) ficou de fora por ter pouca
dengue.

**O custo, dito com todas as letras:** 19 anos de histórico epidemiológico
foram trocados por cobertura climática. Isso penaliza o braço SINAN-only da
ablação, que teria muito mais dados se pudesse usar o histórico inteiro. Por
isso existe o E2: o mesmo modelo SINAN-only sobre 2000-2023, com o **mesmo
ano de teste (2023)**, para que a diferença entre E1 e E2 não misture efeito
de recorte com efeito de período de teste.

### O split temporal

```
E1:  treino 2019-2021  |  validação 2022  |  teste 2023
E2:  treino 2000-2020  |  validação 2021-2022  |  teste 2023
```

Nunca aleatório. Série temporal com split aleatório coloca o futuro no treino
e infla tudo.

Há ainda uma janela de **warm-up (2018)** e **cool-down (2024)** que é
processada mas não entregue: o warm-up existe para que os *lags* de 12
semanas de janeiro de 2019 tenham de onde vir, e o cool-down para que o alvo
t+4 das últimas semanas de 2023 exista.

---

## 6. Decisões de engenharia

### Por que a VM do WSL morria

Sem `.wslconfig`, o WSL2 pode inflar até toda a RAM do host (aqui, ~15,9 GB).
Quando um script pedia 8-15 GB, o Windows derrubava a VM inteira antes que o
OOM killer do Linux pudesse matar só o processo Python. O sintoma era "o
terminal fecha sozinho"; a causa era a VM reiniciando (`uptime` de 1 minuto
depois de cada morte).

Duas providências:

1. **`scripts/rodar_com_teto.sh`** roda qualquer comando dentro de um cgroup
   com `MemoryMax`. Se o script exagerar, o Linux mata o processo e imprime o
   erro; a sessão sobrevive. `ulimit -v` **não serve**: limita espaço
   virtual, e o XGBoost com 16 threads reserva dezenas de GB virtuais usando
   menos de um residente.
2. **Os scripts ficaram leves.** Picos de memória depois:

| Etapa | Antes | Depois |
|---|---|---|
| `prepare_model_dataset --modo e2` | >15 GB (matava a VM) | 3,0 GB |
| `prepare_model_dataset --modo e1` | 7,5 GB | 5,2 GB |
| `baselines_naive` no E2 | >12 GB | 1,9 GB |

As técnicas: ler só as colunas necessárias; converter `float64` para
`float32` na leitura (o XGBoost trabalha em `float32` de qualquer forma);
ler o Parquet em lotes para o `float64` nunca existir inteiro; usar as
estatísticas de nulos do próprio Parquet em vez de ler os dados; e escrever
os splits ano a ano com `ParquetWriter` em vez de concatenar tudo.

Recomendado também criar `C:\Users\<usuário>\.wslconfig`:

```ini
[wsl2]
memory=11GB
swap=8GB
```

Requer `wsl --shutdown` para valer.

### Configuração em vez de constante (FR-1)

Nenhum script tem janela ou lista de UFs escrita no código. Tudo vem de
[`config/recorte.json`](../config/recorte.json). Trocar o recorte é editar um
JSON e rodar de novo, não caçar constantes em cinco arquivos.

---

## 7. Resultados

Tabela completa e sempre atualizada em
[`docs/resultados_modelos_e_baselines.md`](resultados_modelos_e_baselines.md),
gerada por `scripts/tabela_resultados.py`. Nenhum número de XGBoost aparece
sem a linha de baseline ao lado (FR-4).

### 7.1 E1 — recorte 2019-2023, 1.369 municípios, teste em 2023

| Modelo | MAE (casos) | RMSE | R²_orig | R²_log | F1_macro | AUC |
|---|---|---|---|---|---|---|
| baseline: persistência | 9,449 | 83,31 | **0,4003** | 0,5700 | **0,4842** | — |
| baseline: média móvel 4 | 10,891 | 84,59 | 0,3817 | 0,5689 | 0,4583 | — |
| baseline: sazonal | 14,046 | 91,61 | −0,1038 | 0,0473 | 0,3236 | — |
| XGBoost SINAN-only | 8,060 | 85,99 | 0,3611 | 0,7057 | 0,4011 | 0,7775 |
| XGBoost SINAN+INMET | **7,935** | **84,02** | 0,3900 | **0,7068** | 0,3992 | **0,7859** |

**O clima ajuda.** Com os dados corrigidos, o braço com INMET supera o
SINAN-only em MAE (7,935 contra 8,060), R²_orig (0,390 contra 0,361), R²_log
e AUC. Isso **inverte a conclusão anterior do TCC**, que dizia que o clima não
contribuía — conclusão que se apoiava em ponto de orvalho rotulado como
umidade, chuva ausente virando zero e *lags* que paravam antes da janela em
que a correlação existe.

**Mas o modelo não bate a persistência em R²_orig.** 0,390 contra 0,400. Ele
ganha em MAE, ou seja, acerta melhor a semana típica, e perde nos picos, que
é o que domina o R². Para vigilância epidemiológica, errar no pico é
exatamente o erro que importa. Este é o resultado honesto e precisa entrar no
TCC como tal.

**E o baseline vence a classificação.** F1_macro 0,4842 contra 0,3992. Isso
não é um paradoxo: o baseline "classifica" passando o valor atual pelos
limiares do município, o que espalha as previsões pelas quatro faixas; o
XGBoost minimiza *logloss* e, sem tratamento de desbalanceamento, escorrega
para a classe majoritária. É um problema com correção conhecida — pesar as
amostras, [seção 8.4](#84-ajustes-práticos-em-ordem-de-retorno) item 4 — e
não foi aplicada aqui para não misturar o efeito com o da ablação.

### 7.2 US-009 — o mesmo recorte agregado por mesorregião

43 mesorregiões em vez de 1.369 municípios, 11.266 linhas em vez de 358.678.

| Modelo | MAE (casos) | RMSE | R²_orig | R²_log | F1_macro | AUC |
|---|---|---|---|---|---|---|
| baseline: persistência | 252,10 | 736,00 | 0,3971 | 0,5322 | **0,5675** | — |
| baseline: média móvel 4 | 304,65 | 786,81 | 0,3110 | 0,6294 | 0,5069 | — |
| baseline: sazonal | 377,74 | 891,72 | −0,1959 | 0,2166 | 0,3699 | — |
| XGBoost SINAN-only | 224,02 | 745,50 | 0,3814 | 0,6924 | 0,4092 | 0,7244 |
| XGBoost SINAN+INMET | **195,65** | **678,65** | **0,4874** | **0,7031** | 0,4336 | **0,7553** |

**A hipótese da granularidade se confirma.** O ganho do clima salta de
**+0,029** de R²_orig no município para **+0,106** na mesorregião — mais que
o triplo. E é só aqui que o modelo **supera a persistência com folga** em
R²_orig (0,487 contra 0,397), o que não acontecia no nível municipal.

A explicação é direta: mais de um terço das linhas municipais tem zero caso,
e uma estação a 80 km representa mal o microclima de um município pequeno.
Agregar aumenta a razão sinal-ruído e aproxima a escala espacial dos casos da
escala em que o clima foi de fato medido — a mediana é de 2,5 estações por
mesorregião.

A ressalva: são só 2.279 linhas de teste, contra 72.557 no municipal. O
intervalo de confiança é bem mais largo, e vale reportar isso junto do
número.

Na classificação a mesorregião também não bate o baseline, pelo mesmo motivo
da seção anterior.

#### A terceira granularidade: UF

| Modelo | MAE (casos) | R²_orig | R²_log | F1_macro | AUC |
|---|---|---|---|---|---|
| baseline: persistência | 1.650,92 | 0,4544 | −0,1518 | **0,5719** | — |
| baseline: média móvel 4 | 2.020,87 | 0,3278 | 0,3348 | 0,5244 | — |
| XGBoost SINAN-only | **1.204,40** | **0,6245** | **0,0872** | 0,3013 | 0,6827 |
| XGBoost SINAN+INMET | 1.305,99 | 0,5279 | 0,0754 | 0,2932 | **0,7180** |

**A curva não é monotônica.** Agregar mais nem sempre é melhor:

| Granularidade | Séries | Linhas de treino | Δ R²_orig do clima |
|---|---|---|---|
| Município | 1.369 | 214.933 | +0,029 |
| **Mesorregião** | **43** | **6.751** | **+0,106** |
| UF | 6 | 942 | **−0,097** |

O ganho do clima cresce do município para a mesorregião e **inverte** na UF.
A leitura mais provável: a mesorregião é grande o bastante para o ruído de
contagem sumir e ainda pequena o bastante para uma média climática significar
algo. Na UF, a mediana é de 17 estações agregadas numa média só — o que sobra
não descreve o clima de lugar nenhum. E com 942 linhas de treino, 49 features
climáticas a mais são convite a decorar.

**Trate o nível de UF como ponto extremo da curva, não como modelo.** São 318
linhas de teste e o R²_log fica perto de zero nos dois braços, ou seja, o
modelo mal explica a variação em escala log. O R²_orig alto vem de acertar a
ordem de grandeza de seis séries grandes, não de prever bem. A inflação do
vazamento também explode aqui (+0,107 de F1_macro contra +0,033 no
município), porque limiares tirados de 942 linhas são instáveis.

**A conclusão metodológica é que existe uma granularidade ótima**, e neste
recorte ela é a mesorregião. Isso é contribuição de método, não resultado
negativo: diz que a pergunta "o clima ajuda a prever dengue?" não tem
resposta independente da escala espacial em que se pergunta.

### 7.3 E2 — histórico completo 2000-2023, SINAN-only, teste em 2023

Mesmo ano de teste do E1, para que a diferença não misture efeito de recorte
com efeito de período.

| Modelo | MAE (casos) | RMSE | R²_orig | R²_log | F1_macro | AUC |
|---|---|---|---|---|---|---|
| baseline: persistência | 4,654 | 53,36 | 0,4352 | 0,5845 | **0,4969** | — |
| baseline: média móvel 4 | 5,383 | 58,26 | 0,3266 | 0,5880 | 0,4773 | — |
| baseline: sazonal | 7,697 | 63,90 | −0,0186 | −0,0118 | 0,3714 | — |
| XGBoost SINAN-only | **3,744** | **50,72** | **0,4897** | **0,7335** | 0,4514 | **0,8402** |

**Os 21 anos de histórico valem mais que o clima no nível municipal.** O E2
supera a persistência com folga (R²_orig 0,490 contra 0,435), o que nenhum
braço do E1 conseguiu, e chega a AUC 0,840 contra 0,786 do melhor braço do
E1. A conta é direta: 6,1 milhões de linhas de treino contra 215 mil.

Isso não contradiz a seção 7.1 — confirma o preço que o recorte cobra e que
está declarado no README. O recorte trocou 19 anos de histórico por cobertura
climática, e no nível municipal essa troca sai cara. Só na mesorregião o
clima recupera a diferença.

**A inflação do vazamento quase some aqui** (+0,0044 de F1_macro contra
+0,0333 no E1). Faz sentido: limiares calculados sobre 21 anos de treino são
estáveis, e ver ou não o ano de teste muda pouco. Quanto menor o histórico,
mais o vazamento importa.

### 7.4 Um aviso sobre o que a busca otimiza

O E2 foi rodado duas vezes, com 1 e com 20 trials. O resultado é instrutivo:

| | RMSE_log val | R²_log teste | R²_orig teste |
|---|---|---|---|
| 1 trial | 0,5179 | 0,7327 | **0,5331** |
| 20 trials | **0,5172** | **0,7335** | 0,4897 |

**A busca melhorou exatamente aquilo que ela otimiza — RMSE na escala log — e
piorou o R² na escala de contagem.** Não é ruído nem erro: é o comportamento
esperado de otimizar uma métrica e reportar outra. Se o objetivo do TCC é
explicar contagens de casos, o objetivo da busca deveria ser o erro em
contagem, e não o erro em log.

Isso está descrito como ajuste prático na
[seção 8.4](#84-ajustes-práticos-em-ordem-de-retorno), item 2, e é
provavelmente o ajuste de maior retorno que sobrou por fazer.

### 7.5 SHAP: de onde vem a previsão

Calculado sobre o conjunto de teste **inteiro**, não sobre uma amostra. A
versão anterior explicava 50 mil linhas sorteadas de um modelo treinado em
milhões, então a importância relatada não era a do modelo entregue.

| Braço | Peso relativo do clima | Climáticas mais fortes |
|---|---|---|
| e1_sinan_inmet | 17,99% | `dewpoint_mean_c_mm4`, `pressure_mean_mbar`, `rain_sum_mm_mm8`, `dewpoint_mean_c_mm8`, `dewpoint_mean_c` |
| meso_sinan_inmet | 17,03% | `rain_sum_mm_mm4`, `rain_sum_mm_mm8`, `temp_min_c`, `dewpoint_mean_c_lag_12`, `rain_sum_mm_mm12` |

Nos dois casos o topo absoluto é epidemiológico — `notificacoes` e
`notificacoes_lag_1` sozinhas respondem por boa parte —, o que é esperado num
alvo com autocorrelação alta e é a mesma razão de a persistência ser um
baseline forte.

Duas leituras que só são possíveis depois das correções:

- **O ponto de orvalho aparece entre as climáticas mais fortes no recorte
  municipal.** Antes da US-000 essa variável estava rotulada como
  `temp_max_c`, então qualquer análise anterior que a mencionasse estava
  falando de outra coisa.
- **Na mesorregião, `dewpoint_mean_c_lag_12` e `rain_sum_mm_mm12` entram no
  top 5.** São features de 12 semanas, que **não existiam** antes da US-003 —
  os lags paravam em 8. O modelo antigo não tinha como enxergar essa janela,
  que é exatamente onde a auditoria dizia estar o pico da correlação.

### 7.6 Inflação causada pelo vazamento (US-002)

Quanto a classificação *parecia* melhor quando os limiares de risco saíam do
dataset inteiro em vez de só do treino:

| Braço | F1_macro sem vazamento | com vazamento | inflação |
|---|---|---|---|
| e1_sinan | 0,4011 | 0,4284 | +0,0273 |
| e1_sinan_inmet | 0,3992 | 0,4325 | +0,0333 |
| meso_sinan | 0,4092 | 0,4503 | +0,0411 |
| meso_sinan_inmet | 0,4336 | 0,4779 | +0,0443 |
| e2_sinan | 0,4514 | 0,4558 | +0,0044 |

Não é enorme, mas era ganho de graça que o modelo não teria em produção — e
7,76% dos rótulos mudam quando o vazamento sai.

### 7.7 O que isso significa para o TCC

1. **O clima contribui, ao contrário do que o trabalho anterior concluiu.** A
   conclusão anterior era artefato de dados quebrados, não um achado.
2. **A escala espacial importa mais que o modelo, e tem um ótimo.** Trocar
   município por mesorregião rendeu mais que qualquer ajuste de
   hiperparâmetro, mas continuar agregando até a UF inverte o sinal. A
   pergunta "o clima ajuda?" não tem resposta independente da escala.
3. **No recorte, o XGBoost não justifica sua complexidade contra a
   persistência em R²_orig** (0,390 contra 0,400). É um resultado negativo
   legítimo e publicável; escondê-lo seria repetir o erro que o retreino veio
   corrigir. Fora do recorte, com o histórico inteiro, o modelo bate a
   persistência com folga (0,490 contra 0,435) — o problema é o tamanho do
   treino, não o algoritmo.
4. **O recorte cobra um preço mensurável.** 215 mil linhas de treino contra
   6,1 milhões. Só ao agregar por mesorregião o clima recupera a diferença.
5. **O classificador precisa de tratamento de desbalanceamento** antes de ser
   comparado de forma justa ao baseline: perde em F1_macro nos cinco braços.
6. **A busca está otimizando a métrica errada** para a pergunta do TCC. Ver
   [seção 7.4](#74-um-aviso-sobre-o-que-a-busca-otimiza).

---

## 8. Como fazer o fine-tuning do modelo

### 8.1 O que o XGBoost está fazendo

O XGBoost constrói árvores de decisão **em sequência**, cada uma tentando
corrigir o erro que as anteriores deixaram. A primeira árvore chuta algo
grosseiro; a segunda aprende onde a primeira errou; e assim por diante. O
"gradient boosting" é isso: cada árvore é treinada sobre o gradiente do erro
acumulado.

Duas consequências práticas:

- **Mais árvores sempre reduzem o erro no treino.** Por isso existe o
  *early stopping*: o treino para quando o erro na **validação** para de
  melhorar, não quando o do treino para.
- **A regularização é o que decide.** Com 178 features e um alvo de cauda
  longa, o modelo decora facilmente. Os parâmetros que controlam o quanto
  ele pode decorar importam mais que a profundidade das árvores.

### 8.2 Os hiperparâmetros, do mais para o menos importante

| Parâmetro | O que faz | Faixa usada | Sintoma de errar |
|---|---|---|---|
| `learning_rate` (eta) | Quanto cada árvore corrige. Menor = mais lento e mais estável | 0,005 – 0,3 | Alto: erro oscila e não converge. Baixo: precisa de muitas árvores |
| `max_depth` | Profundidade máxima de cada árvore. Controla o quanto ela captura interações | 3 – 12 | Alto: decora. Baixo: não captura o efeito combinado de clima e histórico |
| `min_child_weight` | Peso mínimo de amostras para criar um nó novo. Impede o modelo de criar uma regra baseada em 3 municípios | 1 – 100 | Baixo: regras espúrias em municípios minúsculos |
| `subsample` | Fração das linhas usada em cada árvore | 0,5 – 1,0 | 1,0: árvores muito parecidas entre si |
| `colsample_bytree` | Fração das features por árvore | 0,3 – 1,0 | 1,0: todas as árvores olham as mesmas features dominantes |
| `reg_lambda` (L2) | Penaliza pesos grandes nas folhas | 1e-8 – 10 | Baixo: previsões extremas |
| `reg_alpha` (L1) | Empurra pesos para exatamente zero, fazendo seleção de features | 1e-8 – 10 | Alto: modelo perde features úteis |
| `gamma` | Ganho mínimo para valer a pena dividir um nó | 1e-8 – 5 | Alto: árvores rasas demais |
| `n_estimators` | Teto de árvores | 2000 fixo | Não precisa ajustar: o early stopping decide |

**Por que `n_estimators` é fixo em 2000 e não entra na busca:** com
`early_stopping_rounds=50` ele funciona como teto, não como escolha. O
número que o modelo realmente usou fica gravado em `n_arvores_usadas` no
`metrics.json`. Se esse número bater em 2000, aí sim o teto está apertado.

### 8.3 Como a busca funciona

O Optuna usa **TPE** (Tree-structured Parzen Estimator). Em vez de testar
combinações em grade, ele modela quais regiões do espaço deram bons
resultados e concentra as tentativas seguintes ali. Na prática: 30 trials de
TPE valem por centenas de trials aleatórios.

O objetivo otimizado é **RMSE na escala log, medido na validação**:

```python
# scripts/train_ablation.py
def objetivo(trial):
    m = xgb.XGBRegressor(**espaco(trial), **FIXOS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return float(np.sqrt(np.mean((yva - m.predict(Xva)) ** 2)))
```

A validação (2022) nunca é usada para treinar — só para escolher os
hiperparâmetros e para parar o treino. O teste (2023) é tocado **uma vez**,
no fim. Se você olhar o teste, ajustar, e olhar de novo, o teste virou
validação e o número deixa de valer.

### 8.4 Ajustes práticos, em ordem de retorno

**1. Aumentar o número de trials.** É o ajuste de maior retorno e menor
risco.

```bash
scripts/rodar_com_teto.sh -m 9G -- \
  .venv/bin/python3 scripts/train_ablation.py --arm e1_sinan_inmet --trials 100
```

Custo: linear. 30 trials no E1 levam ~5 minutos; 100 levam ~17.

**2. Mudar o que está sendo otimizado.** Hoje a busca minimiza RMSE em log,
o que privilegia acertar a semana típica. Se o objetivo do TCC é **detectar
surto**, o alvo da otimização deveria ser outro. Em `tune_regressao`, troque
o retorno do objetivo por:

```python
# Erro só nas semanas de surto (acima do p90 do município)
mascara = yva > np.log1p(limiar_p90_por_linha)
return float(np.sqrt(np.mean((yva[mascara] - pred[mascara]) ** 2)))
```

ou otimize diretamente o F1_macro do classificador. Essa é uma decisão de
método, não de código: define o que o modelo considera "bom".

**3. Ajustar as faixas de busca.** Depois de rodar, olhe `melhores_params` no
`metrics.json`. **Se um parâmetro escolhido está colado na borda da faixa, a
faixa está apertada.** Exemplo: se `max_depth` escolhido for 12 (o teto),
amplie para 16 em `espaco()` e rode de novo.

**4. Pesar as amostras.** O XGBoost aceita `sample_weight` no `fit`. Dar peso
maior às semanas de surto faz o modelo priorizá-las:

```python
peso = 1 + 4 * (partes["train"][CLASS_TARGET] == 3)   # surto pesa 5x
modelo.fit(Xtr, ytr, sample_weight=peso, eval_set=[(Xva, yva)])
```

Isso costuma melhorar o F1_macro e piorar o MAE global. É um trade-off
explícito, e vale documentá-lo no TCC como tal.

**5. Trocar o objetivo interno do XGBoost.** Para contagem, `count:poisson` e
`reg:tweedie` costumam bater `reg:squarederror` sobre log1p, porque modelam a
distribuição dos dados em vez de transformá-la:

```python
FIXOS = {"objective": "count:poisson", "tree_method": "hist", ...}
# e treinar sobre y cru, não log1p(y)
```

Vale testar como um braço a mais da ablação.

**6. Validação cruzada temporal (walk-forward).** Hoje há um só corte de
validação (2022). Se 2022 for atípico, os hiperparâmetros escolhidos são bons
para 2022 e não para o problema. O walk-forward treina em 2019, valida em
2020; treina em 2019-2020, valida em 2021; e assim por diante, e usa a média.
Mais robusto e proporcionalmente mais caro.

### 8.5 Sinais de que algo está errado

| Sintoma | Provável causa |
|---|---|
| R²_log alto e R²_orig baixo | Normal aqui. O modelo acerta a ordem de grandeza e erra o pico |
| `n_arvores_usadas` = 2000 | O teto de árvores está apertado; aumente `n_estimators` |
| `n_arvores_usadas` < 30 | `learning_rate` alto demais, ou o modelo não está aprendendo nada |
| Métrica na validação muito melhor que no teste | Overfitting aos hiperparâmetros, ou 2022 é mais fácil que 2023 |
| Uma feature com importância dominante | Suspeite de vazamento: ela contém informação do futuro? |
| Modelo pior que a persistência | O modelo não aprendeu além da autocorrelação. É um resultado válido e precisa ser reportado |

### 8.6 Onde mexer

| O que mudar | Arquivo |
|---|---|
| Recorte, UFs, anos, split | `config/recorte.json` |
| Espaço de busca dos hiperparâmetros | `scripts/train_ablation.py`, função `espaco()` |
| Parâmetros fixos (early stopping, n_estimators) | `scripts/train_ablation.py`, dicionário `FIXOS` |
| Objetivo da otimização | `scripts/train_ablation.py`, função `tune_regressao` |
| Definição das métricas | `scripts/metrics_common.py` |
| Baselines | `scripts/baselines_naive.py` |

---

## 9. Como rodar o pipeline

Sempre com teto de memória, para que um erro mate o processo e não a VM:

```bash
# 1. INMET: extrair, agregar por semana, gerar features municipais
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_extract_standardize.py --years 2018-2024
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_weekly_aggregate.py --years 2018-2024
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_municipal_features.py --years 2018-2024

# 2. Integrar SINAN + INMET na janela útil
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/integrate_sinan_inmet.py

# 3. Datasets prontos para modelo
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/prepare_model_dataset.py --modo e1
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/prepare_model_dataset.py --modo e2
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/prepare_mesoregion_dataset.py

# 4. Baselines (rodar ANTES dos modelos: nenhum resultado vai sem eles)
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/baselines_naive.py
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/baselines_naive.py \
    --dataset data/model_ready/e2 --saida models/baselines_e2

# 5. Ablação
scripts/rodar_com_teto.sh -m 9G -- .venv/bin/python3 scripts/train_ablation.py --arm e1_sinan
scripts/rodar_com_teto.sh -m 9G -- .venv/bin/python3 scripts/train_ablation.py --arm e1_sinan_inmet
scripts/rodar_com_teto.sh -m 9G -- .venv/bin/python3 scripts/train_ablation.py --arm meso_sinan
scripts/rodar_com_teto.sh -m 9G -- .venv/bin/python3 scripts/train_ablation.py --arm meso_sinan_inmet
scripts/rodar_com_teto.sh -m 9G -- .venv/bin/python3 scripts/train_ablation.py --arm e2_sinan --trials 20

# 6. Explicabilidade (conjunto de teste inteiro) e tabela final
scripts/rodar_com_teto.sh -m 6G -- .venv/bin/python3 scripts/explain_shap.py --arm e1_sinan_inmet
scripts/rodar_com_teto.sh -m 6G -- .venv/bin/python3 scripts/explain_shap.py --arm meso_sinan_inmet
python3 scripts/tabela_resultados.py
```

**Tetos de memória medidos** (pico real, com `/usr/bin/time -v`):

| Etapa | Pico |
|---|---|
| `prepare_model_dataset --modo e2` | 3,0 GB |
| `prepare_model_dataset --modo e1` | 5,2 GB |
| `prepare_mesoregion_dataset` | 5,3 GB |
| `train_ablation --arm e1_*` | 3,0 GB |
| `train_ablation --arm meso_*` | 0,5 GB |
| `train_ablation --arm e2_sinan` | **9,0 GB** |
| `explain_shap` | 2,5 GB |
| `baselines_naive` (E2) | 1,9 GB |

O braço E2 é o único que exige teto alto. Dar menos que 8 GB a ele faz o
cgroup matar o processo no meio da busca.

O Optuna e o SHAP vivem no `.venv`; o resto roda no Python do sistema.

### Onde cada coisa fica

```
config/recorte.json              recorte, split, cobertura mínima
data/inmet/{bronze,silver,gold}/ horário → semanal por estação → semanal por município
data/integrated/                 SINAN + INMET, uma linha por município-semana
data/model_ready/                E1: train/val/test + limiares + esquema
data/model_ready/e2/             E2: histórico completo, SINAN-only
data/model_ready/mesorregiao/    US-009: mesmo recorte, agregado
models/baselines*/               baselines ingênuos
models/ablacao/<braço>/          modelos, métricas e importância de features
docs/                            relatórios de cobertura, recorte e este guia
```


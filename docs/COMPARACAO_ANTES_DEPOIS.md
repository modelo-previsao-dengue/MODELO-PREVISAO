# Antes e depois: o que existe de cada versão

Levantamento do que está guardado no repositório para comparar o retreino com
o que veio antes. Feito para responder "o modelo melhorou?" sem depender de
memória.

---

## 1. Versão DF-only: não há resultado salvo

Os notebooks `colab/1_dados_df.ipynb`, `colab/2_treino_df.ipynb`,
`colab/3_resultados_df.ipynb` e `colab/TCC2_Dengue_XGBoost_DF.ipynb`
**não têm nenhuma saída armazenada** — foram commitados com as células
limpas. Nenhuma métrica de um modelo treinado só no Distrito Federal
sobreviveu.

O único número específico do DF que existe é de outra coisa: o **modelo
nacional avaliado sobre as linhas do DF**, em
`models/nacional/metrics_por_uf.csv`:

| UF | Linhas de teste | R²_log | MAE_orig |
|---|---|---|---|
| DF | 180 | 0,523 | 1.363,0 |

Isso não é um modelo DF-only. É o modelo nacional recortado no DF depois de
treinado — comparar com um modelo treinado só no DF seria comparar coisas
diferentes.

**Se você precisa do número DF-only para o TCC, ele terá de ser regerado.**
Com o recorte novo isso é problemático: o DF tem 157 linhas de treino. O
caminho viável é usar o `EXPERIMENTO = "e2/"`, que dá ao DF ~1.250 linhas,
mas sem clima.

---

## 2. Versão anterior — nacional, pipeline com os cinco defeitos

Treino em 2000–2021, teste em 2024+, 947.194 linhas de teste, 5.565
municípios, 159 features.

### Regressão

| Experimento | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| `xgb_regression_mvp` | 166,72 | 7,466 | 0,3093 | 88,81 |
| `xgb_regression_tuned` (20 trials) | 166,76 | 7,014 | 0,309 | 91,83 |
| `nacional` | — | 6,194 (orig) | **0,309 orig · 0,759 log** | — |

O `xgb_regression_tuned` piorou o RMSE em 0,03% em relação ao MVP: 20 trials
de Optuna sobre 500 mil linhas subamostradas não acharam nada.

### Ablação clima — a conclusão que o retreino inverteu

| Braço | RMSE | MAE | R² |
|---|---|---|---|
| SINAN-only | 152,15 | 6,718 | **0,4248** |
| SINAN+INMET | 166,72 | 7,466 | 0,3093 |

Teste t pareado: t = −5,22, p ≈ 0. Conclusão registrada na época: **o clima
piora o modelo**.

### Classificação

| Métrica | Valor |
|---|---|
| AUC macro | 0,8408 |
| F1 macro | 0,4804 |
| F1 "surto" | 0,627 |
| F1 "médio" | 0,2008 |
| F1 "alto" | 0,2155 |

Estes valores usam os limiares **com vazamento temporal** — os percentis eram
calculados sobre o dataset inteiro.

### SHAP

Amostra de 50 mil linhas de um modelo treinado em milhões. Só **1 feature
climática no top 20** (`rain_sum_mm_mm4`), e os lags climáticos nas posições
**143 a 153** de 159 — ou seja, praticamente irrelevantes.

### Multi-horizonte e walk-forward

| Horizonte | RMSE | MAE | R² |
|---|---|---|---|
| t+1 | 157,20 | 5,294 | 0,3753 |
| t+2 | 160,95 | 5,821 | 0,3488 |
| t+4 | 167,07 | 7,030 | 0,3064 |
| t+8 | 185,54 | 9,521 | 0,1639 |

Walk-forward: 9 folds, RMSE médio 38,47 com desvio de 27,30 — coeficiente de
variação de 71%, marcado como `"stable": false`.

---

## 3. Versão local — o retreino

Teste em 2023 nos quatro conjuntos. Tabela completa em
[`resultados_modelos_e_baselines.md`](resultados_modelos_e_baselines.md).

### E1 — recorte 6 UFs, 72.557 linhas de teste

| Modelo | MAE | R²_orig | R²_log | F1_macro | AUC |
|---|---|---|---|---|---|
| baseline persistência | 9,449 | **0,4003** | 0,5700 | **0,4842** | — |
| XGBoost SINAN-only | 8,060 | 0,3611 | 0,7057 | 0,4011 | 0,7775 |
| XGBoost SINAN+INMET | **7,935** | 0,3900 | **0,7068** | 0,3992 | **0,7859** |

### E2 — histórico completo, 294.945 linhas de teste

| Modelo | MAE | R²_orig | R²_log | AUC |
|---|---|---|---|---|
| baseline persistência | 4,654 | 0,4352 | 0,5845 | — |
| XGBoost SINAN-only | **3,744** | **0,4897** | **0,7335** | **0,8402** |

### Mesorregião e UF — 2.279 e 318 linhas de teste

| Granularidade | SINAN-only R²_orig | SINAN+INMET R²_orig | Δ do clima |
|---|---|---|---|
| Município (E1) | 0,3611 | 0,3900 | **+0,029** |
| Mesorregião | 0,3814 | 0,4874 | **+0,106** |
| UF | 0,6245 | 0,5279 | **−0,097** |

### SHAP

Teste inteiro, sem amostragem. Clima responde por **17,99%** do peso SHAP no
E1 e **17,03%** na mesorregião. As climáticas mais fortes são médias móveis
de ponto de orvalho e de chuva em janelas de 8 a 12 semanas.

---

## 4. O que dá e o que não dá para comparar

**Não compare os números diretamente.** Os conjuntos de teste são diferentes:

| | Antes | Depois |
|---|---|---|
| Ano de teste | 2024+ | 2023 |
| Linhas de teste | 947.194 | 72.557 (E1) · 294.945 (E2) |
| Abrangência | 27 UFs | 6 UFs (E1) · 27 UFs (E2) |
| Features | 159 | 178 (E1) · 129 (E2) |
| Escala do R² reportado | não declarada | declarada |

Um MAE de 7,466 em 2024 e um de 7,935 em 2023 não dizem qual modelo é melhor:
2023 e 2024 tiveram epidemias de tamanhos diferentes.

**Compare as conclusões.** Cada ablação é interna à sua versão, sobre as
mesmas linhas nos dois braços — e é aí que a mudança aparece:

| Pergunta | Antes | Depois |
|---|---|---|
| O clima ajuda? | **Não** — SINAN-only 0,4248 contra 0,3093 | **Sim** — 0,3900 contra 0,3611, e +0,106 na mesorregião |
| Quanto do SHAP é clima? | 1 feature no top 20; lags nas posições 143–153 | **18%** do peso total |
| O modelo bate o trivial? | **não sabido** — não havia baseline | Sim no E2 e na mesorregião; **não** no E1 municipal |
| O rótulo de risco vaza? | Sim, e não era medido | Medido: +0,033 de F1_macro no E1, +0,107 na UF |
| A tunagem ajuda? | Piorou 0,03% com subamostragem | Melhora o RMSE_log; **piora** o R²_orig |

---

## 5. Se quiser um número comparável de verdade

O único jeito honesto de comparar as duas versões no mesmo terreno é
**reavaliar o modelo antigo no conjunto de teste novo**. Os artefatos existem:

```
models/xgb_regression_tuned/  (sem model.json — só metrics e trials)
models/xgb_regression_mvp/model.json
models/xgb_baseline_sinan_only/model_sinan_only.json
models/nacional/  (sem model.json)
```

O problema: o modelo antigo espera as 159 features do esquema antigo, com os
nomes climáticos trocados pelo bug de mapeamento da US-000. As colunas não
batem com as 178 do dataset novo, e forçar o encaixe compararia
`humidity_mean_pct` do modelo velho — que era ponto de orvalho — com a
umidade de verdade.

**Recomendação:** trate as duas versões como trabalhos separados. A versão
anterior entra no TCC como o ponto de partida cujos defeitos motivaram o
retreino, com os números dela citados como estavam. A versão nova é o
resultado. A comparação que sustenta a tese é a das **conclusões**, na tabela
da seção 4, não a dos valores absolutos.

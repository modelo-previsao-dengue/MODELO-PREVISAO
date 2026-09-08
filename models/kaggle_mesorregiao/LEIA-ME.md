# Execução no Kaggle — mesorregião, 08/09/2026

Verificação independente do achado central da US-009. Mesmo dataset do
HuggingFace, mesma configuração do notebook, só mudando `EXPERIMENTO`.

43 mesorregiões, 6.751 linhas de treino, teste 2023 com 2.279 linhas.

---

## A tabela

| Modelo | MAE | RMSE | R²_orig | R²_log | Árvores |
|---|---|---|---|---|---|
| baseline: persistência | 252,10 | 736,00 | 0,3971 | 0,5322 | — |
| baseline: sazonal | 377,74 | 891,72 | −0,1959 | 0,2166 | — |
| XGBoost SINAN-only | 224,13 | 748,60 | 0,3763 | 0,6989 | 174 |
| XGBoost SINAN+INMET | **200,88** | **700,00** | **0,4546** | **0,7052** | 352 |

---

## O que mudou em relação ao município

### 1. O ganho do clima é várias vezes maior

| Granularidade | Ganho no Kaggle | Ganho local |
|---|---|---|
| Município (E1) | +0,0116 | +0,0289 |
| **Mesorregião** | **+0,0783** | **+0,1060** |
| Razão | **6,7×** | 3,7× |

As duas máquinas discordam na magnitude, mas concordam no essencial: o efeito
do clima na mesorregião é **de outra ordem de grandeza**. No município ele é
comparável ao ruído de tunagem; aqui ele é o fator dominante da diferença
entre os dois braços.

**É este o resultado que sustenta a contribuição metodológica do trabalho.**
A pergunta "o clima ajuda a prever dengue?" não tem resposta independente da
escala espacial em que se pergunta.

### 2. Aqui o modelo finalmente bate o baseline

| | R²_orig |
|---|---|
| baseline persistência | 0,3971 |
| XGBoost SINAN+INMET | **0,4546** |

No município nenhum braço superava a persistência (0,3925 contra 0,4003). Na
mesorregião o modelo com clima passa na frente com folga, e é o **único**
arranjo do recorte em que isso acontece.

Repare que o SINAN-only (0,3763) continua **perdendo** para a persistência.
Ou seja: não é a agregação sozinha que faz o modelo valer a pena — é a
agregação **com clima**. Agregar aumenta a razão sinal-ruído a ponto de o
sinal climático virar utilizável.

### 3. A chuva sobe ao topo das climáticas

| | Município | Mesorregião |
|---|---|---|
| 1ª climática | `dewpoint_mean_c_mm4` (15ª geral) | `rain_sum_mm_mm4` (**14ª geral**) |
| `dewpoint_mean_c_lag_12` | 30ª geral | **18ª geral** |

Duas leituras:

- **A chuva agregada passa a valer mais que o ponto de orvalho.** Faz sentido
  físico: chuva é pontual e mal representada por uma estação a 80 km; média
  de 43 mesorregiões suaviza isso. Ponto de orvalho varia menos no espaço e
  já sobrevivia ao ruído municipal.
- **O lag de 12 semanas sobe doze posições.** É a feature que **não existia**
  antes da US-003, quando os lags climáticos paravam em 8. Ela ser mais
  importante na escala onde o clima funciona é confirmação direta de que
  estender a janela era necessário, e não decoração.

### 4. Menos árvores no braço sem clima

174 contra 352. Com 6.751 linhas de treino e sem as features climáticas, o
early stopping corta cedo — não há o que aprender depois disso. O braço com
clima continua melhorando por mais o dobro de árvores.

---

## O que se repete do município

**Os baselines saíram idênticos ao local** (252,0987 e −0,1959). Mesma prova
de sempre: o parquet no HuggingFace é o do disco.

**O sazonal é negativo** (−0,1959, ainda pior que os −0,1038 do município).
2023 não repetiu 2022 em nenhuma escala.

**O walk-forward mostra 2022 mais fácil:** fold 2022 dá R²_orig 0,702 contra
0,4546 no teste de 2023.

---

## Ressalvas para o texto

**São 2.279 linhas de teste**, contra 72.557 no município. O intervalo de
confiança é bem mais largo e isso precisa aparecer junto do número.

**Os hiperparâmetros não são desta granularidade.** Vieram do braço
`e1_sinan_inmet`, que foi tunado em 214.933 linhas com 178 features. Rodar o
Optuna sobre a mesorregião provavelmente melhora os dois braços — e é a
explicação mais provável para o ganho aqui (+0,078) ter ficado abaixo do
local (+0,106), já que localmente cada braço teve busca própria.

**Reporte o intervalo, não um número.** O ganho do clima na mesorregião fica
entre **+0,078 e +0,106** conforme a escolha de hiperparâmetros. A conclusão
não depende disso: nas duas execuções ele é várias vezes maior que no
município, e nas duas o modelo com clima supera a persistência.

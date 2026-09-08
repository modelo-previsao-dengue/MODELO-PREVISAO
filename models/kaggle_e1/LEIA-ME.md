# Execução no Kaggle — E1, 08/09/2026

Resultado da primeira execução no Kaggle depois do retreino. Serve de
verificação independente do pipeline local: mesmos dados, máquina diferente,
GPU em vez de CPU.

Ambiente: Kaggle Notebooks, GPU T4, XGBoost 3.2.0.
Dataset: `thiagorfreitas/dengue-tcc2-data`, `data/model_ready/` (E1).
Notebook: `kaggle/notebook_retreino.py`.

---

## A tabela

| Modelo | MAE | RMSE | R²_orig | R²_log | Árvores |
|---|---|---|---|---|---|
| baseline: persistência | 9,449 | 83,31 | **0,4003** | 0,5700 | — |
| baseline: sazonal | 14,046 | 91,61 | −0,1038 | 0,0473 | — |
| XGBoost SINAN-only | 7,958 | 84,65 | 0,3809 | **0,7108** | 435 |
| XGBoost SINAN+INMET | **7,919** | **83,85** | 0,3925 | 0,7065 | 343 |

Teste: 2023, 72.557 linhas, 1.369 municípios de 6 UFs.

---

## O que cada resultado significa

### 1. Os baselines saíram idênticos ao local

Persistência 9,4491 e R²_orig 0,4003; sazonal 14,0458 e −0,1038. Não é
aproximação — é o mesmo número até a quarta casa.

**Por que importa:** os baselines não têm treino, nem aleatoriedade, nem
GPU envolvida. Eles são função pura dos dados. Sair idêntico prova que o
parquet que o Kaggle baixou é bit a bit o mesmo que está no disco local — o
upload não corrompeu nada e o notebook não está lendo o dataset errado.

Se algum dia esses dois números mudarem sem que os dados tenham mudado, o
problema é de dados, não de modelo.

### 2. O clima ajuda, mas o ganho é menor do que o local sugeria

| | Kaggle | Local |
|---|---|---|
| SINAN-only R²_orig | 0,3809 | 0,3611 |
| SINAN+INMET R²_orig | 0,3925 | 0,3900 |
| **Ganho do clima** | **+0,0116** | **+0,0289** |

O sinal é o mesmo nas duas execuções — o clima contribui, o que já inverte a
conclusão da versão anterior do trabalho. Mas a **magnitude mudou pela
metade**, e a causa não é a GPU.

Localmente, cada braço teve a própria busca de hiperparâmetros com 30 trials
de Optuna. O braço SINAN-only saiu com `max_depth=5, min_child_weight=1`; o
com clima, com `max_depth=11, min_child_weight=93`. O notebook do Kaggle usa
**os parâmetros do braço com clima para os dois**.

Ou seja: o SINAN-only rodou melhor no Kaggle com parâmetros que não eram
dele. Isso é evidência de que a busca local do SINAN-only havia
**superajustado a validação** — encontrou uma combinação boa para 2022 que
não generalizou para 2023.

**Como reportar no TCC:** o ganho do clima em R²_orig fica entre **+0,012 e
+0,029** dependendo de como os hiperparâmetros são escolhidos. Publicar só o
número maior seria escolher a execução mais favorável. O intervalo é o
resultado honesto, e a variação em si é informação: o efeito do clima no
nível municipal é pequeno o bastante para ser da ordem do ruído de tunagem.

### 3. Nenhum modelo bate o baseline mais bobo

R²_orig: persistência **0,4003**, melhor XGBoost **0,3925**.

Repetir o número de hoje quatro semanas à frente explica mais da variação dos
casos do que um XGBoost com 178 features e 30 trials de Optuna.

**Como ler isso.** O modelo ganha em MAE (7,92 contra 9,45), ou seja, erra
menos na semana típica. E perde em R², que é dominado pelos valores grandes.
Traduzindo: **o modelo acerta melhor o dia a dia e erra mais nos picos** — e
o pico é exatamente o que a vigilância epidemiológica precisa prever.

Este é um resultado negativo legítimo, confirmado de forma independente em
duas máquinas. Ele não invalida o trabalho; ele delimita o que o trabalho
mostrou. A conclusão defensável é que **no nível municipal, com três anos de
treino, o XGBoost não justifica sua complexidade contra a persistência** — e
que os caminhos que funcionam estão em outro lugar: agregar por mesorregião
(R²_orig 0,4874) ou usar o histórico completo (0,4897).

### 4. O baseline sazonal é negativo, e isso é bom sinal

R²_orig de **−0,1038**: prever 2023 repetindo a mesma semana de 2022 é pior
do que chutar a média histórica.

**Por que isso é bom:** significa que 2023 não se pareceu com 2022. O ano de
teste não é uma cópia do de validação, então o modelo não pode ter acertado
por decorar sazonalidade. O split temporal está fazendo o trabalho dele.

### 5. O walk-forward confirma que 2023 é difícil

| | MAE | R²_orig | R²_log |
|---|---|---|---|
| fold 2022 (validação) | 4,451 | **0,7159** | 0,7455 |
| teste 2023 | 7,919 | **0,3925** | 0,7065 |

O mesmo modelo, treinado em 2019–2021 e avaliado em 2022, chega a R²_orig de
0,72. No teste de 2023 cai para 0,39 — quase a metade.

**O que isso quer dizer:** a queda não é do modelo, é do ano. 2023 foi uma
epidemia de perfil diferente. Isso é argumento a favor do desenho
experimental: se eu tivesse reportado o desempenho na validação, teria um
número quase duas vezes melhor e completamente enganoso.

Repare que o R²_log quase não muda (0,7455 contra 0,7065). O modelo continua
acertando a ordem de grandeza; o que ele perde em 2023 é a escala dos picos.
É a mesma história do item 3, vista de outro ângulo.

### 6. O SHAP reproduziu a ordem das climáticas

As cinco climáticas mais fortes no Kaggle: `dewpoint_mean_c_mm4`,
`pressure_mean_mbar`, `rain_sum_mm_mm8`, `dewpoint_mean_c`, `temp_min_c`.
Praticamente a mesma lista da execução local.

Dois detalhes que só existem por causa das correções deste branch:

- **O ponto de orvalho lidera.** Antes da US-000 essa variável estava rotulada
  como `temp_max_c` por causa do erro de mapeamento posicional das colunas do
  INMET. Qualquer análise anterior que a mencionasse estava falando de outra
  coisa.
- **`dewpoint_mean_c_lag_12` aparece na 30ª posição** de 178. É uma feature de
  12 semanas, que **não existia** antes da US-003 — os lags climáticos
  paravam em 8. O modelo antigo não tinha como enxergar essa janela.

O topo absoluto continua epidemiológico: `notificacoes`,
`notificacoes_media_movel_3` e `notificacoes_lag_1` sozinhas dominam. Isso é
esperado num alvo com autocorrelação alta, e é a mesma razão de a persistência
ser um baseline tão forte.

### 7. As features `label_*` não são vazamento

`label_alerta_q90`, `label_alerta_q75`, `label_semana_critica` e
`label_surto_local` aparecem entre as 20 mais importantes, o que levanta
suspeita legítima. Não é o caso:

| Feature | Correlação com `notificacoes` (t) | Correlação com o alvo (t+4) |
|---|---|---|
| `label_alerta_q75` | +0,825 | +0,457 |
| `label_semana_critica` | +0,707 | +0,403 |
| `label_alerta_q90` | +0,646 | +0,367 |
| `label_surto_local` | +0,428 | +0,267 |

Todas correlacionam muito mais com a semana **atual** do que com o alvo. Essa
é a assinatura de uma feature derivada de `t`, que está disponível no momento
da previsão. Se viessem do t+4, a ordem seria inversa — foi assim que
`risco_surto_t4_com_vazamento` foi detectada, com Spearman de 0,863 contra o
alvo.

### 8. O early stopping está funcionando

435 e 343 árvores de um teto de 2.000. O treino parou sozinho quando a
validação deixou de melhorar; o teto não está apertando nada. Se algum dia um
braço reportar 2.000 árvores, aí sim `n_estimators` precisa subir.

---

## O que não fazer com estes números

**Não compare com a versão anterior do trabalho.** O teste antigo era 2024+
com 947 mil linhas nacionais; este é 2023 com 72 mil no recorte. MAE 7,466 lá
e 7,919 aqui não dizem qual modelo é melhor — dizem que 2023 e 2024 tiveram
epidemias de tamanhos diferentes. O que se compara são as conclusões de cada
ablação, em [`docs/COMPARACAO_ANTES_DEPOIS.md`](../../docs/COMPARACAO_ANTES_DEPOIS.md).

**Não reporte só o R²_log.** 0,7065 parece muito melhor que 0,3925 e mede
outra coisa: o quanto o modelo explica da variação de `log(1+casos)`, não dos
casos. Os dois juntos são o resultado; separados, o primeiro é otimista e o
segundo esconde o que o modelo aprendeu.

**Não use estes hiperparâmetros para outro experimento.** Eles vieram do
braço `e1_sinan_inmet`. Para mesorregião, UF ou E2, rode o Optuna de novo — o
ótimo depende do tamanho da amostra e do conjunto de features, que é
exatamente o que o item 2 acima demonstra.

# Como rodar o retreino no Kaggle

Guia operacional: o que precisa acontecer antes, o que subir para o
HuggingFace, e como rodar no Kaggle. Para entender **o quê** cada número
significa, veja o [Guia técnico](GUIA_TECNICO_RETREINO.md).

---

## Resposta rápida às suas perguntas

**Preciso subir o dataset para o HuggingFace?** Sim, obrigatoriamente. Os
arquivos em `data/model_ready/` foram **inteiramente reconstruídos** — não é
uma atualização incremental. Os que estão hoje no HF vêm do pipeline com os
cinco defeitos corrigidos neste branch, e usá-los reproduziria os resultados
antigos.

**Preciso rodar tratamento de dados antes?** Só se você não tiver os
`.parquet` prontos localmente. Se `data/model_ready/train.parquet` existe e é
mais novo que os scripts, pode ir direto para o passo 2.

**Quais estados entraram?** Seis. A tabela está na
[seção 1](#1-quais-estados-entraram-e-por-quê).

---

## 1. Quais estados entraram, e por quê

O recorte usa **6 UFs**: **DF, ES, GO, MG, RJ, SP**.

O critério foi cobertura climática do INMET — o clima só existe onde há
estação automática funcionando. A cobertura foi **medida depois** de corrigir
os zeros falsos de chuva (US-001); antes da correção a chuva "nunca faltava"
e a cobertura aparente era muito maior que a real.

### Aprovados

| UF | Cobertura climática¹ | Municípios no recorte | Linhas | Notificações | Motivo |
|---|---|---|---|---|---|
| DF | 99,62% | 1 | 262 | 239.002 | Cobertura quase perfeita |
| RJ | 95,10% | 83 | 21.746 | 122.369 | Cobertura alta |
| MG | 95,01% | 745 | 195.190 | 1.010.365 | Maior volume do recorte |
| GO | 94,48% | 173 | 45.326 | 541.627 | Cobertura alta |
| ES | 92,92% | 54 | 14.148 | 219.065 | Cobertura alta |
| SP | 92,37% | 313 | 82.006 | 1.177.273 | Entrou por peso epidemiológico² |
| **TOTAL** | **94,26%** | **1.369** | **358.678** | **3.309.701** | |

¹ Percentual de semanas-município com as três variáveis climáticas exigidas
(`rain_sum_mm`, `temp_mean_c`, `humidity_mean_pct`) simultaneamente presentes,
já dentro do recorte e após o filtro municipal de 80%.

² SP tem 78,77% de cobertura antes do filtro municipal, abaixo até de SC, que
foi reprovado. Entrou por concentrar o maior número de notificações da janela:
excluí-lo tiraria o estado com mais dengue do país de um trabalho sobre dengue.
Depois do filtro de 80%, os municípios de SP que sobraram têm 92,37%.

### Reprovados e o motivo

Cobertura medida **após** as correções, em `data/integrated/coverage_by_uf.csv`.
`pct_completo` é o percentual de semanas-município com as três variáveis
climáticas presentes ao mesmo tempo, antes de qualquer filtro:

| UF | pct_completo | Por que ficou de fora |
|---|---|---|
| SC | 80,64% | Cobertura suficiente, mas **SINAN fraco**: só 13,3% das semanas têm caso e 20,5% dos municípios têm série usável¹ |
| PB | 77,20% | SINAN fraco (20,2% das semanas com caso) e cobertura abaixo de SP |
| PR | 76,81% | Cobertura no limite |
| TO | 76,06% | SINAN fraco (22,5% das semanas com caso) |
| BA | 72,49% | Cobertura baixa + confirmação laboratorial de 36,4% |
| PE | 66,57% | Cobertura baixa + confirmação laboratorial de 28,0% |
| PA, CE, SE, RN, PI | 58-63% | Cobertura climática insuficiente |
| MA, MS, RO, AL | 52-56% | Cobertura climática insuficiente |
| AP, AM, RS, AC, MT | 42-50% | Cobertura climática insuficiente |
| RR | 29,49% | Pior cobertura do país |

¹ Os indicadores de qualidade do SINAN (semanas com caso, municípios com série
usável, confirmação laboratorial) vêm da auditoria registrada no
[PRD](../tasks/prd-retreino-recorte-2019-2023.md); a cobertura climática vem da
medição pós-correção.

**Nota sobre SP:** com 78,77% de `pct_completo`, SP fica **abaixo de SC**
(80,64%). Entrou mesmo assim porque concentra o maior volume de notificações
da janela e SC quase não tem dengue — o critério não foi só cobertura, e é
honesto dizer isso. Após o filtro municipal de 80%, os municípios de SP que
sobraram têm 92,37% de cobertura.

**Em resumo:** 21 das 27 UFs ficaram de fora, a maioria por falta de estação
meteorológica. Esse é o custo declarado do recorte, e é a razão de existir o
braço E2, que roda sem clima sobre o Brasil inteiro.

### Filtro adicional dentro das 6 UFs

Nem todo município das 6 UFs entrou. Exige-se **≥ 80% das semanas do recorte
com as três variáveis climáticas presentes**. Isso derrubou 546 dos 1.915
municípios candidatos — restaram 1.369.

---

## 2. Passos anteriores (só se precisar reconstruir os dados)

Se `data/model_ready/*.parquet` já existe e está atualizado, **pule para o
passo 3**.

Todos os comandos usam `scripts/rodar_com_teto.sh`, que roda dentro de um
cgroup com limite de memória. Sem ele, um script que exagera derruba a VM
inteira do WSL em vez de morrer sozinho.

```bash
# 2.1 — INMET: extrair, agregar por semana, features municipais
#       Requer os ZIPs anuais do INMET em ~/Downloads ou /mnt/c/Users/<você>/Downloads
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_extract_standardize.py --years 2018-2024
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_weekly_aggregate.py   --years 2018-2024
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/inmet_municipal_features.py --years 2018-2024

# 2.2 — Integrar SINAN + INMET
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/integrate_sinan_inmet.py

# 2.3 — Datasets prontos para modelo (os quatro)
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/prepare_model_dataset.py --modo e1
scripts/rodar_com_teto.sh -m 9G -- python3 scripts/prepare_model_dataset.py --modo e2
scripts/rodar_com_teto.sh -m 7G -- python3 scripts/prepare_mesoregion_dataset.py --nivel mesorregiao
scripts/rodar_com_teto.sh -m 7G -- python3 scripts/prepare_mesoregion_dataset.py --nivel uf
```

**Tempo total:** ~40 minutos, dominado pela extração do INMET (35 milhões de
registros horários).

**Para mudar o recorte** — outras UFs, outros anos, outro limiar de cobertura
— edite [`config/recorte.json`](../config/recorte.json) e rode de novo a partir
do 2.3. Nenhum script tem UF ou ano escrito no código.

### Verificação antes de subir

```bash
python3 -c "
import json
for d in ['', '/e2', '/mesorregiao', '/uf']:
    r = json.load(open(f'data/model_ready{d}/resumo_dataset.json'))
    print(d or '/e1', r['linhas'], r['n_features'], 'features')
"
```

Esperado:

```
/e1            {'train': 214933, 'val': 71188, 'test': 72557}   178 features
/e2            {'train': 6110370, 'val': 578760, 'test': 294945} 129 features
/mesorregiao   {'train': 6751, 'val': 2236, 'test': 2279}        163 features
/uf            {'train': 942, 'val': 312, 'test': 318}           163 features
```

---

## 3. Subir para o HuggingFace

O Kaggle não lê o seu disco. Os dados vão pelo dataset
`pedrolucassantanaf/dengue-tcc2-data`.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx        # token com permissão de escrita
python3 scripts/upload_data_to_hf.py
```

**O que sobe:** ~172 MB de `data/model_ready/`, mais as camadas SINAN e INMET.

| Diretório | Arquivos | Tamanho |
|---|---|---|
| `model_ready/` (E1) | 7 | 36 MB |
| `model_ready/e2/` | 6 | 129 MB |
| `model_ready/mesorregiao/` | 6 | 6 MB |
| `model_ready/uf/` | 6 | 1 MB |

> **Atenção — corrigido neste branch.** O script usava o padrão `"*"` para
> `model_ready`, que pega só arquivos da raiz e **ignora silenciosamente os
> subdiretórios**. Como `e2/`, `mesorregiao/` e `uf/` são novos, eles não
> subiriam e o notebook baixaria só o E1, sem erro nenhum. O padrão agora é
> `"**/*"`. Se você usa uma cópia antiga do script, corrija antes de subir.

Cada arquivo `.parquet` vem acompanhado de três metadados que o notebook
precisa:

- `feature_schema.csv` — quais features são `sinan` e quais são `inmet`. É o
  que separa os braços da ablação.
- `risk_thresholds.csv` — os limiares de risco **derivados só do treino**. Se
  você recalcular no notebook, reintroduz o vazamento que a US-002 removeu.
- `resumo_dataset.json` — volumetria, para conferir que baixou o certo.

---

## 4. Rodar no Kaggle

### 4.1 Configuração do notebook

1. **Settings → Accelerator:** `None` (CPU). O XGBoost com `tree_method="hist"`
   não usa GPU aqui, e a CPU do Kaggle tem mais RAM disponível.
2. **Settings → Internet:** ligado (necessário para o `hf_hub_download`).
3. **Add-ons → Secrets:** crie `HF_TOKEN` com o seu token do HuggingFace.

### 4.2 Célula de setup

```python
!pip install -q xgboost optuna shap huggingface_hub

from kaggle_secrets import UserSecretsClient
import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO  = "pedrolucassantanaf/dengue-tcc2-data"
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

# Escolha o experimento:
#   ""              E1  — recorte 6 UFs, com e sem clima
#   "e2/"           E2  — histórico completo, SINAN-only
#   "mesorregiao/"  US-009 — agregado por mesorregião
#   "uf/"           US-009 — agregado por UF
EXPERIMENTO = ""

def baixar(nome, ext="parquet"):
    caminho = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"data/model_ready/{EXPERIMENTO}{nome}.{ext}",
        repo_type="dataset", token=HF_TOKEN,
    )
    return pd.read_parquet(caminho) if ext == "parquet" else pd.read_csv(caminho)

train, val, test = baixar("train"), baixar("val"), baixar("test")
schema   = baixar("feature_schema", "csv")
limiares = baixar("risk_thresholds", "csv")

print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
```

### 4.3 Separar os braços da ablação

**Não** derive as features por nome de coluna. Use o `feature_schema.csv`, que
é a mesma fonte que o pipeline local usa:

```python
feats_sinan = schema.loc[schema.origem == "sinan", "feature"].tolist()
feats_todas = schema["feature"].tolist()          # sinan + inmet

# Colunas que NUNCA podem virar feature:
#   notificacoes_t4                 é o alvo
#   risco_surto_t4                  é o alvo de classificação
#   risco_surto_t4_com_vazamento    existe só para medir a inflação do
#                                   vazamento; usar como feature ou como alvo
#                                   de produção reintroduz o problema da US-002
#   threshold_source                é auditoria de qual limiar foi aplicado
```

### 4.4 Treinar

```python
import numpy as np, xgboost as xgb

FIXOS = dict(tree_method="hist", random_state=42, n_jobs=-1,
             n_estimators=2000, early_stopping_rounds=50)

def treinar(feats, params):
    m = xgb.XGBRegressor(**params, **FIXOS)
    m.fit(train[feats], np.log1p(train["notificacoes_t4"]),
          eval_set=[(val[feats], np.log1p(val["notificacoes_t4"]))],
          verbose=False)
    return m
```

O alvo é treinado em **`log1p`** e as previsões voltam com **`np.expm1`**.
Reportar R² sem desfazer a transformação dá um número diferente e mais
otimista — é o erro que o relatório anterior cometia. Veja a
[seção 3 do Guia técnico](GUIA_TECNICO_RETREINO.md#3-as-métricas-explicadas).

### 4.5 Baselines — obrigatórios

Nenhum resultado de XGBoost deve ser reportado sem eles ao lado:

```python
serie = pd.concat([train, val, test]).sort_values(
    ["ibge_municipio", "ano", "semana_epidemiologica"])

# Persistência: repete o valor de hoje quatro semanas à frente
pred_persistencia = test["notificacoes"]

# Sazonal: o alvo da mesma semana no ano anterior
anterior = serie[["ibge_municipio","ano","semana_epidemiologica","notificacoes_t4"]].copy()
anterior["ano"] += 1
pred_sazonal = test.merge(
    anterior.rename(columns={"notificacoes_t4": "pred"}),
    on=["ibge_municipio","ano","semana_epidemiologica"], how="left")["pred"]
```

Ou, mais simples: baixe `models/baselines*/metrics.json`, já calculado.

### 4.6 Limites do Kaggle

| Experimento | Cabe no Kaggle? | Observação |
|---|---|---|
| E1 (município) | Sim | ~5 min por braço com 30 trials |
| Mesorregião | Sim | ~2 min por braço |
| UF | Sim | segundos |
| **E2** | **Cuidado** | 6,1 milhões de linhas, pico de **9 GB** e ~2 h com 20 trials. O Kaggle dá 30 GB de RAM mas corta a sessão em 12 h — cabe, mas rode o Optuna com poucos trials ou salve o estudo em disco |

---

## 5. Armadilha: os notebooks atuais filtram só o Distrito Federal

Os notebooks em `colab/` (`1_dados_df`, `2_treino_df`, `3_resultados_df`)
contêm:

```python
DF_CODE = "5300108"
df_train = df_train[df_train["ibge_municipio"] == DF_CODE]
```

**Isso não é viável com o recorte novo.** O DF tem 1 município, e no recorte
2019-2023 sobram:

| Split | Linhas do DF |
|---|---|
| train | **157** |
| val | 52 |
| test | 53 |

157 linhas de treino para 178 features é overfitting garantido. Duas saídas:

- **Remova o filtro** e treine nas 6 UFs (214.933 linhas de treino). É o que o
  pipeline local faz, e é o que as tabelas do TCC reportam.
- **Se o TCC precisa ser sobre o DF especificamente**, use o `EXPERIMENTO =
  "e2/"`, que cobre 2000-2023 e dá ao DF ~1.250 linhas de treino em vez de
  157 — mas sem clima, porque o E2 é SINAN-only.

---

## 6. O que não repetir dos notebooks antigos

| Prática antiga | Por quê está errada |
|---|---|
| Recalcular os percentis de risco sobre o dataset inteiro | Vazamento temporal: infla F1_macro em até 0,11. Use `risk_thresholds.csv` |
| `TUNE_SAMPLE = 500_000` no Optuna | Hiperparâmetros ótimos dependem do tamanho da amostra; tunar em 500 mil e treinar em milhões entrega parâmetros de outro problema |
| `SAMPLE_SIZE = 50000` no SHAP | Explica um modelo que não é o entregue |
| Reportar `R2` sem dizer a escala | R²_log e R²_orig são números diferentes. Publique os dois |
| Usar `risco_surto_t4_com_vazamento` | Existe só para medir a inflação, nunca como alvo |
| Comparar XGBoost só com XGBoost | Sem baseline não dá para saber se o modelo aprendeu algo |

---

## 7. Referência rápida dos arquivos

| Arquivo no HF | Para quê |
|---|---|
| `data/model_ready/{train,val,test}.parquet` | E1 — recorte 6 UFs 2019-2023 |
| `data/model_ready/feature_schema.csv` | Separa features `sinan` de `inmet` |
| `data/model_ready/risk_thresholds.csv` | Limiares de risco sem vazamento |
| `data/model_ready/resumo_dataset.json` | Volumetria, para conferência |
| `data/model_ready/e2/*` | E2 — histórico completo, SINAN-only |
| `data/model_ready/mesorregiao/*` | US-009 — 43 mesorregiões |
| `data/model_ready/uf/*` | US-009 — 6 UFs |

---

## 8. Análise do notebook atual — três bloqueadores

Auditoria do notebook `Treino XGBoost — Dengue TCC2` contra o dataset novo.
Um notebook corrigido está em
[`kaggle/notebook_retreino.py`](../kaggle/notebook_retreino.py); cole-o no
Kaggle no lugar do atual.

### 8.1 O dataset no HuggingFace ainda é o antigo

`thiagorfreitas/dengue-tcc2-data`, pasta `data/model_ready/`:

| | No HF hoje | Depois do retreino |
|---|---|---|
| `train.parquet` | 249 MB (3 meses atrás) | 23 MB |
| `val.parquet` | 51,7 MB | 6,2 MB |
| `test.parquet` | 66,4 MB | 7,2 MB |
| `feature_schema.csv` | **ausente** | presente |
| `risk_thresholds.csv` | **ausente** | presente |
| `resumo_dataset.json` | **ausente** | presente |
| `e2/`, `mesorregiao/`, `uf/` | **ausentes** | presentes |

O arquivo grande é o dataset nacional 2000-2026 anterior ao recorte. Rodar o
notebook contra ele reproduz os resultados antigos, com os cinco defeitos
dentro.

O script de upload aponta para outro repositório. Ele passou a aceitar
variável de ambiente:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export HF_REPO_ID=thiagorfreitas/dengue-tcc2-data
python3 scripts/upload_data_to_hf.py
```

### 8.2 Vazamento do alvo — o mais grave

O notebook monta as features assim:

```python
META_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
DROP_COLS = [c for c in (META_COLS + [CLASS_TARGET]) if c in df_train.columns]
X = df.drop(columns=DROP_COLS + [TARGET], errors="ignore")
```

Isso remove cinco colunas e **deixa passar duas** que o dataset novo
introduziu:

| Coluna que sobra | O que é | Efeito |
|---|---|---|
| `risco_surto_t4_com_vazamento` | O próprio alvo binado em quatro faixas. Spearman de **0,863** com `notificacoes_t4` | O modelo recebe a resposta |
| `threshold_source` | Texto (`municipio` / `uf_fallback`) | `ValueError` no `fit` |

Distribuição real do alvo dentro de cada faixa dessa coluna, no teste:

| `risco_surto_t4_com_vazamento` | Linhas | Média de `notificacoes_t4` |
|---|---|---|
| 0 | 45.017 | 1,0 |
| 1 | 7.456 | 10,6 |
| 2 | 9.036 | 19,9 |
| 3 | 11.048 | 61,3 |

Essa coluna existe **só** para medir de quanto era a inflação do vazamento
temporal (US-002). Ela nunca é feature nem alvo de produção.

> **O que salva o notebook de publicar um resultado falso é um acidente:**
> `threshold_source` é texto, então o XGBoost lança `ValueError` antes de
> treinar. Se você "resolver" isso com um `select_dtypes(include=np.number)`,
> a coluna de texto some, o vazamento fica, e o modelo passa a entregar
> métricas excelentes e sem sentido. Use uma lista de exclusão explícita.

A correção:

```python
NUNCA_FEATURE = [
    "notificacoes_t4",               # o alvo
    "risco_surto_t4",                # alvo da classificação
    "risco_surto_t4_com_vazamento",  # o alvo binado
    "threshold_source",              # texto de auditoria
    "ibge_municipio", "ano", "semana_epidemiologica",
]
```

### 8.3 O walk-forward não roda, e falha em silêncio

A última célula tem três defeitos:

| Linha | Problema |
|---|---|
| `xgb.XGBRegressor(**study.best_params, ...)` | `study` nunca é definido — a célula do Optuna foi substituída por `KNOWN_BEST`. `NameError` |
| `TRAIN_MIN_ANOS = 5` | O recorte tem **4 anos** em train+val (2019-2022). Nenhum fold é gerado, `cv_df` sai vazio e a média vira `NaN` — **sem erro nenhum** |
| `m.fit(..., early_stopping_rounds=20)` | Removido do `fit()` no XGBoost 2.0+; é parâmetro do construtor. `TypeError` |

No notebook corrigido o mínimo de anos vem do dado
(`max(1, min(3, len(anos) - 1))`) em vez de ser fixo.

### 8.4 Problemas menores, mas que mudam os números

| Ponto | Por quê |
|---|---|
| Treina em contagem crua, sem `log1p` | Um município com 3.000 casos numa semana domina o treino. O pipeline local treina em `log1p` e volta com `expm1` — os números do notebook não são comparáveis às tabelas do TCC |
| `KNOWN_BEST` com MAE 2,795 | Veio do dataset antigo, com outro conjunto de features. Como referência não vale mais. Os hiperparâmetros reais do braço `e1_sinan_inmet` estão no notebook corrigido |
| Nenhum baseline | Sem a linha de persistência não há como afirmar que o modelo aprendeu algo. No recorte a persistência dá R²_orig 0,400 contra 0,390 do melhor XGBoost |
| Não usa `feature_schema.csv` | Sem ele não dá para separar `sinan` de `sinan+inmet`, que é a pergunta central do trabalho |
| `load_dataset(...).to_pandas()` | Converte via Arrow mantendo as duas cópias vivas. No E2, com 6,1 milhões de linhas, isso dobra a memória. `hf_hub_download` + `read_parquet` é mais enxuto |
| `device="cuda"` fixo | Válido no XGBoost 2.0+, e bom para o E1. Mas a sessão de GPU do Kaggle tem menos RAM que a de CPU, e o E2 teve pico medido de 9 GB — para ele, prefira CPU |

### 8.5 Ordem recomendada

1. Rodar a [etapa 1](#2-passos-anteriores-só-se-precisar-reconstruir-os-dados) local, se ainda não rodou
2. Subir com `HF_REPO_ID=thiagorfreitas/dengue-tcc2-data`
3. Conferir no HF que `feature_schema.csv` e as pastas `e2/`, `mesorregiao/` e `uf/` apareceram
4. Substituir o notebook por `kaggle/notebook_retreino.py`
5. Rodar com `EXPERIMENTO = ""` primeiro; a célula de checagem de vazamento
   deve imprimir `Checagem de vazamento: OK`
6. Conferir que a persistência aparece na tabela antes de qualquer número de
   XGBoost

**Referência para saber se deu certo** — braço `e1_sinan_inmet`, teste 2023:
MAE 7,935 · R²_orig 0,390 · R²_log 0,707. Diferenças pequenas são esperadas
(GPU vs CPU muda a ordem de soma em ponto flutuante). Um MAE muito **menor**
que isso é sinal de vazamento, não de sucesso.

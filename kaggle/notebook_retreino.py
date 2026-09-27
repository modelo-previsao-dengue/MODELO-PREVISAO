# %% [markdown]
# # Retreino XGBoost — Dengue TCC2
#
# Substitui o notebook anterior, que com o dataset novo **quebra ou vaza o
# alvo**. Tres correcoes principais:
#
# 1. A lista de colunas a descartar passa a ser explicita. O `DROP_COLS`
#    anterior nao removia `risco_surto_t4_com_vazamento`, que tem correlacao
#    de Spearman de 0,86 com o alvo por ser o proprio alvo binado, nem
#    `threshold_source`, que e texto e faz o XGBoost lancar erro no fit.
# 2. O alvo e treinado em `log1p` e as previsoes voltam com `expm1`. Sem isso
#    o R2 sai numa escala e e lido como se fosse de outra.
# 3. Os baselines ingenuos entram na mesma tabela. No recorte municipal a
#    persistencia entrega R2 de 0,400 e o melhor XGBoost 0,390 — sem essa
#    linha nao da para afirmar que o modelo aprendeu algo.
#
# Ambiente: Kaggle Notebooks. Ver `docs/COMO_RODAR_NO_KAGGLE.md`.

# %% [code]
!pip install -q -U "shap>=0.47.1" optuna huggingface-hub

# %% [code]
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

REPO_ID = "thiagorfreitas/dengue-tcc2-data"
OUT_DIR = Path("/kaggle/working")

# "" = E1 (recorte 6 UFs) | "e2/" | "mesorregiao/" | "uf/"
EXPERIMENTO = ""

def escolher_device():
    """Detecta a GPU em vez de assumir que ela existe.

    device="cuda" com o acelerador desligado faz o XGBoost 2.0+ abortar, e
    esquecer de liga-lo em Settings e o jeito mais facil de um Run All morrer
    na primeira celula de treino. O E2 fica em CPU de proposito: sao 6,1
    milhoes de linhas com pico medido de 9 GB, e a sessao de GPU do Kaggle
    tem menos RAM que a de CPU.
    """
    if EXPERIMENTO == "e2/":
        return "cpu"
    try:
        import subprocess
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return "cuda"
    except Exception:
        print("  GPU nao encontrada (Settings > Accelerator). Seguindo em CPU.")
        return "cpu"


DEVICE = escolher_device()

print("Token HF:", bool(HF_TOKEN), "| experimento:", EXPERIMENTO or "e1",
      "| device:", DEVICE)

# %% [code]
# hf_hub_download + read_parquet em vez de load_dataset: o datasets converte
# via Arrow e mantem as duas copias vivas, o que dobra a memoria no E2.
def baixar(nome, ext="parquet"):
    caminho = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"data/model_ready/{EXPERIMENTO}{nome}.{ext}",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    return pd.read_parquet(caminho) if ext == "parquet" else pd.read_csv(caminho)


df_train = baixar("train")
df_val = baixar("val")
df_test = baixar("test")
schema = baixar("feature_schema", "csv")

print(f"train {df_train.shape} | val {df_val.shape} | test {df_test.shape}")
print(f"anos: train {sorted(df_train.ano.unique())} | "
      f"val {sorted(df_val.ano.unique())} | test {sorted(df_test.ano.unique())}")

# %% [code]
TARGET = "notificacoes_t4"

# Deny-list explicita. Cada uma destas colunas quebra o modelo de um jeito
# diferente se entrar como feature.
NUNCA_FEATURE = [
    "notificacoes_t4",               # o alvo
    "risco_surto_t4",                # alvo da classificacao
    "risco_surto_t4_com_vazamento",  # o alvo binado: Spearman 0,86 com ele
    "threshold_source",              # texto; o XGBoost lanca ValueError
    "ibge_municipio", "ano", "semana_epidemiologica",  # identificadores
]

# Os bracos da ablacao saem do feature_schema.csv, nao de nome de coluna.
FEATS = {"sinan": schema.loc[schema.origem == "sinan", "feature"].tolist()}
# No E2 nao existe feature climatica, entao um segundo braco seria o mesmo
# modelo treinado duas vezes e um "ganho do clima" de zero que nao significa
# nada. So cria o braco com clima quando ha clima.
if (schema.origem == "inmet").any():
    FEATS["sinan_inmet"] = schema["feature"].tolist()
for nome, fs in FEATS.items():
    FEATS[nome] = [f for f in fs if f not in NUNCA_FEATURE and f in df_train.columns]
    print(f"{nome}: {len(FEATS[nome])} features")

# Conferencia de seguranca: nenhuma coluna de texto pode ter sobrado, e
# nenhuma coluna proibida pode estar na lista.
for nome, fs in FEATS.items():
    obj = [c for c in fs if df_train[c].dtype == object]
    proibida = [c for c in fs if c in NUNCA_FEATURE]
    assert not obj, f"{nome}: colunas de texto sobraram: {obj}"
    assert not proibida, f"{nome}: coluna proibida na lista: {proibida}"
print("Checagem de vazamento: OK")

# %% [code]
def metricas(y_true, y_pred, nome):
    """Sempre nas duas escalas: o treino otimiza log, o TCC reporta contagem."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    return {
        "modelo": nome,
        "n": len(y_true),
        "MAE_orig": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE_orig": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2_orig": round(r2_score(y_true, y_pred), 4),
        "MAE_log": round(mean_absolute_error(lt, lp), 4),
        "R2_log": round(r2_score(lt, lp), 4),
    }

# %% [code]
# BASELINES — rodam primeiro, de proposito. Nenhum numero de XGBoost deve ser
# lido sem eles ao lado.
serie = pd.concat([df_train, df_val, df_test], ignore_index=True).sort_values(
    ["ibge_municipio", "ano", "semana_epidemiologica"])

anterior = serie[["ibge_municipio", "ano", "semana_epidemiologica", TARGET]].copy()
anterior["ano"] += 1
sazonal = df_test.merge(
    anterior.rename(columns={TARGET: "_sazonal"}),
    on=["ibge_municipio", "ano", "semana_epidemiologica"], how="left")["_sazonal"]

linhas = [
    metricas(df_test[TARGET], df_test["notificacoes"], "baseline: persistencia"),
]
ok = sazonal.notna()
linhas.append(metricas(df_test.loc[ok.values, TARGET], sazonal[ok],
                       "baseline: sazonal"))

for r in linhas:
    print(f"{r['modelo']:26s} MAE {r['MAE_orig']:9.3f} | R2_orig {r['R2_orig']:7.4f} "
          f"| R2_log {r['R2_log']:.4f}")

# %% [code]
FIXOS = dict(tree_method="hist", device=DEVICE, random_state=42,
             n_estimators=2000, early_stopping_rounds=50)

# Hiperparametros do braco e1_sinan_inmet, obtidos localmente com 30 trials de
# Optuna sobre o TREINO INTEIRO. Se voce mudar de EXPERIMENTO, rode o Optuna de
# novo: o otimo depende do tamanho da amostra e do conjunto de features.
BEST_PARAMS = {
    "max_depth": 11,
    "learning_rate": 0.015594054205343648,
    "subsample": 0.5607784504928788,
    "colsample_bytree": 0.36459603466129137,
    "min_child_weight": 93,
    "reg_alpha": 2.0511456764725984,
    "reg_lambda": 0.017135281765880774,
    "gamma": 0.5788413262117456,
}


def treinar(feats, params=None):
    """Alvo em log1p. A previsao volta para contagem com expm1."""
    p = dict(params or BEST_PARAMS)
    m = xgb.XGBRegressor(**p, **FIXOS)
    m.fit(df_train[feats], np.log1p(df_train[TARGET]),
          eval_set=[(df_val[feats], np.log1p(df_val[TARGET]))], verbose=False)
    return m


modelos = {}
for nome, feats in FEATS.items():
    if not feats:
        continue
    m = treinar(feats)
    pred = np.expm1(m.predict(df_test[feats]))
    r = metricas(df_test[TARGET], pred, f"XGBoost: {nome}")
    r["arvores"] = int(m.best_iteration + 1)
    linhas.append(r)
    modelos[nome] = m
    print(f"{r['modelo']:26s} MAE {r['MAE_orig']:9.3f} | R2_orig {r['R2_orig']:7.4f} "
          f"| R2_log {r['R2_log']:.4f} | {r['arvores']} arvores")

# %% [code]
tabela = pd.DataFrame(linhas)
tabela.to_csv(OUT_DIR / "resultados.csv", index=False)
print(tabela.to_string(index=False))

# A leitura que importa: o clima ajudou?
if "XGBoost: sinan" in tabela.modelo.values and "XGBoost: sinan_inmet" in tabela.modelo.values:
    sem = tabela.loc[tabela.modelo == "XGBoost: sinan", "R2_orig"].iloc[0]
    com = tabela.loc[tabela.modelo == "XGBoost: sinan_inmet", "R2_orig"].iloc[0]
    print(f"\nGanho do clima em R2_orig: {com - sem:+.4f}")

# %% [code]
# Optuna — rode so se quiser retunar. Sem subamostragem: tunar numa fracao e
# treinar no todo entrega hiperparametros escolhidos para outro problema.
RODAR_OPTUNA = False
N_TRIALS = 30

if RODAR_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feats = FEATS["sinan_inmet"]
    Xtr, ytr = df_train[feats], np.log1p(df_train[TARGET])
    Xva, yva = df_val[feats], np.log1p(df_val[TARGET])

    def objetivo(trial):
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        m = xgb.XGBRegressor(**p, **FIXOS)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return float(np.sqrt(np.mean((yva - m.predict(Xva)) ** 2)))

    estudo = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=N_TRIALS,
                    callbacks=[lambda s, t: print(
                        f"trial {t.number + 1}/{N_TRIALS}: {t.value:.4f} "
                        f"| melhor {s.best_value:.4f}")])
    BEST_PARAMS = estudo.best_params
    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(BEST_PARAMS, f, indent=2)
    print(BEST_PARAMS)

# %% [code]
# Walk-forward. O numero minimo de anos de treino vem do dado, nao fixo em 5:
# o recorte E1 tem apenas 2019-2022 em train+val, e um minimo de 5 produziria
# zero folds e uma media NaN em silencio.
feats = FEATS["sinan_inmet"]
df_full = pd.concat([df_train, df_val], ignore_index=True)
anos = sorted(df_full["ano"].unique())
MIN_ANOS_TREINO = max(1, min(3, len(anos) - 1))

print(f"Anos disponiveis: {anos} | minimo de treino: {MIN_ANOS_TREINO}")

folds = []
for i, ano_val in enumerate(anos):
    if i < MIN_ANOS_TREINO:
        continue
    tr = df_full[df_full["ano"] < ano_val]
    va = df_full[df_full["ano"] == ano_val]
    if len(va) == 0:
        continue
    m = xgb.XGBRegressor(**BEST_PARAMS, **FIXOS)
    m.fit(tr[feats], np.log1p(tr[TARGET]),
          eval_set=[(va[feats], np.log1p(va[TARGET]))], verbose=False)
    pred = np.expm1(m.predict(va[feats]))
    r = metricas(va[TARGET], pred, f"fold {ano_val}")
    folds.append(r)
    print(f"  {ano_val}: MAE {r['MAE_orig']:.3f} | R2_orig {r['R2_orig']:.4f} "
          f"| R2_log {r['R2_log']:.4f}")

if folds:
    cv = pd.DataFrame(folds)
    cv.to_csv(OUT_DIR / "walk_forward_cv.csv", index=False)
    print(f"\nMedia: MAE {cv.MAE_orig.mean():.3f} | R2_orig {cv.R2_orig.mean():.4f}")
else:
    print("\nNenhum fold: a janela tem anos demais de menos para walk-forward.")

# %% [code]
# Importancia por SHAP, sobre o teste inteiro. Sem amostragem: explicar 50 mil
# linhas sorteadas descreve um modelo que nao e o entregue.
import shap
import matplotlib.pyplot as plt

braco = "sinan_inmet" if "sinan_inmet" in modelos else "sinan"
modelo, feats = modelos[braco], FEATS[braco]
X = df_test[feats]
print(f"Explicando o braco {braco}: {len(X):,} linhas, {len(feats)} features")

valores = shap.TreeExplainer(modelo).shap_values(X)
media_abs = np.abs(valores).mean(axis=0)

origem = dict(zip(schema.feature, schema.origem))
ranking = pd.DataFrame({
    "feature": feats,
    "shap_medio_abs": media_abs,
    "origem": [origem.get(f, "sinan") for f in feats],
}).sort_values("shap_medio_abs", ascending=False)
ranking.to_csv(OUT_DIR / "shap_ranking.csv", index=False)

peso_clima = ranking.loc[ranking.origem == "inmet", "shap_medio_abs"].sum()
total = ranking.shap_medio_abs.sum()
if peso_clima:
    print(f"Peso relativo do clima: {peso_clima / total:.2%}")
print(ranking.head(15).to_string(index=False))

plt.figure(figsize=(11, 9))
shap.summary_plot(valores, X, show=False, max_display=25, plot_size=None)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [code]
modelo.save_model(OUT_DIR / "xgb_dengue.ubj")
with open(OUT_DIR / "procedencia.json", "w") as f:
    json.dump({
        "repo_hf": REPO_ID,
        "experimento": EXPERIMENTO or "e1",
        "n_features": {k: len(v) for k, v in FEATS.items()},
        "linhas": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "params": BEST_PARAMS,
        "xgboost": xgb.__version__,
    }, f, indent=2, ensure_ascii=False)
print("Artefatos em", OUT_DIR)

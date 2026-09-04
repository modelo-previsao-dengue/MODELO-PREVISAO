#!/usr/bin/env python3
"""US-008: retreino e ablacao limpa, com baseline na mesma tabela.

Substitui os quatro scripts anteriores (train_xgb_regression,
train_xgb_classification, tune_xgb_optuna, explain_shap), que dividiam a
mesma logica em copias divergentes e traziam tres problemas:

- Subamostravam. O Optuna tunava em 500 mil linhas e o SHAP explicava 50
  mil, entao os hiperparametros escolhidos e a importancia relatada nao
  eram os do modelo entregue.
- Reportavam um R2 sem escala, com o treino em log1p e a avaliacao ora em
  contagem ora nao.
- Nao conheciam as colunas novas (rotulo com vazamento, origem do limiar) e
  as teriam usado como feature.

Bracos:
    e1_sinan        recorte 2019-2023, so features do SINAN
    e1_sinan_inmet  recorte 2019-2023, SINAN + clima
    e2_sinan        historico completo, so SINAN

Os dois bracos do E1 rodam sobre exatamente as mesmas linhas, entao a
diferenca entre eles isola o efeito do clima. O E2 usa o mesmo ano de teste
para que a comparacao com o E1 nao misture efeito de recorte com efeito de
periodo.

Uso:
    .venv/bin/python3 scripts/train_ablation.py --arm e1_sinan_inmet
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics_common

BASE_DIR = Path(__file__).resolve().parent.parent

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_TARGET_LEAKY = "risco_surto_t4_com_vazamento"
NAO_FEATURE = ID_COLS + [TARGET, CLASS_TARGET, CLASS_TARGET_LEAKY,
                         "threshold_source"]

ARMS = {
    "e1_sinan":       {"dataset": "data/model_ready",    "origens": ["sinan"]},
    "e1_sinan_inmet": {"dataset": "data/model_ready",    "origens": ["sinan", "inmet"]},
    "e2_sinan":       {"dataset": "data/model_ready/e2", "origens": ["sinan"]},
    # US-009: o mesmo recorte agregado por mesorregiao. Sao 43 series em vez
    # de 1.369, entao o par de bracos testa se a granularidade municipal e
    # que estava enterrando o sinal do clima no ruido.
    "meso_sinan":       {"dataset": "data/model_ready/mesorregiao",
                         "origens": ["sinan"]},
    "meso_sinan_inmet": {"dataset": "data/model_ready/mesorregiao",
                         "origens": ["sinan", "inmet"]},
    # Terceira granularidade pedida pela US-009. Sao 6 series e 942 linhas de
    # treino: o resultado vale como ponto extremo da curva, nao como modelo.
    "uf_sinan":       {"dataset": "data/model_ready/uf", "origens": ["sinan"]},
    "uf_sinan_inmet": {"dataset": "data/model_ready/uf",
                       "origens": ["sinan", "inmet"]},
}

# Espaco de busca. Faixas amplas de proposito: com 178 features e alvo de
# cauda longa, o que costuma decidir e a regularizacao, nao a profundidade.
def espaco(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


FIXOS = {"tree_method": "hist", "random_state": 42, "n_jobs": -1,
         "n_estimators": 2000, "early_stopping_rounds": 50}


def fixos_para(n_linhas):
    """Ajusta so o paralelismo ao tamanho do dado, sem tocar no modelo.

    Com 6.751 linhas, como no braco de mesorregiao, espalhar cada arvore por
    16 threads gasta mais em sincronizacao do que em calculo: o braco chegou a
    queimar duas horas de CPU em dezessete minutos de relogio. O numero de
    arvores continua sendo decidido pelo early stopping, nao aqui.
    """
    fixos = dict(FIXOS)
    if n_linhas < 100_000:
        fixos["n_jobs"] = 4
    return fixos


def procedencia():
    """Tudo que a FR-5 exige para reproduzir uma execucao.

    O metrics.json anterior guardava os hiperparametros e mais nada, entao um
    numero na tabela do TCC nao dizia de que codigo nem de que versao de
    biblioteca tinha saido.
    """
    import platform
    import subprocess

    def versao(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    try:
        commit = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        sujo = bool(subprocess.run(
            ["git", "-C", str(BASE_DIR), "status", "--porcelain", "scripts"],
            capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, sujo = None, None

    return {
        "commit": commit,
        "scripts_com_alteracao_nao_commitada": sujo,
        "python": platform.python_version(),
        "xgboost": versao("xgboost"),
        "optuna": versao("optuna"),
        "sklearn": versao("sklearn"),
        "pandas": versao("pandas"),
        "numpy": versao("numpy"),
    }


def carregar(arm):
    """Le so as colunas do braco, para nao abrir 6 milhoes de linhas inteiras."""
    cfg = ARMS[arm]
    dataset = BASE_DIR / cfg["dataset"]
    schema = pd.read_csv(dataset / "feature_schema.csv")
    feats = schema.loc[schema["origem"].isin(cfg["origens"]), "feature"].tolist()

    partes = {}
    for nome in ("train", "val", "test"):
        caminho = dataset / f"{nome}.parquet"
        partes[nome] = {
            "X": _ler_matriz(caminho, feats),
            "meta": pd.read_parquet(caminho, columns=NAO_FEATURE),
        }
    gc.collect()
    return partes, feats, dataset


def _ler_matriz(caminho, feats):
    """Le as features direto para um float32 contiguo, sem DataFrame no meio.

    A API sklearn do XGBoost converte o DataFrame para numpy antes de montar
    a DMatrix, entao um read_parquet seguido de to_numpy mantem as duas
    copias vivas ao mesmo tempo. No braco E2, com 6,1 milhoes de linhas por
    129 colunas, sao 3,1 GB cada, e o processo era morto antes do primeiro
    trial. Preenchendo um array ja alocado, lote a lote, o pico e o array
    final mais um lote.
    """
    arquivo = pq.ParquetFile(caminho)
    X = np.empty((arquivo.metadata.num_rows, len(feats)), dtype=np.float32)
    i = 0
    for lote in arquivo.iter_batches(batch_size=200_000, columns=feats):
        for j in range(len(feats)):
            X[i:i + lote.num_rows, j] = lote.column(j).to_numpy(
                zero_copy_only=False)
        i += lote.num_rows
    return X


def tune_regressao(partes, feats, trials, semente=42):
    """Optuna sobre o conjunto inteiro de treino, sem subamostragem.

    A versao anterior tunava em 500 mil linhas e retreinava em 2 milhoes, o
    que entrega hiperparametros escolhidos para outro problema: profundidade
    e min_child_weight otimos dependem do tamanho da amostra.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    Xtr, ytr = partes["train"]["X"], np.log1p(partes["train"]["meta"][TARGET])
    Xva, yva = partes["val"]["X"], np.log1p(partes["val"]["meta"][TARGET])

    fixos = fixos_para(len(Xtr))

    def objetivo(trial):
        m = xgb.XGBRegressor(**espaco(trial), **fixos)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = m.predict(Xva)
        return float(np.sqrt(np.mean((yva - pred) ** 2)))

    def progresso(estudo, trial):
        # Sem isto a busca fica muda por dezenas de minutos e nao da para
        # distinguir "esta lento" de "travou".
        print(f"    trial {trial.number + 1}/{trials}: "
              f"RMSE_log {trial.value:.4f} | melhor {estudo.best_value:.4f} | "
              f"{trial.duration.total_seconds():.0f}s", flush=True)

    estudo = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=semente))
    estudo.optimize(objetivo, n_trials=trials, show_progress_bar=False,
                    callbacks=[progresso])
    print(f"  melhor RMSE_log na validacao: {estudo.best_value:.4f}")
    return estudo.best_params, estudo.best_value


def treinar_regressao(partes, feats, params):
    Xtr, ytr = partes["train"]["X"], np.log1p(partes["train"]["meta"][TARGET])
    Xva, yva = partes["val"]["X"], np.log1p(partes["val"]["meta"][TARGET])
    modelo = xgb.XGBRegressor(**params, **fixos_para(len(Xtr)))
    modelo.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return modelo


def treinar_classificacao(partes, feats, params, rotulo):
    Xtr = partes["train"]["X"]
    Xva = partes["val"]["X"]
    p = {k: v for k, v in params.items()}
    modelo = xgb.XGBClassifier(objective="multi:softprob", num_class=4,
                               eval_metric="mlogloss", **p,
                               **fixos_para(len(Xtr)))
    modelo.fit(Xtr, partes["train"]["meta"][rotulo],
               eval_set=[(Xva, partes["val"]["meta"][rotulo])], verbose=False)
    return modelo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    saida = BASE_DIR / "models" / "ablacao" / args.arm
    saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Braco: {args.arm}")
    partes, feats, dataset = carregar(args.arm)
    n = {k: len(v["meta"]) for k, v in partes.items()}
    print(f"  features: {len(feats)} | train {n['train']:,} | "
          f"val {n['val']:,} | test {n['test']:,}")

    print(f"\nOptuna, {args.trials} trials, treino inteiro (sem subamostragem)...")
    melhores, melhor_val = tune_regressao(partes, feats, args.trials)

    print("\nModelo final de regressao...")
    reg = treinar_regressao(partes, feats, melhores)
    pred = np.expm1(reg.predict(partes["test"]["X"]))
    m_reg = metrics_common.regression_metrics(
        partes["test"]["meta"][TARGET], pred)
    m_reg["melhor_rmse_log_val"] = round(melhor_val, 4)
    m_reg["n_arvores_usadas"] = int(reg.best_iteration + 1)
    print(f"  MAE  {m_reg['MAE_orig']:>10.3f} casos | log {m_reg['MAE_log']:.4f}")
    print(f"  R2   {m_reg['R2_orig']:>10.4f} orig  | log {m_reg['R2_log']:.4f}")

    print("\nClassificacao, rotulo sem vazamento...")
    clf = treinar_classificacao(partes, feats, melhores, CLASS_TARGET)
    proba = clf.predict_proba(partes["test"]["X"])
    m_clf = metrics_common.classification_metrics(
        partes["test"]["meta"][CLASS_TARGET], proba.argmax(axis=1), proba)
    print(f"  F1_macro {m_clf['F1_macro']:.4f} | AUC {m_clf.get('AUC_macro_ovr')}")

    print("Classificacao, rotulo com vazamento, so para medir a inflacao...")
    clf_l = treinar_classificacao(partes, feats, melhores, CLASS_TARGET_LEAKY)
    proba_l = clf_l.predict_proba(partes["test"]["X"])
    m_clf_l = metrics_common.classification_metrics(
        partes["test"]["meta"][CLASS_TARGET_LEAKY], proba_l.argmax(axis=1),
        proba_l, alvo=CLASS_TARGET_LEAKY)
    print(f"  F1_macro {m_clf_l['F1_macro']:.4f} | AUC {m_clf_l.get('AUC_macro_ovr')}")

    inflacao = {
        "F1_macro": round(m_clf_l["F1_macro"] - m_clf["F1_macro"], 4),
        "AUC_macro_ovr": (round(m_clf_l["AUC_macro_ovr"] - m_clf["AUC_macro_ovr"], 4)
                          if m_clf.get("AUC_macro_ovr") and m_clf_l.get("AUC_macro_ovr")
                          else None),
    }
    print(f"  inflacao pelo vazamento: {inflacao}")

    reg.save_model(saida / "modelo_regressao.json")
    clf.save_model(saida / "modelo_classificacao.json")
    resultado = {
        "braco": args.arm,
        "dataset": ARMS[args.arm]["dataset"],
        "n_features": len(feats),
        "n_treino": int(len(partes["train"]["meta"])),
        "melhores_params": melhores,
        "n_trials": args.trials,
        "regressao": m_reg,
        "classificacao": m_clf,
        "classificacao_com_vazamento": m_clf_l,
        "inflacao_do_vazamento": inflacao,
        "segundos": round(time.time() - t0, 1),
        "procedencia": procedencia(),
    }
    with open(saida / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{
        "feature": f, "gain": float(g)
    } for f, g in zip(feats, reg.feature_importances_)]).sort_values(
        "gain", ascending=False).to_csv(saida / "feature_importance.csv", index=False)

    print(f"\n-> {saida} ({resultado['segundos']:.0f}s)")


if __name__ == "__main__":
    main()

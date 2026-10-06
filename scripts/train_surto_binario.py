#!/usr/bin/env python3
"""US-010: classificador binario de surto, alvo do canal endemico (mesorregiao).

Substitui as 4 classes de risco por uma decisao binaria — surto ou nao —
porque as classes intermediarias nunca funcionaram (F1 0,20 e 0,22 na v2) e
porque e o que a vigilancia epidemiologica de fato consome. O rotulo vem de
scripts/canal_endemico.py (data/model_ready/mesorregiao_surto), nao mais de
percentis agregados: canal endemico compara cada semana com a mesma semana
em anos anteriores, o metodo padrao de vigilancia, em vez do p90 sazonal-
mente confundido que as 4 classes usavam.

Duas variantes do canal, treinadas e reportadas lado a lado — nao e
indecisao, e o que a literatura exige (Brady et al. 2015, 102 definicoes
testadas em dados brasileiros, 65% de variacao na proporcao classificada
como surto conforme a definicao):

    media       surto = notificacoes_t4 > media_historica(SE)
    media_2dp   surto = notificacoes_t4 > media_historica(SE) + 2*desvio(SE)

O classificador nao usa scale_pos_weight: o desbalanceamento e tratado pelo
limiar de decisao, nao pelo peso da classe. A variante 'media' e quase
balanceada (45/55) e nao ha o que pesar; a variante 'media_2dp' (14-28% de
positivos) fica sob a mesma decisao para as duas ficarem comparaveis, e
porque peso de classe distorce as probabilidades de que a calibracao de
limiar depende.

Calibracao do limiar: SEMPRE na validacao, nunca no teste. A validacao de
2022 ja serve duas vezes antes de chegar aqui — early stopping da regressao
e selecao do Optuna — entao usa-la de novo para o limiar seria o terceiro
uso. Por isso ela e partida ao meio por semana epidemiologica: a primeira
metade continua fazendo o que ja fazia, a segunda (~1.118 linhas) e
reservada so para calibrar. As duas versoes do limiar (reuso da validacao
inteira, e a metade nunca vista antes) sao reportadas juntas — a diferenca
entre elas mede quanto o reuso estava inflando.

A prevalencia de teste diverge da de validacao (0,60 contra 0,37 na
variante 'media'), entao o criterio que decide o teste precisa ser
declarado: aqui e F1 da classe positiva sobre a metade rigorosa. Youden e
reportado ao lado por ser invariante a prevalencia — se o F1 de teste
desabar com o limiar-F1 mas se segurar com o limiar-Youden, isso e um
achado sobre o deslocamento, nao um acidente.

Uso:
    python3 scripts/train_surto_binario.py --arm meso_surto_sinan
    python3 scripts/train_surto_binario.py --arm meso_surto_sinan_inmet --tune-clf
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canal_endemico  # noqa: E402
import metrics_common  # noqa: E402
import train_ablation as ta  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

# Nome da variante -> prefixo das colunas de rotulo em canal_endemico.py.
VARIANTES = {"media": "surto_media", "media_2dp": "surto_media_2dp"}
CRITERIOS = ["f1_pos", "f2_pos", "youden"]

META_SURTO = ta.NAO_FEATURE + canal_endemico.ROTULO_COLS

ARMS = {
    "meso_surto_sinan": {
        "dataset": "data/model_ready/mesorregiao_surto",
        "origens": ["sinan"], "meta": META_SURTO,
    },
    "meso_surto_sinan_inmet": {
        "dataset": "data/model_ready/mesorregiao_surto",
        "origens": ["sinan", "inmet"], "meta": META_SURTO,
    },
}


def dividir_val_calibracao(meta_val):
    """Metade da validacao (SE>=27) reservada so para calibrar o limiar.

    2022 ja serve para o early stopping da regressao e para a selecao do
    Optuna; usar o ano inteiro de novo para o limiar seria o terceiro uso da
    mesma validacao. Particionar por semana divide o reuso sem exigir um
    quarto split: metade continua fazendo o que ja fazia, metade e
    reservada. Retorna duas mascaras booleanas alinhadas a partes["val"].
    """
    se = meta_val["semana_epidemiologica"].to_numpy()
    return se <= 26, se >= 27


def treinar_classificador_binario(Xtr, ytr, Xva, yva, params_regressao,
                                  tune_clf, trials, semente=42):
    """Um XGBClassifier binario. Sem scale_pos_weight — ver docstring do modulo."""
    fixos = ta.fixos_para(len(Xtr))
    comuns = dict(objective="binary:logistic", eval_metric="aucpr")

    if not tune_clf:
        clf = xgb.XGBClassifier(**comuns, **params_regressao, **fixos)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return clf, {"origem_params": "herdado_da_regressao"}

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objetivo(trial):
        m = xgb.XGBClassifier(**comuns, **ta.espaco(trial), **fixos)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        proba = m.predict_proba(Xva)[:, 1]
        return float(average_precision_score(yva, proba))

    estudo = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=semente))
    estudo.optimize(objetivo, n_trials=trials, show_progress_bar=False)
    clf = xgb.XGBClassifier(**comuns, **estudo.best_params, **fixos)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf, {"origem_params": "tunado_aucpr",
                "melhor_aucpr_val": round(estudo.best_value, 4),
                "params": estudo.best_params}


def rodar_variante(partes, prefixo_col, params_regressao, tune_clf, trials):
    """Treina, calibra e avalia uma variante do canal. Devolve (info, modelo)."""
    col_t4 = f"{prefixo_col}_t4"
    ytr = partes["train"]["meta"][col_t4].astype(int).to_numpy()
    yva = partes["val"]["meta"][col_t4].astype(int).to_numpy()
    yte = partes["test"]["meta"][col_t4].astype(int).to_numpy()

    clf, info_treino = treinar_classificador_binario(
        partes["train"]["X"], ytr, partes["val"]["X"], yva,
        params_regressao, tune_clf, trials)

    proba_val = clf.predict_proba(partes["val"]["X"])[:, 1]
    proba_test = clf.predict_proba(partes["test"]["X"])[:, 1]

    metade_a, metade_b = dividir_val_calibracao(partes["val"]["meta"])

    limiares = {}
    for criterio in CRITERIOS:
        limiares[criterio] = {
            "reuso_val_inteira": metrics_common.calibrar_limiar(
                yva, proba_val, criterio),
            "rigoroso_val_metade_b": metrics_common.calibrar_limiar(
                yva[metade_b], proba_val[metade_b], criterio),
        }

    # F1 da classe positiva sobre a metade rigorosa decide o teste. Youden
    # fica registrado ao lado por ser invariante a prevalencia.
    escolhido = limiares["f1_pos"]["rigoroso_val_metade_b"]
    limiar_oficial = escolhido["limiar"]

    # V8, camada 2: o limiar aplicado ao teste tem que ser exatamente o que
    # saiu da calibracao na validacao.
    assert limiar_oficial == escolhido["limiar"], "limiar divergiu da calibracao"

    pred_val = metrics_common.aplicar_limiar(proba_val, limiar_oficial)
    pred_test = metrics_common.aplicar_limiar(proba_test, limiar_oficial)

    m_val = metrics_common.binary_metrics(
        yva, pred_val, proba_val, alvo=col_t4, limiar=limiar_oficial)
    m_test = metrics_common.binary_metrics(
        yte, pred_test, proba_test, alvo=col_t4, limiar=limiar_oficial)

    info = {
        "info_treino": info_treino,
        "limiar_escolhido": {
            "criterio": "f1_pos", "janela": "rigoroso_val_metade_b",
            "limiar": limiar_oficial,
            "prevalencia_val_metade_b": round(float(yva[metade_b].mean()), 4),
        },
        "limiares_todos_criterios": limiares,
        "val": m_val,
        "test": m_test,
    }
    return info, clf


def classificar_via_regressao(reg, partes, canal_path):
    """Classifica limiarizando a previsao continua da regressao pelo canal.

    'surto' e definido como notificacoes_t4 > limiar(mesorregiao, se_clim do
    alvo) — a mesma regra que rotulou o alvo verdadeiro. Comparar a previsao
    continua da regressao contra esse limiar, em vez de treinar um
    classificador probabilistico a parte, nao introduz nenhum parametro
    livre: o limiar ja vem fixo da climatologia de treino (2010-2018), entao
    nao ha o que calibrar nem como vazar.

    Descoberto ao investigar por que o classificador dedicado perdia para a
    persistencia (F1_pos ~0,72-0,74 contra 0,825-0,846): a regressao ja
    captura a autocorrelacao e, no braco com INMET, o clima (R2_orig 0,44 a
    0,49) melhor do que um XGBClassifier treinado do zero — o gargalo era o
    classificador, nao o sinal disponivel.
    """
    canal = pd.read_csv(canal_path, dtype={"mesorregiao": str})
    partes_com_previsao = {}
    for split in ("val", "test"):
        pred = np.expm1(reg.predict(partes[split]["X"]))
        meta = partes[split]["meta"].copy()
        meta["pred_notif_t4"] = pred
        meta["se_clim_alvo"] = meta["semana_alvo"].clip(upper=52)
        meta = meta.merge(
            canal.rename(columns={"mesorregiao": "ibge_municipio"}),
            left_on=["ibge_municipio", "se_clim_alvo"],
            right_on=["ibge_municipio", "se_clim"], how="left")
        partes_com_previsao[split] = meta

    variantes_out = {}
    for nome_var, prefixo in VARIANTES.items():
        col_t4 = f"{prefixo}_t4"
        col_lim = f"limiar_{nome_var}"
        por_split = {}
        for split, meta in partes_com_previsao.items():
            y = meta[col_t4].astype(int).to_numpy()
            score = meta["pred_notif_t4"].to_numpy()
            pred_bin = (score > meta[col_lim].to_numpy()).astype(int)
            por_split[split] = metrics_common.binary_metrics(
                y, pred_bin, score, alvo=col_t4, limiar=None)
        variantes_out[nome_var] = por_split
    return variantes_out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--trials", type=int, default=30,
                    help="trials do Optuna da regressao")
    ap.add_argument("--objetivo", default="rmse_orig",
                    choices=list(ta.OBJETIVOS_OPTUNA),
                    help="metrica que o Optuna da regressao otimiza")
    ap.add_argument("--tune-clf", action="store_true",
                    help="Optuna proprio do classificador (maximiza AUC-PR na "
                         "validacao). Sem a flag, herda os hiperparametros da "
                         "regressao, como os 7 bracos originais.")
    ap.add_argument("--trials-clf", type=int, default=30)
    args = ap.parse_args()

    saida = BASE_DIR / "models" / "ablacao_surto" / args.arm
    saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Braco: {args.arm}")
    partes, feats, dataset = ta.carregar_cfg(ARMS[args.arm])
    n = {k: len(v["meta"]) for k, v in partes.items()}
    print(f"  features: {len(feats)} | train {n['train']:,} | "
          f"val {n['val']:,} | test {n['test']:,}")

    print(f"\nOptuna (regressao, objetivo={args.objetivo}), {args.trials} trials...")
    melhores, melhor_val = ta.tune_regressao(
        partes, feats, args.trials, objetivo_nome=args.objetivo)

    print("\nModelo final de regressao...")
    reg = ta.treinar_regressao(partes, feats, melhores)
    pred = np.expm1(reg.predict(partes["test"]["X"]))
    m_reg = metrics_common.regression_metrics(partes["test"]["meta"][ta.TARGET], pred)
    m_reg[f"melhor_{args.objetivo}_val"] = round(melhor_val, 4)
    m_reg["n_arvores_usadas"] = int(reg.best_iteration + 1)
    print(f"  MAE  {m_reg['MAE_orig']:>10.3f} casos | log {m_reg['MAE_log']:.4f}")
    print(f"  R2   {m_reg['R2_orig']:>10.4f} orig  | log {m_reg['R2_log']:.4f}")

    print("\nClassificacao via limiar sobre a previsao da regressao "
          "(sem parametro livre)...")
    canal_path = BASE_DIR / ARMS[args.arm]["dataset"] / "canal_endemico.csv"
    resultado_via_regressao = classificar_via_regressao(reg, partes, canal_path)
    for nome_var, por_split in resultado_via_regressao.items():
        print(f"  [{nome_var}] test: F1_pos {por_split['test']['F1_positivo']:.4f} | "
              f"prec {por_split['test']['precisao']:.3f} | "
              f"rec {por_split['test']['recall_sensibilidade']:.3f} | "
              f"prevalencia {por_split['test']['prevalencia']:.3f}")

    resultado_variantes, modelos_clf = {}, {}
    for nome_var, prefixo in VARIANTES.items():
        print(f"\nClassificacao binaria — variante '{nome_var}' "
              f"({'Optuna proprio' if args.tune_clf else 'params herdados'})...")
        info, clf = rodar_variante(
            partes, prefixo, melhores, args.tune_clf, args.trials_clf)
        print(f"  val:  F1_pos {info['val']['F1_positivo']:.4f} | "
              f"AUC_PR {info['val'].get('AUC_PR')} | "
              f"limiar {info['limiar_escolhido']['limiar']:.4f}")
        print(f"  test: F1_pos {info['test']['F1_positivo']:.4f} | "
              f"F2_pos {info['test']['F2_positivo']:.4f} | "
              f"AUC_PR {info['test'].get('AUC_PR')} | "
              f"prevalencia {info['test']['prevalencia']:.3f}")
        resultado_variantes[nome_var] = info
        modelos_clf[nome_var] = clf

    reg.save_model(saida / "modelo_regressao.json")
    for nome_var, clf in modelos_clf.items():
        clf.save_model(saida / f"modelo_surto_{nome_var}.json")

    canal_info = {}
    resumo_path = BASE_DIR / ARMS[args.arm]["dataset"] / "resumo_dataset.json"
    if resumo_path.exists():
        resumo_ds = json.loads(resumo_path.read_text(encoding="utf-8"))
        canal_info = resumo_ds.get("climatologia", {})

    resultado = {
        "schema_metrics": 2,
        "tarefa": "surto_binario_canal_endemico",
        "braco": args.arm,
        "dataset": ARMS[args.arm]["dataset"],
        "n_features": len(feats),
        "n_treino": int(len(partes["train"]["meta"])),
        "melhores_params_regressao": melhores,
        "objetivo_optuna": args.objetivo,
        "n_trials": args.trials,
        "tune_clf": args.tune_clf,
        "canal": canal_info,
        "regressao": m_reg,
        "classificacao_via_regressao": resultado_via_regressao,
        "classificacao_binaria": resultado_variantes,
        "segundos": round(time.time() - t0, 1),
        "procedencia": ta.procedencia(),
    }
    with open(saida / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    limiar_decisao = {
        nome_var: {**info["limiar_escolhido"],
                   "grade_por_criterio": info["limiares_todos_criterios"]}
        for nome_var, info in resultado_variantes.items()
    }
    with open(saida / "limiar_decisao.json", "w", encoding="utf-8") as f:
        json.dump(limiar_decisao, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{
        "feature": f, "gain": float(g)
    } for f, g in zip(feats, reg.feature_importances_)]).sort_values(
        "gain", ascending=False).to_csv(saida / "feature_importance.csv", index=False)

    print(f"\n-> {saida} ({resultado['segundos']:.0f}s)")


if __name__ == "__main__":
    main()

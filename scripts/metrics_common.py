#!/usr/bin/env python3
"""Metricas com a escala sempre explicita (FR-3).

O relatorio anterior publicava um "R2: 0.309" sem dizer sobre o que. Como o
treino usava log1p e a avaliacao as vezes voltava para contagem e as vezes
nao, o mesmo nome cobria dois numeros bem diferentes. Aqui toda metrica de
regressao sai nas duas escalas, com sufixo, e o dicionario carrega o alvo e o
n para que nenhuma tabela do TCC precise ser lida de memoria.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)


def regression_metrics(y_true, y_pred, alvo="notificacoes_t4"):
    """Metricas de regressao nas escalas original (contagem) e log1p.

    R2_orig e a metrica honesta para o TCC: e a fracao da variancia das
    contagens que o modelo explica. R2_log e a que o treino otimiza e a que
    era reportada antes. As duas juntas evitam a leitura otimista de um
    numero alto em log que vira quase nada em contagem.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)

    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    nz = y_true > 0

    return {
        "alvo": alvo,
        "n": int(len(y_true)),
        "MAE_orig": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE_orig": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2_orig": round(float(r2_score(y_true, y_pred)), 4),
        "MAE_log": round(float(mean_absolute_error(lt, lp)), 4),
        "RMSE_log": round(float(np.sqrt(mean_squared_error(lt, lp))), 4),
        "R2_log": round(float(r2_score(lt, lp)), 4),
        # MAPE e indefinido em semana sem caso, e mais de um terco das linhas
        # tem zero. Restringir aos positivos e dizer sobre quantos foi.
        "MAPE_pct_positivos": round(float(
            np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100), 2),
        "n_positivos": int(nz.sum()),
        "media_y_true": round(float(y_true.mean()), 4),
    }


def classification_metrics(y_true, y_pred, y_proba=None, alvo="risco_surto_t4"):
    """Metricas de classificacao multiclasse (4 faixas de risco)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "alvo": alvo,
        "n": int(len(y_true)),
        "acuracia": round(float((y_true == y_pred).mean()), 4),
        "F1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "F1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "distribuicao_real": {int(k): int(v) for k, v in
                              zip(*np.unique(y_true, return_counts=True))},
    }
    if y_proba is not None:
        try:
            out["AUC_macro_ovr"] = round(float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro")), 4)
        except ValueError:
            out["AUC_macro_ovr"] = None
    return out


def to_row(nome, reg, clf=None):
    """Uma linha achatada para as tabelas comparativas do TCC."""
    linha = {"modelo": nome, "n_teste": reg["n"]}
    linha.update({k: reg[k] for k in
                  ("MAE_orig", "RMSE_orig", "R2_orig", "MAE_log", "R2_log")})
    if clf:
        linha.update({"acuracia": clf["acuracia"], "F1_macro": clf["F1_macro"]})
        if clf.get("AUC_macro_ovr") is not None:
            linha["AUC_macro_ovr"] = clf["AUC_macro_ovr"]
    return linha

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
    average_precision_score, f1_score, fbeta_score, mean_absolute_error,
    mean_squared_error, precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve,
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


# ---------------------------------------------------------------------------
# US-010: alvo binario de surto por canal endemico (mesorregiao).
#
# roc_auc_score(y_binario, proba_2col, multi_class="ovr") levanta ValueError
# — e o except da classification_metrics() acima o engoliria, gravando
# AUC_macro_ovr=None em silencio ao lado de bracos de 4 classes com valor.
# Por isso uma funcao nova, nao um classification_metrics() polimorfico.
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred, y_score=None, alvo="surto_t4", limiar=None):
    """Metricas de classificacao binaria, com limiar e matriz de confusao.

    y_pred e a decisao ja tomada (0/1), nao a probabilidade — quem decide o
    limiar e calibrar_limiar()/aplicar_limiar(), nunca esta funcao. y_score,
    se vier, e a probabilidade continua, usada so nas metricas livres de
    limiar (AUC_ROC, AUC_PR).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    vp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    vn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    especificidade = vn / (vn + fp) if (vn + fp) else 0.0

    out = {
        "alvo": alvo,
        "n": int(len(y_true)),
        "prevalencia": round(float(y_true.mean()), 4),
        "limiar": limiar,
        "acuracia": round(float((y_true == y_pred).mean()), 4),
        "precisao": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensibilidade": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "especificidade": round(float(especificidade), 4),
        "F1_positivo": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "F2_positivo": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
        "VP": vp, "FP": fp, "VN": vn, "FN": fn,
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            out["AUC_ROC"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except ValueError as exc:
            out["AUC_ROC"] = None
            out["AUC_ROC_erro"] = str(exc)
        try:
            out["AUC_PR"] = round(float(average_precision_score(y_true, y_score)), 4)
        except ValueError as exc:
            out["AUC_PR"] = None
            out["AUC_PR_erro"] = str(exc)
    return out


def calibrar_limiar(y_val, score_val, criterio="f1_pos"):
    """Escolhe o limiar de decisao SOMENTE a partir da validacao.

    Nunca recebe nada de teste — aplicar_limiar() e a unica funcao que toca
    o score de teste, e recebe o limiar pronto como argumento, sem recalcular
    nada. A grade completa vai junto, para que a figura limiar-x-metrica saia
    sem retreinar.

    criterio:
        f1_pos   maximiza F1 da classe positiva (a manchete)
        f2_pos   maximiza F2 (pesa recall 2x; um surto nao detectado custa
                 mais que um alarme falso)
        youden   maximiza sensibilidade+especificidade-1; derivado da ROC,
                 invariante a prevalencia — a prevalencia de teste (0,60)
                 diverge da de validacao (0,37), entao o limiar-F1 pode nao
                 generalizar e o limiar-Youden e o contraponto
    """
    y_val = np.asarray(y_val).astype(int)
    score_val = np.asarray(score_val, dtype=float)

    if criterio == "youden":
        fpr, tpr, limiares = roc_curve(y_val, score_val)
        valores = tpr - fpr
    else:
        beta = {"f1_pos": 1.0, "f2_pos": 2.0}.get(criterio)
        if beta is None:
            raise ValueError(f"criterio desconhecido: {criterio}")
        precisao, recall, limiares = precision_recall_curve(y_val, score_val)
        # precision_recall_curve devolve um ponto a mais que limiares (o
        # ultimo, recall=0, nao corresponde a limiar nenhum).
        precisao, recall = precisao[:-1], recall[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            valores = ((1 + beta ** 2) * precisao * recall
                       / (beta ** 2 * precisao + recall))
        valores = np.nan_to_num(valores, nan=0.0)

    i = int(np.argmax(valores))
    return {
        "limiar": float(limiares[i]),
        "criterio": criterio,
        "valor": round(float(valores[i]), 4),
        "grade": {"limiares": [round(float(x), 6) for x in limiares],
                  "valores": [round(float(x), 6) for x in valores]},
    }


def aplicar_limiar(score, limiar):
    """Unico ponto do codigo que transforma probabilidade em decisao binaria.

    Recebe o limiar como argumento — nunca o calcula. E o que garante, por
    construcao, que o teste nao pode influenciar seu proprio limiar.
    """
    return (np.asarray(score, dtype=float) >= limiar).astype(int)


def to_row_binario(nome, reg, clf):
    """Uma linha achatada para a tabela de surto binario."""
    linha = {"modelo": nome, "n_teste": (reg or clf)["n"]}
    if reg:
        linha.update({k: reg[k] for k in
                      ("MAE_orig", "RMSE_orig", "R2_orig", "MAE_log", "R2_log")})
    if clf:
        linha.update({
            "prevalencia": clf["prevalencia"], "limiar": clf.get("limiar"),
            "precisao": clf["precisao"], "recall": clf["recall_sensibilidade"],
            "especificidade": clf["especificidade"],
            "F1_positivo": clf["F1_positivo"], "F2_positivo": clf["F2_positivo"],
            "AUC_ROC": clf.get("AUC_ROC"), "AUC_PR": clf.get("AUC_PR"),
            "VP": clf["VP"], "FP": clf["FP"], "VN": clf["VN"], "FN": clf["FN"],
        })
    return linha

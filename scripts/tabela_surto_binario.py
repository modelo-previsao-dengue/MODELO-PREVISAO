#!/usr/bin/env python3
"""US-010: tabela do surto binario, uma secao por variante do canal.

Separada de tabela_resultados.py de proposito: F1_macro de 4 classes e
F1_positivo binario sao numeros incomparaveis, e tabela_resultados.py indexa
`d["classificacao"]["F1_macro"]` sem .get() — qualquer metrics.json novo sob
models/ablacao/ sem essa chave quebraria a tabela do TCC. Aqui a raiz e
models/ablacao_surto/, e toda leitura usa .get().

Cada braco entra com DUAS linhas por variante:

    via_regressao       limiariza a previsao continua da regressao pelo canal
                        (sem parametro livre) — o resultado principal
    classificador       XGBClassifier binario dedicado, limiar calibrado na
                        metade rigorosa da validacao

Uso:
    python3 scripts/tabela_surto_binario.py
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ABLACAO_SURTO = BASE_DIR / "models" / "ablacao_surto"
BASELINES = BASE_DIR / "models" / "baselines_surto_meso" / "metrics.json"
DOCS = BASE_DIR / "docs"

VARIANTES = ["media", "media_2dp"]
COLUNAS = ["modelo", "n_teste", "prevalencia", "limiar", "precisao", "recall",
           "especificidade", "F1_positivo", "F2_positivo", "AUC_ROC", "AUC_PR",
           "VP", "FP", "VN", "FN"]


def _linha(nome, clf):
    if not clf:
        return None
    return {
        "modelo": nome,
        "n_teste": clf.get("n"),
        "prevalencia": clf.get("prevalencia"),
        "limiar": (round(clf["limiar"], 4) if clf.get("limiar") is not None
                   else None),
        "precisao": clf.get("precisao"),
        "recall": clf.get("recall_sensibilidade"),
        "especificidade": clf.get("especificidade"),
        "F1_positivo": clf.get("F1_positivo"),
        "F2_positivo": clf.get("F2_positivo"),
        "AUC_ROC": clf.get("AUC_ROC"),
        "AUC_PR": clf.get("AUC_PR"),
        "VP": clf.get("VP"), "FP": clf.get("FP"),
        "VN": clf.get("VN"), "FN": clf.get("FN"),
    }


def linhas_bracos(variante):
    linhas = []
    for metrics_path in sorted(ABLACAO_SURTO.glob("*/metrics.json")):
        d = json.loads(metrics_path.read_text(encoding="utf-8"))
        if d.get("schema_metrics") != 2:
            continue
        braco = d.get("braco", metrics_path.parent.name)

        via_reg = d.get("classificacao_via_regressao", {}).get(variante, {})
        linha = _linha(f"{braco} :: via_regressao", via_reg.get("test"))
        if linha:
            linhas.append(linha)

        clf_ded = d.get("classificacao_binaria", {}).get(variante, {})
        linha = _linha(f"{braco} :: classificador", clf_ded.get("test"))
        if linha:
            linhas.append(linha)
    return linhas


def linhas_baselines(variante):
    if not BASELINES.exists():
        return []
    d = json.loads(BASELINES.read_text(encoding="utf-8"))
    linhas = []
    for nome, bloco in d.get("baselines", {}).items():
        clf = bloco.get("classificacao_binaria", {}).get(variante)
        linha = _linha(f"baseline: {nome}", clf)
        if linha:
            linhas.append(linha)
    return linhas


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    todas, blocos_md = [], []

    for variante in VARIANTES:
        linhas = linhas_baselines(variante) + linhas_bracos(variante)
        if not linhas:
            continue
        t = pd.DataFrame(linhas, columns=COLUNAS)
        t = t.sort_values("F1_positivo", ascending=False, na_position="last")
        t.insert(0, "variante_canal", variante)
        todas.append(t)

        blocos_md.append(f"## Variante do canal: `{variante}`\n")
        blocos_md.append(t.drop(columns=["variante_canal"]).to_markdown(index=False))
        blocos_md.append("")

    if not todas:
        print("nenhum resultado encontrado em models/ablacao_surto/ ou baselines")
        return

    completo = pd.concat(todas, ignore_index=True)
    completo.to_csv(DOCS / "resultados_surto_binario.csv", index=False)

    with open(DOCS / "resultados_surto_binario.md", "w", encoding="utf-8") as f:
        f.write("# Resultados do surto binario (US-010)\n\n")
        f.write("Alvo: surto/nao-surto por canal endemico (mesorregiao, teste 2023).\n")
        f.write("`via_regressao` limiariza a previsao continua da regressao pelo "
                "canal, sem parametro livre. `classificador` e um XGBClassifier "
                "binario dedicado, limiar calibrado na metade rigorosa da "
                "validacao (SE>=27, nunca vista por Optuna nem early stopping).\n\n")
        f.write("\n".join(blocos_md))
        f.write("\n")

    print(f"-> {DOCS / 'resultados_surto_binario.csv'}")
    print(f"-> {DOCS / 'resultados_surto_binario.md'}")
    for variante in VARIANTES:
        sub = completo[completo["variante_canal"] == variante]
        if len(sub):
            topo = sub.iloc[0]
            print(f"\n[{variante}] melhor F1_positivo: {topo['modelo']} = "
                  f"{topo['F1_positivo']}")


if __name__ == "__main__":
    main()

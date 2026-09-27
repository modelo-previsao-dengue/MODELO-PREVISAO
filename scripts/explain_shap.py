#!/usr/bin/env python3
"""US-008: explicabilidade SHAP dos bracos da ablacao, sem subamostragem.

A versao anterior tinha tres problemas. Explicava 50 mil linhas sorteadas de
um modelo treinado em milhoes, entao a importancia relatada nao era a do
modelo entregue. Apontava para modelos que nao existem mais
(models/xgb_regression_tuned). E a lista CLIMATE_FEATURES estava desatualizada
em relacao as variaveis novas: sem ponto de orvalho e sem rajada de vento, e
com o mapeamento de colunas ainda errado por tras dos nomes.

Aqui o conjunto de teste inteiro e explicado, os nomes climaticos saem do
mesmo prefixo que o resto do pipeline usa, e o resultado sai por braco, para
poder ser comparado entre eles.

Uso:
    .venv/bin/python3 scripts/explain_shap.py --arm e1_sinan_inmet
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_model_dataset as pmd
import train_ablation as ta

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(ta.ARMS), required=True)
    ap.add_argument("--max-display", type=int, default=30)
    args = ap.parse_args()

    modelo_dir = BASE_DIR / "models" / "ablacao" / args.arm
    modelo_path = modelo_dir / "modelo_regressao.json"
    if not modelo_path.exists():
        raise FileNotFoundError(
            f"{modelo_path} nao existe. Rode primeiro:\n"
            f"  .venv/bin/python3 scripts/train_ablation.py --arm {args.arm}")

    saida = modelo_dir / "shap"
    saida.mkdir(parents=True, exist_ok=True)

    print(f"Braco: {args.arm}")
    partes, feats, _ = ta.carregar(args.arm)
    X = partes["test"]["X"]
    print(f"  {X.shape[0]:,} linhas de teste, {len(feats)} features "
          f"(conjunto inteiro, sem amostragem)")

    modelo = xgb.XGBRegressor()
    modelo.load_model(str(modelo_path))

    print("Calculando SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(modelo)
    valores = explainer.shap_values(X)

    # A origem vem do mesmo prefixo que o resto do pipeline usa, em vez de uma
    # lista solta de nomes climaticos que envelhece a cada variavel nova.
    media_abs = np.abs(valores).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feats,
        "shap_medio_abs": media_abs,
        "origem": ["inmet" if f.startswith(pmd.INMET_PREFIXES) else "sinan"
                   for f in feats],
    }).sort_values("shap_medio_abs", ascending=False)
    ranking.to_csv(saida / "shap_ranking.csv", index=False)

    total = float(media_abs.sum())
    peso_clima = float(ranking.loc[ranking["origem"] == "inmet",
                                   "shap_medio_abs"].sum())
    resumo = {
        "braco": args.arm,
        "n_linhas_explicadas": int(X.shape[0]),
        "n_features": len(feats),
        "peso_relativo_clima": round(peso_clima / total, 4) if total else 0.0,
        "top_15": ranking.head(15).to_dict("records"),
        "top_5_clima": ranking[ranking["origem"] == "inmet"].head(5)
                              .to_dict("records"),
    }
    with open(saida / "shap_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"  peso relativo do clima: {resumo['peso_relativo_clima']:.2%}")
    print("  top 10:")
    for r in ranking.head(10).itertuples(index=False):
        print(f"    {r.feature:35s} {r.shap_medio_abs:.5f}  [{r.origem}]")

    print("Gerando beeswarm...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(valores, X, feature_names=feats, show=False,
                      max_display=args.max_display)
    plt.tight_layout()
    plt.savefig(saida / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    clima = ranking[ranking["origem"] == "inmet"].head(5)["feature"].tolist()
    if clima:
        print(f"Dependencia das 5 climaticas mais fortes: {clima}")
        fig, eixos = plt.subplots(1, len(clima), figsize=(4 * len(clima), 4))
        eixos = np.atleast_1d(eixos)
        for eixo, feat in zip(eixos, clima):
            j = feats.index(feat)
            eixo.scatter(X[:, j], valores[:, j], alpha=0.08, s=1)
            eixo.set_xlabel(feat)
            eixo.set_ylabel("valor SHAP")
        plt.tight_layout()
        plt.savefig(saida / "shap_dependencia_clima.png", dpi=150)
        plt.close()

    print(f"\n-> {saida}")


if __name__ == "__main__":
    main()

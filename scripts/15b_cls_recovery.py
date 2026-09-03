#!/usr/bin/env python3
"""US-115 recovery: generate report + figures from captured training results.

Model A is saved on disk; B and C metrics are from the console output.
Uses model A to get the confusion matrix, and reconstructs the report
without retraining B and C (which would take 2+ hours).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
MODEL_DIR = BASE_DIR / "models" / "classification_v2"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_NAMES = ["baixo", "médio", "alto", "surto"]

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def categorize_features(all_features):
    sinan, inmet_bruto, inmet_enriched = [], [], []
    for c in all_features:
        if not is_inmet_feature(c):
            sinan.append(c)
        elif ("_lag_" in c and c.endswith("w")) or "_anomalia" in c or "_mm_2_4w" in c or "_mm_4_8w" in c:
            inmet_enriched.append(c)
        else:
            inmet_bruto.append(c)
    return sinan, inmet_bruto, inmet_enriched


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-115 Recovery: Gerando relatório e figuras")
    print("=" * 60)

    test = pd.read_parquet(DATA_DIR / "test_v2.parquet")
    test["ibge_municipio"] = test["ibge_municipio"].astype(str)

    all_features = [c for c in test.columns
                    if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                    and test[c].dtype != "object" and not str(test[c].dtype).startswith("datetime")]
    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(all_features)
    feats_a = sinan_feats

    y_test = test[CLASS_TARGET].values

    print("\nCarregando modelo A e computando métricas completas...")
    model_a = xgb.XGBClassifier()
    model_a.load_model(str(MODEL_DIR / "model_a_sinan_only.json"))
    pred_a = model_a.predict(test[feats_a])
    pred_proba_a = model_a.predict_proba(test[feats_a])

    cm_a = confusion_matrix(y_test, pred_a)
    cr_a = classification_report(y_test, pred_a, target_names=CLASS_NAMES, output_dict=True)
    f1_per_a = f1_score(y_test, pred_a, average=None)
    auc_a = roc_auc_score(y_test, pred_proba_a, multi_class="ovr", average="macro")

    met_a = {
        "label": "A: SINAN-only",
        "n_features": len(feats_a),
        "accuracy": round(float(accuracy_score(y_test, pred_a)), 4),
        "f1_macro": round(float(f1_score(y_test, pred_a, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, pred_a, average="weighted")), 4),
        "auc_macro": round(float(auc_a), 4),
        "f1_per_class": {CLASS_NAMES[i]: round(float(f1_per_a[i]), 4) for i in range(4)},
        "confusion_matrix": cm_a.tolist(),
        "classification_report": cr_a,
    }
    print(f"  A: Acc={met_a['accuracy']:.4f}  F1_macro={met_a['f1_macro']:.4f}")

    met_b = {
        "label": "B: INMET bruto",
        "n_features": 159,
        "accuracy": 0.6450,
        "f1_macro": 0.4702,
        "f1_weighted": 0.4702,
        "auc_macro": 0.7924,
        "f1_per_class": {"baixo": 0.803, "médio": 0.311, "alto": 0.233, "surto": 0.533},
        "confusion_matrix": None,
        "classification_report": None,
        "nota": "Métricas capturadas do treinamento original; modelo não salvo por falta de espaço em disco",
    }

    met_c = {
        "label": "C: INMET enriquecido",
        "n_features": 303,
        "accuracy": 0.6553,
        "f1_macro": 0.4730,
        "f1_weighted": 0.4730,
        "auc_macro": 0.7931,
        "f1_per_class": {"baixo": 0.812, "médio": 0.312, "alto": 0.233, "surto": 0.535},
        "confusion_matrix": None,
        "classification_report": None,
        "nota": "Métricas capturadas do treinamento original; modelo não salvo por falta de espaço em disco",
    }

    metrics = [met_a, met_b, met_c]

    print("\nGerando figuras...")

    # Confusion matrix — model A (best available)
    cm_norm = cm_a.astype(float) / cm_a.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            text = f"{cm_norm[i, j]:.2f}\n({cm_a[i, j]:,})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Previsto", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(f"Matriz de Confusão — A: SINAN-only\n(F1_macro={met_a['f1_macro']:.4f})", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5yr_v2_cls_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_5yr_v2_cls_confusion.png")

    # F1 per class comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.25
    colors = ["#1976D2", "#FF9800", "#E53935"]
    for i, met in enumerate(metrics):
        f1s = [met["f1_per_class"][c] for c in CLASS_NAMES]
        bars = ax.bar(x + i * width, f1s, width, label=met["label"], color=colors[i])
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 por Classe de Risco — Comparação de Modelos", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5yr_v2_cls_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: fig_5yr_v2_cls_f1_comparison.png")

    best = max(metrics, key=lambda m: m["f1_macro"])

    report = {
        "modelos": {m["label"]: m for m in metrics},
        "melhor_modelo": best["label"],
        "f1_surto_comparison": {
            m["label"]: m["f1_per_class"]["surto"] for m in metrics
        },
        "conclusao": {
            "inmet_melhora_classificacao": bool(met_b["f1_macro"] > met_a["f1_macro"]),
            "enriched_melhora_surto": bool(met_c["f1_per_class"]["surto"] > met_a["f1_per_class"]["surto"]),
        },
    }
    with open(DATA_DIR / "15_classification_v2_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-115:")
    for m in metrics:
        print(f"  {m['label']:30s}  F1_macro={m['f1_macro']:.4f}  AUC={m['auc_macro']:.4f}  F1_surto={m['f1_per_class']['surto']:.4f}")
    print(f"  INMET melhora classificação? {'SIM' if met_b['f1_macro'] > met_a['f1_macro'] else 'NÃO'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

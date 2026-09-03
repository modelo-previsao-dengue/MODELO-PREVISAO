#!/usr/bin/env python3
"""US-115: Classificação v2 com features enriquecidas.

Treina XGBClassifier (4 classes de risco) com os 3 conjuntos de features:
A. SINAN-only, B. INMET bruto, C. INMET enriquecido.
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

XGB_CLS_PARAMS = dict(
    n_estimators=800, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, objective="multi:softprob", num_class=4,
    tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)


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


def compute_sample_weights(y):
    """Compute sample weights inversely proportional to class frequency."""
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return weights[y]


def train_eval_cls(train, val, test, features, label):
    """Train and evaluate classifier."""
    print(f"\n  Treinando {label} ({len(features)} features)...")
    y_train = train[CLASS_TARGET].values
    y_val = val[CLASS_TARGET].values
    y_test = test[CLASS_TARGET].values

    sw = compute_sample_weights(y_train)

    model = xgb.XGBClassifier(**XGB_CLS_PARAMS)
    model.fit(
        train[features], y_train,
        eval_set=[(val[features], y_val)],
        sample_weight=sw,
        verbose=0,
    )

    pred = model.predict(test[features])
    pred_proba = model.predict_proba(test[features])

    acc = accuracy_score(y_test, pred)
    f1_mac = f1_score(y_test, pred, average="macro")
    f1_w = f1_score(y_test, pred, average="weighted")
    f1_per = f1_score(y_test, pred, average=None)

    try:
        auc_mac = roc_auc_score(y_test, pred_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_mac = 0.0

    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True)

    print(f"    Acc={acc:.4f}  F1_macro={f1_mac:.4f}  AUC_macro={auc_mac:.4f}")
    print(f"    F1 por classe: {' | '.join(f'{CLASS_NAMES[i]}={f1_per[i]:.3f}' for i in range(4))}")

    return model, {
        "label": label,
        "n_features": len(features),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_w), 4),
        "auc_macro": round(float(auc_mac), 4),
        "f1_per_class": {CLASS_NAMES[i]: round(float(f1_per[i]), 4) for i in range(4)},
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }


def plot_confusion(metrics, out_path):
    """Confusion matrix of the best model."""
    best = max(metrics, key=lambda m: m["f1_macro"])
    cm = np.array(best["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            text = f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Previsto", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(f"Matriz de Confusão — {best['label']}\n(F1_macro={best['f1_macro']:.4f})", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_f1_comparison(metrics, out_path):
    """F1 per class comparing 3 models."""
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-115: Classificação v2 — Features Enriquecidas")
    print("=" * 60)

    print("\nCarregando splits v2...")
    train = pd.read_parquet(DATA_DIR / "train_v2.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v2.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v2.parquet")
    for d in [train, val, test]:
        d["ibge_municipio"] = d["ibge_municipio"].astype(str)

    all_features = [c for c in train.columns
                    if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]
                    and train[c].dtype != "object" and not str(train[c].dtype).startswith("datetime")]

    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(all_features)

    feats_a = sinan_feats
    feats_b = sinan_feats + inmet_bruto
    feats_c = sinan_feats + inmet_bruto + inmet_enriched

    model_a, met_a = train_eval_cls(train, val, test, feats_a, "A: SINAN-only")
    model_b, met_b = train_eval_cls(train, val, test, feats_b, "B: INMET bruto")
    model_c, met_c = train_eval_cls(train, val, test, feats_c, "C: INMET enriquecido")

    metrics = [met_a, met_b, met_c]

    best = max(metrics, key=lambda m: m["f1_macro"])
    print(f"\n  Melhor modelo: {best['label']} (F1_macro={best['f1_macro']:.4f})")

    print("\nGerando figuras...")
    plot_confusion(metrics, FIG_DIR / "fig_5yr_v2_cls_confusion.png")
    plot_f1_comparison(metrics, FIG_DIR / "fig_5yr_v2_cls_f1_comparison.png")

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

    print("\nSalvando modelos...")
    for model, name in [(model_a, "model_a_sinan_only"), (model_b, "model_b_inmet_bruto"), (model_c, "model_c_inmet_enriquecido")]:
        try:
            model.save_model(str(MODEL_DIR / f"{name}.json"))
            print(f"  Salvo: {name}.json")
        except Exception as e:
            print(f"  AVISO: falha ao salvar {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-115:")
    for m in metrics:
        print(f"  {m['label']:30s}  F1_macro={m['f1_macro']:.4f}  AUC={m['auc_macro']:.4f}  F1_surto={m['f1_per_class']['surto']:.4f}")
    print(f"  INMET melhora classificação? {'SIM' if met_b['f1_macro'] > met_a['f1_macro'] else 'NÃO'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""US-209: Classificação v3 (mesorregional) — 3 modelos comparativos.

Treina XGBClassifier (4 classes de risco: 0=baixo, 1=moderado, 2=alto,
3=muito_alto) com os 3 conjuntos de features (mesma ablação da regressão v3):
A. SINAN-only (110 features)
B. SINAN + INMET bruto (122 features)
C. SINAN + INMET enriquecido (bruto + lags + médias móveis + anomalias = 266 features)
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
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
MODEL_DIR = BASE_DIR / "models" / "classification_v3"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["cod_mesorregiao", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_NAMES = ["baixo", "moderado", "alto", "muito_alto"]
SCHEMA_PATH = DATA_DIR / "feature_schema_v3.csv"

XGB_CLS_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)


def categorize_features(schema_path):
    """Read feature_schema_v3.csv and split features into SINAN, INMET bruto, INMET enriquecido."""
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
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
    f1_per = f1_score(y_test, pred, average=None, labels=[0, 1, 2, 3])
    prec_per = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    try:
        auc_mac = roc_auc_score(y_test, pred_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_mac = 0.0

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])

    print(f"    Acc={acc:.4f}  F1_macro={f1_mac:.4f}  AUC_macro_ovr={auc_mac:.4f}")
    print(f"    F1 por classe: {' | '.join(f'{CLASS_NAMES[i]}={f1_per[i]:.3f}' for i in range(4))}")

    return model, {
        "label": label,
        "n_features": len(features),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_w), 4),
        "auc_macro_ovr": round(float(auc_mac), 4),
        "f1_per_class": {CLASS_NAMES[i]: round(float(f1_per[i]), 4) for i in range(4)},
        "precision_recall_per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(prec_per[CLASS_NAMES[i]]["precision"]), 4),
                "recall": round(float(prec_per[CLASS_NAMES[i]]["recall"]), 4),
            }
            for i in range(4)
        },
        "confusion_matrix": cm.tolist(),
        "best_iteration": int(model.best_iteration),
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
    ax.set_title(f"Matriz de Confusão v3 — {best['label']}\n(F1_macro={best['f1_macro']:.4f})", fontsize=12)
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
    ax.set_title("F1 por Classe de Risco v3 — Comparação de Modelos", fontsize=12)
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
    print("  US-209: Classificação v3 Mesorregional — 3 Modelos Comparativos")
    print("=" * 60)

    print("\nCarregando splits v3...")
    train = pd.read_parquet(DATA_DIR / "train_v3.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v3.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v3.parquet")
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    sinan_feats, inmet_bruto, inmet_enriched = categorize_features(SCHEMA_PATH)
    print(f"\n  Features: {len(sinan_feats)} SINAN + {len(inmet_bruto)} INMET bruto + {len(inmet_enriched)} INMET enriquecido")

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
    plot_confusion(metrics, FIG_DIR / "fig_v3_cls_confusion.png")
    plot_f1_comparison(metrics, FIG_DIR / "fig_v3_cls_f1_comparison.png")

    report = {
        "modelos": {
            "A_sinan_only": met_a,
            "B_inmet_bruto": met_b,
            "C_inmet_enriquecido": met_c,
        },
        "melhor_modelo": best["label"],
        "f1_muito_alto_comparison": {
            m["label"]: m["f1_per_class"]["muito_alto"] for m in metrics
        },
        "conclusao": {
            "inmet_bruto_melhora_classificacao": bool(met_b["f1_macro"] > met_a["f1_macro"]),
            "inmet_enriquecido_melhora_classificacao": bool(met_c["f1_macro"] > met_a["f1_macro"]),
            "enriched_melhora_muito_alto": bool(met_c["f1_per_class"]["muito_alto"] > met_a["f1_per_class"]["muito_alto"]),
        },
    }
    with open(DATA_DIR / "29_classification_v3_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nSalvando modelos...")
    for model, name in [(model_a, "model_a_sinan_only"), (model_b, "model_b_inmet_bruto"), (model_c, "model_c_inmet_enriquecido")]:
        model.save_model(str(MODEL_DIR / f"{name}.ubj"))
        print(f"  Salvo: {name}.ubj")

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-209:")
    for m in metrics:
        print(f"  {m['label']:30s}  F1_macro={m['f1_macro']:.4f}  AUC={m['auc_macro_ovr']:.4f}  F1_muito_alto={m['f1_per_class']['muito_alto']:.4f}")
    print(f"  INMET bruto melhora classificação? {'SIM' if met_b['f1_macro'] > met_a['f1_macro'] else 'NÃO'}")
    print(f"  INMET enriquecido melhora classificação? {'SIM' if met_c['f1_macro'] > met_a['f1_macro'] else 'NÃO'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

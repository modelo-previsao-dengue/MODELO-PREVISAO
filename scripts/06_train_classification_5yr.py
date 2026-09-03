#!/usr/bin/env python3
"""US-106: XGBoost classificacao de risco (4 classes) — dataset filtrado 5yr."""

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
MODEL_DIR = BASE_DIR / "models" / "classification_5yr"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_LABELS = ["Baixo", "Médio", "Alto", "Surto"]

INMET_PREFIXES = [
    "rain_", "temp_mean_c", "temp_min_c", "temp_max_c", "temp_range_c",
    "humidity_", "pressure_", "wind_", "radiation_",
]


def is_inmet_feature(col):
    return any(col.startswith(p) or col == p for p in INMET_PREFIXES)


def train_cls(X_train, y_train, X_val, y_val, label, n_classes=4):
    scale_pos = []
    for c in range(n_classes):
        neg = (y_train != c).sum()
        pos = (y_train == c).sum()
        scale_pos.append(neg / max(pos, 1))

    sample_weights = np.ones(len(y_train))
    for c in range(n_classes):
        sample_weights[y_train == c] = scale_pos[c]

    model = xgb.XGBClassifier(
        n_estimators=800, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, tree_method="hist", objective="multi:softprob",
        num_class=n_classes, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train, sample_weight=sample_weights,
        eval_set=[(X_val, y_val)], verbose=0,
    )
    print(f"  [{label}] best_iteration={model.best_iteration}")
    return model


def eval_cls(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan

    print(f"  [{label}] Acc={acc:.4f}, F1_macro={f1_macro:.4f}, F1_weighted={f1_weighted:.4f}, AUC_macro={auc:.4f}")
    return {
        "model": label, "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4), "f1_weighted": round(float(f1_weighted), 4),
        "auc_macro": round(float(auc), 4),
    }, y_pred, y_proba


def plot_confusion_matrices(y_test, y_pred_sinan, y_pred_full, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, preds, title in zip(axes, [y_pred_sinan, y_pred_full], ["SINAN-only", "SINAN+INMET"]):
        cm = confusion_matrix(y_test, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({cm[i, j]:,})",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_pct[i, j] > 50 else "black")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_LABELS, fontsize=9)
        ax.set_yticklabels(CLASS_LABELS, fontsize=9)
        ax.set_xlabel("Predito", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.set_title(f"Matriz de Confusão — {title}", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="% da Classe Real")
    fig.suptitle("Classificação de Risco de Dengue — Dataset Filtrado 5yr", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_f1_per_class(report_sinan, report_full, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    f1_sinan = [report_sinan[str(i)]["f1-score"] for i in range(4)]
    f1_full = [report_full[str(i)]["f1-score"] for i in range(4)]
    ax.bar(x - width / 2, f1_sinan, width, label="SINAN-only", color="#1976D2")
    ax.bar(x + width / 2, f1_full, width, label="SINAN+INMET", color="#E53935")
    for i in range(4):
        ax.annotate(f"{f1_sinan[i]:.3f}", xy=(i - width / 2, f1_sinan[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{f1_full[i]:.3f}", xy=(i + width / 2, f1_full[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("F1-Score por Classe de Risco", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-106: XGBoost Classificacao de Risco (5yr)")
    print("=" * 60)

    print("\nCarregando splits...")
    train = pd.read_parquet(DATA_DIR / "train_5yr.parquet")
    val = pd.read_parquet(DATA_DIR / "val_5yr.parquet")
    test = pd.read_parquet(DATA_DIR / "test_5yr.parquet")
    for d in [train, val, test]:
        d["ibge_municipio"] = d["ibge_municipio"].astype(str)

    all_features = [c for c in train.columns if c not in ID_COLS + [TARGET, CLASS_TARGET, "notificacoes"]]
    sinan_features = [c for c in all_features if not is_inmet_feature(c)]
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  Distribuicao treino: {train[CLASS_TARGET].value_counts().sort_index().to_dict()}")

    y_train = train[CLASS_TARGET].astype(int)
    y_val = val[CLASS_TARGET].astype(int)
    y_test = test[CLASS_TARGET].astype(int)

    # SINAN-only
    print("\nTreinando SINAN-only...")
    model_sinan = train_cls(train[sinan_features], y_train, val[sinan_features], y_val, "SINAN-only")
    metrics_sinan, pred_sinan, proba_sinan = eval_cls(model_sinan, test[sinan_features], y_test, "SINAN-only")

    # SINAN+INMET
    print("\nTreinando SINAN+INMET...")
    model_full = train_cls(train[all_features], y_train, val[all_features], y_val, "SINAN+INMET")
    metrics_full, pred_full, proba_full = eval_cls(model_full, test[all_features], y_test, "SINAN+INMET")

    # Reports
    report_sinan = classification_report(y_test, pred_sinan, target_names=CLASS_LABELS, output_dict=True, zero_division=0)
    report_full = classification_report(y_test, pred_full, target_names=CLASS_LABELS, output_dict=True, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  SINAN-only:  F1_macro={metrics_sinan['f1_macro']:.4f}  AUC={metrics_sinan['auc_macro']:.4f}")
    print(f"  SINAN+INMET: F1_macro={metrics_full['f1_macro']:.4f}  AUC={metrics_full['auc_macro']:.4f}")
    delta_f1 = metrics_full["f1_macro"] - metrics_sinan["f1_macro"]
    delta_auc = metrics_full["auc_macro"] - metrics_sinan["auc_macro"]
    print(f"  Delta F1: {delta_f1:+.4f}")
    print(f"  Delta AUC: {delta_auc:+.4f}")
    print(f"{'=' * 60}")

    # Classification report text
    print("\n  Classification Report (SINAN+INMET):")
    print(classification_report(y_test, pred_full, target_names=CLASS_LABELS, zero_division=0))

    # Figures
    print("Gerando figuras...")
    plot_confusion_matrices(y_test, pred_sinan, pred_full,
                            FIG_DIR / "fig_5yr_cls_confusion_matrix.png")

    report_sinan_idx = classification_report(y_test, pred_sinan, output_dict=True, zero_division=0)
    report_full_idx = classification_report(y_test, pred_full, output_dict=True, zero_division=0)
    plot_f1_per_class(report_sinan_idx, report_full_idx,
                      FIG_DIR / "fig_5yr_cls_f1_per_class.png")

    # Save
    model_sinan.save_model(str(MODEL_DIR / "model_sinan_only.json"))
    model_full.save_model(str(MODEL_DIR / "model_sinan_inmet.json"))

    report_data = {
        "sinan_only": metrics_sinan,
        "sinan_inmet": metrics_full,
        "delta_f1_macro": round(float(delta_f1), 4),
        "delta_auc_macro": round(float(delta_auc), 4),
        "classification_report_sinan": {k: v for k, v in report_sinan.items() if k in CLASS_LABELS + ["macro avg", "weighted avg"]},
        "classification_report_full": {k: v for k, v in report_full.items() if k in CLASS_LABELS + ["macro avg", "weighted avg"]},
    }
    with open(DATA_DIR / "06_classification_report.json", "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Salvo: 06_classification_report.json")


if __name__ == "__main__":
    main()

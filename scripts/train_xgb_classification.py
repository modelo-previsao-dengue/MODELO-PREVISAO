#!/usr/bin/env python3
"""US-008: XGBoost Classificação MVP — risco de surto t+4."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY = BASE_DIR / "data" / "model_ready"
MODEL_DIR = BASE_DIR / "models" / "xgb_classification_mvp"

ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_NAMES = {0: "baixo", 1: "medio", 2: "alto", 3: "surto"}


def load_split(name):
    df = pd.read_parquet(MODEL_READY / f"{name}.parquet")
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET, CLASS_TARGET]]
    X = df[feature_cols]
    y = df[CLASS_TARGET].astype(int)
    return X, y, df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando splits...")
    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Class distribution
    print("\nDistribuição de classes:")
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = y.value_counts().sort_index()
        print(f"  {split_name}: {counts.to_dict()}")

    print("\nTreinando XGBClassifier MVP...")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # AUC
    try:
        auc_macro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = np.nan

    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()), output_dict=True)

    metrics = {
        "AUC_macro": round(auc_macro, 4) if not np.isnan(auc_macro) else None,
        "F1_macro": round(f1_macro, 4),
    }
    for cls_name in CLASS_NAMES.values():
        if cls_name in report:
            metrics[f"precision_{cls_name}"] = round(report[cls_name]["precision"], 4)
            metrics[f"recall_{cls_name}"] = round(report[cls_name]["recall"], 4)
            metrics[f"f1_{cls_name}"] = round(report[cls_name]["f1-score"], 4)

    print(f"\n=== Métricas no Teste ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values()))
    cm_df.to_csv(MODEL_DIR / "confusion_matrix.csv")
    print(f"\nMatriz de confusão:\n{cm_df}")

    # Class imbalance report
    imbalance = pd.DataFrame({
        "class": list(CLASS_NAMES.values()),
        "train_count": [int((y_train == k).sum()) for k in CLASS_NAMES],
        "test_count": [int((y_test == k).sum()) for k in CLASS_NAMES],
    })
    imbalance.to_csv(MODEL_DIR / "class_imbalance.csv", index=False)

    # Save model
    model.save_model(str(MODEL_DIR / "model.json"))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(list(CLASS_NAMES.values()))
    ax.set_yticklabels(list(CLASS_NAMES.values()))
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão (F1 macro={f1_macro:.3f})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nGráficos salvos em {MODEL_DIR}")


if __name__ == "__main__":
    main()

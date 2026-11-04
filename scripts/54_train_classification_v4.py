#!/usr/bin/env python3
"""v4-54: Classificação de risco v4 — 3 variantes × 3 sub-modelos = 9 modelos.

Target: risco_surto_t4 (4 classes: 0=baixo, 1=moderado, 2=alto, 3=surto).
Thresholds: P50/P75/P90 de notificacoes_t4 no treino, por mesorregião.
"""

import json
import time
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

import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "classification_v4"
FIG_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados_v4"
TABLE_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "tabelas"
SCHEMA_PATH = DATA_DIR / "feature_schema_v4.csv"

CLASS_TARGET = "risco_surto_t4"
CLASS_NAMES = ["baixo", "moderado", "alto", "surto"]

VARIANTES = ["a", "b", "c"]
SUB_MODELS = ["sinan_only", "inmet_bruto", "inmet_enriquecido"]

XGB_CLS_PARAMS = dict(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=-1,
    early_stopping_rounds=50,
)


def categorize_features(schema_path):
    schema = pd.read_csv(schema_path)
    sinan = schema.loc[schema["category"] == "sinan", "feature"].tolist()
    inmet_bruto = schema.loc[schema["category"] == "inmet_bruto", "feature"].tolist()
    inmet_enriched = schema.loc[
        schema["category"].isin(["inmet_lag_bio", "inmet_mm_bio", "inmet_anomalia"]), "feature"
    ].tolist()
    return sinan, inmet_bruto, inmet_enriched


def get_feature_sets(schema_path):
    sinan, inmet_bruto, inmet_enriched = categorize_features(schema_path)
    return {
        "sinan_only": sinan,
        "inmet_bruto": sinan + inmet_bruto,
        "inmet_enriquecido": sinan + inmet_bruto + inmet_enriched,
    }


def compute_sample_weights_cls(y):
    counts = np.bincount(y, minlength=4)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return weights[y]


def compute_decay_weights(train_df, var_cfg):
    lam = var_cfg.get("sample_weight_lambda")
    ref = var_cfg.get("sample_weight_ref_year")
    if lam is None or ref is None:
        return None
    return np.exp(-lam * (ref - train_df["ano"].values.astype(float)))


def train_eval_cls(train, val, test, features, label, decay_weights=None):
    t0 = time.time()
    print(f"\n  [{label}] Treinando ({len(features)} features, {len(train):,} train rows)...")

    y_train = train[CLASS_TARGET].values.astype(int)
    y_val = val[CLASS_TARGET].values.astype(int)
    y_test = test[CLASS_TARGET].values.astype(int)

    sw = compute_sample_weights_cls(y_train)
    if decay_weights is not None:
        sw = sw * decay_weights

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
    prec_rec = classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    try:
        auc_mac = roc_auc_score(y_test, pred_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_mac = 0.0

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    elapsed = time.time() - t0

    print(f"    Acc={acc:.4f}  F1_macro={f1_mac:.4f}  AUC_macro={auc_mac:.4f}  ({elapsed:.1f}s)")
    print(f"    F1: {' | '.join(f'{CLASS_NAMES[i]}={f1_per[i]:.3f}' for i in range(4))}")

    return model, {
        "label": label,
        "n_features": len(features),
        "n_train": len(train),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_w), 4),
        "auc_macro_ovr": round(float(auc_mac), 4),
        "f1_per_class": {CLASS_NAMES[i]: round(float(f1_per[i]), 4) for i in range(4)},
        "precision_recall_per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(prec_rec[CLASS_NAMES[i]]["precision"]), 4),
                "recall": round(float(prec_rec[CLASS_NAMES[i]]["recall"]), 4),
            }
            for i in range(4)
        },
        "confusion_matrix": cm.tolist(),
        "best_iteration": int(model.best_iteration),
        "train_time_s": round(elapsed, 1),
    }


def plot_confusion_best(all_metrics, out_path):
    enriq = {k: v for k, v in all_metrics.items() if "inmet_enriquecido" in k}
    best_key = max(enriq, key=lambda k: enriq[k]["f1_macro"])
    best = enriq[best_key]
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
    ax.set_title(f"Matriz de Confusão v4 — {best['label']}\n(F1_macro={best['f1_macro']:.4f})", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_f1_comparison(all_metrics, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_var = {"A": "#1976D2", "B": "#FF9800", "C": "#E53935"}

    for ax, sub_model in zip(axes, SUB_MODELS):
        sub_label = {"sinan_only": "SINAN-only", "inmet_bruto": "INMET bruto",
                     "inmet_enriquecido": "INMET enriquecido"}[sub_model]
        x = np.arange(4)
        width = 0.25

        for i, var in enumerate(["A", "B", "C"]):
            key = f"{var}_{sub_model}"
            if key not in all_metrics:
                continue
            f1s = [all_metrics[key]["f1_per_class"][c] for c in CLASS_NAMES]
            bars = ax.bar(x + i * width, f1s, width, label=f"Var {var}", color=colors_var[var])
            for bar, v in zip(bars, f1s):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x + width)
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_ylabel("F1 Score")
        ax.set_title(sub_label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("F1 por Classe de Risco — Classificação v4", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def generate_classification_table(all_metrics, out_path):
    rows = []
    for var in ["A", "B", "C"]:
        for sub in SUB_MODELS:
            key = f"{var}_{sub}"
            m = all_metrics[key]
            sub_label = {"sinan_only": "SINAN", "inmet_bruto": "INMET bruto",
                         "inmet_enriquecido": "INMET enriq."}[sub]
            f1_surto = m["f1_per_class"]["surto"]
            rows.append(
                f"    {var} & {sub_label} & {m['f1_macro']:.4f} & {m['auc_macro_ovr']:.4f} "
                f"& {m['accuracy']:.4f} & {f1_surto:.4f} \\\\"
            )

    latex = r"""\begin{table}[htbp]
\centering
\caption{Classificação de risco v4 --- 3 variantes $\times$ 3 sub-modelos}
\label{tab:v4-classification}
\small
\begin{tabular}{llcccc}
\toprule
Var. & Sub-modelo & F1\_macro & AUC\_macro & Acc & F1\_surto \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  v4-54: Classificação de Risco — 3 Variantes × 3 Sub-modelos")
    print("=" * 70)

    feature_sets = get_feature_sets(SCHEMA_PATH)
    for name, feats in feature_sets.items():
        print(f"  {name}: {len(feats)} features")

    all_metrics = {}

    for var in VARIANTES:
        var_upper = var.upper()
        var_cfg = recorte_config.load_v4(f"variante_{var}")
        var_dir = DATA_DIR / f"splits_{var}"

        print(f"\n{'=' * 70}")
        print(f"  VARIANTE {var_upper}: train={var_cfg['train'][0]}-{var_cfg['train'][-1]}")
        print(f"{'=' * 70}")

        train = pd.read_parquet(var_dir / "train.parquet")
        val = pd.read_parquet(var_dir / "val.parquet")
        test = pd.read_parquet(var_dir / "test.parquet")
        print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

        dist = train[CLASS_TARGET].value_counts().sort_index()
        print(f"  Dist treino: {dict(dist)}")

        decay_w = compute_decay_weights(train, var_cfg)

        for sub_name in SUB_MODELS:
            feats = feature_sets[sub_name]
            sub_label = {"sinan_only": "SINAN-only", "inmet_bruto": "INMET bruto",
                         "inmet_enriquecido": "INMET enriquecido"}[sub_name]
            label = f"Var {var_upper}: {sub_label}"

            model, metrics = train_eval_cls(train, val, test, feats, label, decay_weights=decay_w)

            key = f"{var_upper}_{sub_name}"
            all_metrics[key] = metrics
            model.save_model(str(MODEL_DIR / f"model_{key}_cls.ubj"))

    print(f"\n{'=' * 70}")
    print("  Gerando figuras e tabelas...")
    print(f"{'=' * 70}")

    plot_confusion_best(all_metrics, FIG_DIR / "fig_v4_confusion_matrix.png")
    plot_f1_comparison(all_metrics, FIG_DIR / "fig_v4_cls_f1_comparison.png")
    generate_classification_table(all_metrics, TABLE_DIR / "tab_v4_classification.tex")

    enriq_b = all_metrics.get("B_inmet_enriquecido", {})

    report = {
        "modelos": all_metrics,
        "melhor_modelo_f1_macro": max(all_metrics, key=lambda k: all_metrics[k]["f1_macro"]),
        "conclusao": {
            "f1_macro_B_enriquecido": enriq_b.get("f1_macro"),
            "auc_macro_B_enriquecido": enriq_b.get("auc_macro_ovr"),
            "serie_completa_melhora_classificacao": (
                all_metrics.get("B_inmet_enriquecido", {}).get("f1_macro", 0)
                > all_metrics.get("A_inmet_enriquecido", {}).get("f1_macro", 0)
            ),
        },
        "tempo_total_s": round(time.time() - t_total, 1),
    }

    report_path = DATA_DIR / "54_classification_v4_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report salvo: {report_path.name}")

    print(f"\n{'=' * 70}")
    print(f"  RESUMO v4-54 (INMET enriquecido):")
    for var in ["A", "B", "C"]:
        m = all_metrics[f"{var}_inmet_enriquecido"]
        print(f"  Var {var}: F1_macro={m['f1_macro']:.4f}  AUC={m['auc_macro_ovr']:.4f}  F1_surto={m['f1_per_class']['surto']:.4f}")
    print(f"  Série completa melhora? {'SIM' if report['conclusao']['serie_completa_melhora_classificacao'] else 'NAO'}")
    print(f"  Tempo total: {time.time() - t_total:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

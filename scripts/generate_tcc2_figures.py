"""
Generate all TCC2 figures locally and train comparison models (SARIMA, RF).
Saves figures to Overleaf directory at 300 DPI.

Usage:
    python scripts/generate_tcc2_figures.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, classification_report,
    confusion_matrix,
)
import shap

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DF_CODE = "5300108"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent.parent / "Overleaf" / "TCC2 Base FCTE UnB" / "figuras" / "resultados"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET_REG = "notificacoes_t4"
TARGET_CLS = "risco_surto_t4"
CLASS_NAMES = ["Low", "Medium", "High", "Outbreak"]

INMET_COLS = [
    "rain_sum_mm", "rain_mean_mm", "temp_mean_c", "temp_min_c", "temp_max_c",
    "humidity_mean_pct", "pressure_mean_mbar", "wind_speed_mean_ms",
    "radiation_mean_kj", "n_valid_hours", "rain_days", "rain_heavy_days",
    "temp_range_c", "low_coverage",
    "rain_sum_mm_lag_1", "rain_sum_mm_lag_2", "rain_sum_mm_lag_4", "rain_sum_mm_lag_8",
    "temp_mean_c_lag_1", "temp_mean_c_lag_2", "temp_mean_c_lag_4", "temp_mean_c_lag_8",
    "humidity_mean_pct_lag_1", "humidity_mean_pct_lag_2",
    "humidity_mean_pct_lag_4", "humidity_mean_pct_lag_8",
    "rain_sum_mm_mm4", "temp_mean_c_mm4", "humidity_mean_pct_mm4",
    "rain_heavy_days_lag2", "rain_heavy_days_lag4", "temp_range_c_lag2",
]


def load_df_data():
    df_train = pd.read_parquet(DATA_DIR / "model_ready" / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "model_ready" / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "model_ready" / "test.parquet")

    df_train = df_train[df_train["ibge_municipio"] == DF_CODE].copy()
    df_val = df_val[df_val["ibge_municipio"] == DF_CODE].copy()
    df_test = df_test[df_test["ibge_municipio"] == DF_CODE].copy()

    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    return df_trainval, df_test


def split_xy(df, target, drop_inmet=False):
    drop_cols = META_COLS + [TARGET_REG, TARGET_CLS]
    if drop_inmet:
        drop_cols = drop_cols + [c for c in INMET_COLS if c in df.columns]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].values
    return X, y


def savefig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def main():
    print("=" * 60)
    print("  TCC2 Figure Generation + Model Comparison")
    print("=" * 60)

    df_trainval, df_test = load_df_data()
    print(f"\nDF data: trainval={len(df_trainval)}, test={len(df_test)}")

    X_trainval, y_reg_trainval = split_xy(df_trainval, TARGET_REG)
    X_test, y_reg_test = split_xy(df_test, TARGET_REG)
    _, y_cls_trainval = split_xy(df_trainval, TARGET_CLS)
    _, y_cls_test = split_xy(df_test, TARGET_CLS)

    y_reg_trainval_log = np.log1p(y_reg_trainval)
    y_reg_test_log = np.log1p(y_reg_test)

    X_trainval_sinan, _ = split_xy(df_trainval, TARGET_REG, drop_inmet=True)
    X_test_sinan, _ = split_xy(df_test, TARGET_REG, drop_inmet=True)

    feature_names = list(X_trainval.columns)
    print(f"Features: {len(feature_names)} (full), {X_trainval_sinan.shape[1]} (SINAN-only)")

    # ── 1. XGBoost Baseline Regression ──
    print("\n── XGBoost Baseline Regression ──")
    xgb_base = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        tree_method="hist", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_base.fit(X_trainval, y_reg_trainval_log)
    pred_base_log = xgb_base.predict(X_test)
    pred_base_orig = np.expm1(np.maximum(pred_base_log, 0))
    r2_base_log = r2_score(y_reg_test_log, pred_base_log)
    r2_base_orig = r2_score(y_reg_test, pred_base_orig)
    print(f"  R²_log={r2_base_log:.4f}, R²_orig={r2_base_orig:.4f}")

    # ── 2. XGBoost Tuned Regression (Optuna-like params) ──
    print("\n── XGBoost Tuned Regression ──")
    tuned_params = {
        "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "n_estimators": 800,
        "tree_method": "hist", "random_state": RANDOM_STATE, "verbosity": 0,
    }
    xgb_tuned = xgb.XGBRegressor(**tuned_params)
    xgb_tuned.fit(X_trainval, y_reg_trainval_log)
    pred_tuned_log = xgb_tuned.predict(X_test)
    pred_tuned_orig = np.expm1(np.maximum(pred_tuned_log, 0))
    r2_tuned_log = r2_score(y_reg_test_log, pred_tuned_log)
    r2_tuned_orig = r2_score(y_reg_test, pred_tuned_orig)
    mae_tuned_log = mean_absolute_error(y_reg_test_log, pred_tuned_log)
    print(f"  R²_log={r2_tuned_log:.4f}, R²_orig={r2_tuned_orig:.4f}")

    # ── 3. XGBoost SINAN-only ──
    print("\n── XGBoost SINAN-only ──")
    xgb_sinan = xgb.XGBRegressor(**tuned_params)
    xgb_sinan.fit(X_trainval_sinan, y_reg_trainval_log)
    pred_sinan_log = xgb_sinan.predict(X_test_sinan)
    pred_sinan_orig = np.expm1(np.maximum(pred_sinan_log, 0))
    r2_sinan_log = r2_score(y_reg_test_log, pred_sinan_log)
    r2_sinan_orig = r2_score(y_reg_test, pred_sinan_orig)
    print(f"  R²_log={r2_sinan_log:.4f}, R²_orig={r2_sinan_orig:.4f}")

    # ── 4. Random Forest Regression ──
    print("\n── Random Forest Regression ──")
    rf_reg = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_reg.fit(X_trainval, y_reg_trainval_log)
    pred_rf_log = rf_reg.predict(X_test)
    pred_rf_orig = np.expm1(np.maximum(pred_rf_log, 0))
    r2_rf_log = r2_score(y_reg_test_log, pred_rf_log)
    r2_rf_orig = r2_score(y_reg_test, pred_rf_orig)
    mae_rf_log = mean_absolute_error(y_reg_test_log, pred_rf_log)
    print(f"  R²_log={r2_rf_log:.4f}, R²_orig={r2_rf_orig:.4f}")

    # ── 5. SARIMA Regression ──
    print("\n── SARIMA Regression ──")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_train_series = pd.Series(y_reg_trainval_log, name="notif_log")
        sarima = SARIMAX(
            y_train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit = sarima.fit(disp=False, maxiter=200)
        pred_sarima_log = sarima_fit.forecast(steps=len(y_reg_test_log))
        pred_sarima_orig = np.expm1(np.maximum(pred_sarima_log.values, 0))
        r2_sarima_log = r2_score(y_reg_test_log, pred_sarima_log.values)
        r2_sarima_orig = r2_score(y_reg_test, pred_sarima_orig)
        mae_sarima_log = mean_absolute_error(y_reg_test_log, pred_sarima_log.values)
        print(f"  R²_log={r2_sarima_log:.4f}, R²_orig={r2_sarima_orig:.4f}")
        sarima_ok = True
    except Exception as e:
        print(f"  SARIMA failed: {e}")
        sarima_ok = False
        r2_sarima_log = r2_sarima_orig = mae_sarima_log = float("nan")
        pred_sarima_log = np.full_like(y_reg_test_log, np.nan)
        pred_sarima_orig = np.full_like(y_reg_test, np.nan)

    # ── 6. XGBoost Classification ──
    print("\n── XGBoost Classification (Baseline + Tuned) ──")
    xgb_cls_base = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, objective="multi:softprob", num_class=4,
        tree_method="hist", random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_cls_base.fit(X_trainval, y_cls_trainval)
    pred_cls_base = xgb_cls_base.predict(X_test)
    pred_proba_base = xgb_cls_base.predict_proba(X_test)
    f1_cls_base = f1_score(y_cls_test, pred_cls_base, average="macro", zero_division=0)

    cls_tuned_params = {
        "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "n_estimators": 800,
        "objective": "multi:softprob", "num_class": 4,
        "tree_method": "hist", "random_state": RANDOM_STATE, "verbosity": 0,
    }
    xgb_cls_tuned = xgb.XGBClassifier(**cls_tuned_params)
    xgb_cls_tuned.fit(X_trainval, y_cls_trainval)
    pred_cls_tuned = xgb_cls_tuned.predict(X_test)
    pred_proba_tuned = xgb_cls_tuned.predict_proba(X_test)
    f1_cls_tuned = f1_score(y_cls_test, pred_cls_tuned, average="macro", zero_division=0)
    print(f"  Baseline F1={f1_cls_base:.4f}, Tuned F1={f1_cls_tuned:.4f}")

    # ── 7. Random Forest Classification ──
    print("\n── Random Forest Classification ──")
    rf_cls = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_cls.fit(X_trainval, y_cls_trainval)
    pred_cls_rf = rf_cls.predict(X_test)
    f1_cls_rf = f1_score(y_cls_test, pred_cls_rf, average="macro", zero_division=0)
    print(f"  RF F1={f1_cls_rf:.4f}")

    # ═══════════════════════════════════════════════════════
    #  FIGURES
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Generating Figures")
    print("=" * 60)

    # ── Fig 1: Scatter plot tuned regression (log + orig) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.scatter(y_reg_test_log, pred_tuned_log, alpha=0.6, s=40, c="#4CAF50", edgecolors="white", linewidth=0.5)
    lims = [min(y_reg_test_log.min(), pred_tuned_log.min()) - 0.5,
            max(y_reg_test_log.max(), pred_tuned_log.max()) + 0.5]
    ax.plot(lims, lims, "--", color="#E53935", linewidth=2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual (log scale)"); ax.set_ylabel("Predicted (log scale)")
    ax.set_title("XGBoost Tuned — Log Scale"); ax.legend(); ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(y_reg_test, pred_tuned_orig, alpha=0.6, s=40, c="#FF9800", edgecolors="white", linewidth=0.5)
    lims_orig = [0, max(y_reg_test.max(), pred_tuned_orig.max()) * 1.05]
    ax.plot(lims_orig, lims_orig, "--", color="#E53935", linewidth=2, label="Perfect prediction")
    ax.set_xlim(lims_orig); ax.set_ylim(lims_orig)
    ax.set_xlabel("Actual (original scale)"); ax.set_ylabel("Predicted (original scale)")
    ax.set_title("XGBoost Tuned — Original Scale"); ax.legend(); ax.set_aspect("equal")
    fig.suptitle("Predicted vs Actual — XGBoost Tuned Regression", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_scatter_tuned")

    # ── Fig 2: Time series real vs predicted ──
    weeks = np.arange(len(y_reg_test))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.plot(weeks, y_reg_test_log, "o-", color="#1976D2", label="Actual", markersize=4, linewidth=1.5)
    ax.plot(weeks, pred_tuned_log, "s--", color="#E53935", label="XGBoost Tuned", markersize=4, linewidth=1.5, alpha=0.8)
    ax.fill_between(weeks, y_reg_test_log, pred_tuned_log, alpha=0.15, color="#E53935")
    ax.set_ylabel("log1p(notifications)"); ax.set_title("Test Period — Log Scale"); ax.legend()

    ax = axes[1]
    ax.plot(weeks, y_reg_test, "o-", color="#1976D2", label="Actual", markersize=4, linewidth=1.5)
    ax.plot(weeks, pred_tuned_orig, "s--", color="#E53935", label="XGBoost Tuned", markersize=4, linewidth=1.5, alpha=0.8)
    ax.fill_between(weeks, y_reg_test, pred_tuned_orig, alpha=0.15, color="#E53935")
    ax.set_ylabel("Notifications"); ax.set_xlabel("Test week index")
    ax.set_title("Test Period — Original Scale"); ax.legend()
    fig.suptitle("Actual vs Predicted Over Test Period", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_timeseries_tuned")

    # ── Fig 3: Ablation SINAN-only vs SINAN+INMET ──
    metrics_to_plot = ["MAE (log)", "R² (log)", "R² (orig)"]
    sinan_vals = [
        mean_absolute_error(y_reg_test_log, pred_sinan_log),
        r2_sinan_log, r2_sinan_orig
    ]
    inmet_vals = [
        mae_tuned_log, r2_tuned_log, r2_tuned_orig
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label, sv, iv in zip(axes, metrics_to_plot, sinan_vals, inmet_vals):
        colors = ["#78909C", "#26A69A"]
        bars = ax.bar(["SINAN-only", "SINAN+INMET"], [sv, iv], color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, [sv, iv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(abs(sv), abs(iv)),
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_ylabel(label); ax.set_title(label)
    fig.suptitle("Ablation: SINAN-only vs SINAN+INMET", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_ablation")

    # ── Fig 4: Confusion matrices (baseline vs tuned) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cm_base = confusion_matrix(y_cls_test, pred_cls_base)
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Baseline Classification (F1={f1_cls_base:.3f})")

    cm_tuned = confusion_matrix(y_cls_test, pred_cls_tuned)
    sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    axes[1].set_title(f"Tuned Classification (F1={f1_cls_tuned:.3f})")
    fig.suptitle("Confusion Matrices — Baseline vs Tuned XGBoost", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_confusion_matrices")

    # ── Fig 5: SHAP summary — regression ──
    print("\n  Computing SHAP values (regression)...")
    explainer_reg = shap.TreeExplainer(xgb_tuned)
    shap_values_reg = explainer_reg.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values_reg, X_test, max_display=15, show=False)
    plt.title("SHAP Feature Importance — XGBoost Regression (Tuned)")
    plt.tight_layout()
    savefig(plt.gcf(), "fig_shap_regression")

    # ── Fig 6: SHAP summary — classification ──
    print("  Computing SHAP values (classification)...")
    explainer_cls = shap.TreeExplainer(xgb_cls_tuned)
    shap_values_cls = explainer_cls.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(12, 8))
    if isinstance(shap_values_cls, list):
        shap.summary_plot(shap_values_cls[3], X_test, max_display=15, show=False)
        plt.title("SHAP Feature Importance — XGBoost Classification (Outbreak class)")
    else:
        shap.summary_plot(shap_values_cls, X_test, max_display=15, show=False)
        plt.title("SHAP Feature Importance — XGBoost Classification")
    plt.tight_layout()
    savefig(plt.gcf(), "fig_shap_classification")

    # ── Fig 7: Distribution shift analysis ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(y_reg_trainval, bins=50, alpha=0.7, color="#1976D2", label=f"Train (max={y_reg_trainval.max():.0f})", density=True)
    ax.hist(y_reg_test, bins=50, alpha=0.7, color="#E53935", label=f"Test (max={y_reg_test.max():.0f})", density=True)
    ax.set_xlabel("Notifications"); ax.set_ylabel("Density")
    ax.set_title("Original Scale"); ax.legend()

    ax = axes[1]
    ax.hist(y_reg_trainval_log, bins=50, alpha=0.7, color="#1976D2", label=f"Train (max={y_reg_trainval_log.max():.2f})", density=True)
    ax.hist(y_reg_test_log, bins=50, alpha=0.7, color="#E53935", label=f"Test (max={y_reg_test_log.max():.2f})", density=True)
    ax.set_xlabel("log1p(Notifications)"); ax.set_ylabel("Density")
    ax.set_title("Log Scale"); ax.legend()
    fig.suptitle("Distribution Shift: Train vs Test (2024 Unprecedented Outbreak)", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_distribution_shift")

    # ── Fig 8: Model comparison bar chart ──
    models = ["XGB\nBaseline", "XGB\nTuned", "XGB\nSINAN-only", "Random\nForest"]
    r2_logs = [r2_base_log, r2_tuned_log, r2_sinan_log, r2_rf_log]
    r2_origs = [r2_base_orig, r2_tuned_orig, r2_sinan_orig, r2_rf_orig]
    if sarima_ok:
        models.append("SARIMA")
        r2_logs.append(r2_sarima_log)
        r2_origs.append(r2_sarima_orig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#90CAF9", "#4CAF50", "#78909C", "#FF9800", "#9C27B0"][:len(models)]

    ax = axes[0]
    bars = ax.bar(models, r2_logs, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, r2_logs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("R² (log scale)"); ax.set_title("Regression: R² (Log Scale)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax = axes[1]
    bars = ax.bar(models, r2_origs, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, r2_origs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("R² (original scale)"); ax.set_title("Regression: R² (Original Scale)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    fig.suptitle("Model Comparison — Regression Performance (DF, t+4)", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_model_comparison_regression")

    # ── Fig 9: Classification comparison ──
    cls_models = ["XGB Baseline", "XGB Tuned", "Random Forest"]
    cls_f1s = [f1_cls_base, f1_cls_tuned, f1_cls_rf]
    cls_colors = ["#90CAF9", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cls_models, cls_f1s, color=cls_colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, cls_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("F1 Macro"); ax.set_title("Classification Comparison — F1 Macro (DF, 4 Risk Levels)")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    savefig(fig, "fig_model_comparison_classification")

    # ── Fig 10: Walk-forward CV ──
    print("\n  Running walk-forward CV...")
    df_wf = df_trainval.copy()
    df_wf["ano"] = df_wf["ano"].astype(int)
    available_years = sorted(df_wf["ano"].unique())
    wf_test_years = available_years[4:]

    wf_reg, wf_cls = [], []
    for test_year in wf_test_years:
        train_years = [y for y in available_years if y < test_year]
        df_tr = df_wf[df_wf["ano"].isin(train_years)]
        df_te = df_wf[df_wf["ano"] == test_year]
        if len(df_te) == 0:
            continue

        X_tr, y_tr = split_xy(df_tr, TARGET_REG)
        X_te, y_te = split_xy(df_te, TARGET_REG)
        y_tr_log = np.log1p(y_tr)
        y_te_log = np.log1p(y_te)

        m = xgb.XGBRegressor(**tuned_params)
        m.fit(X_tr, y_tr_log)
        p = m.predict(X_te)
        wf_reg.append({
            "test_year": test_year,
            "MAE_log": mean_absolute_error(y_te_log, p),
            "R2_log": r2_score(y_te_log, p),
        })

        _, y_tr_c = split_xy(df_tr, TARGET_CLS)
        _, y_te_c = split_xy(df_te, TARGET_CLS)
        mc = xgb.XGBClassifier(**cls_tuned_params)
        mc.fit(X_tr, y_tr_c)
        pc = mc.predict(X_te)
        wf_cls.append({
            "test_year": test_year,
            "F1_macro": f1_score(y_te_c, pc, average="macro", zero_division=0),
        })

    wf_df_reg = pd.DataFrame(wf_reg)
    wf_df_cls = pd.DataFrame(wf_cls)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.plot(wf_df_reg["test_year"], wf_df_reg["MAE_log"], "o-", color="#1976D2", linewidth=2, markersize=8, label="MAE (log)")
    ax_r2 = ax.twinx()
    ax_r2.plot(wf_df_reg["test_year"], wf_df_reg["R2_log"], "s--", color="#E53935", linewidth=2, markersize=8, label="R² (log)")
    ax.set_xlabel("Test Year"); ax.set_ylabel("MAE (log)", color="#1976D2")
    ax_r2.set_ylabel("R² (log)", color="#E53935")
    ax.set_title("Walk-Forward CV: Regression")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax = axes[1]
    ax.plot(wf_df_cls["test_year"], wf_df_cls["F1_macro"], "o-", color="#4CAF50", linewidth=2, markersize=8, label="F1 (macro)")
    ax.set_xlabel("Test Year"); ax.set_ylabel("F1 Macro")
    ax.set_title("Walk-Forward CV: Classification"); ax.legend()
    ax.set_ylim(0, 1.05)

    fig.suptitle("Walk-Forward Cross-Validation: Temporal Stability", fontsize=15, y=1.02)
    fig.tight_layout()
    savefig(fig, "fig_walkforward_cv")

    # ── Fig 11: Multi-model time series comparison ──
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(weeks, y_reg_test_log, "o-", color="#1976D2", label="Actual", markersize=5, linewidth=2)
    ax.plot(weeks, pred_tuned_log, "s--", color="#4CAF50", label=f"XGBoost Tuned (R²={r2_tuned_log:.3f})", markersize=4, linewidth=1.5, alpha=0.8)
    ax.plot(weeks, pred_rf_log, "^--", color="#FF9800", label=f"Random Forest (R²={r2_rf_log:.3f})", markersize=4, linewidth=1.5, alpha=0.8)
    if sarima_ok:
        ax.plot(weeks, pred_sarima_log.values, "d--", color="#9C27B0", label=f"SARIMA (R²={r2_sarima_log:.3f})", markersize=4, linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Test Week Index"); ax.set_ylabel("log1p(Notifications)")
    ax.set_title("Multi-Model Comparison — Test Period (Log Scale)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig_multimodel_timeseries")

    # ── Save metrics summary ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Regression (DF, test 2023-2026):")
    print(f"    XGBoost Baseline:   R²_log={r2_base_log:.4f}  R²_orig={r2_base_orig:.4f}")
    print(f"    XGBoost Tuned:      R²_log={r2_tuned_log:.4f}  R²_orig={r2_tuned_orig:.4f}")
    print(f"    XGBoost SINAN-only: R²_log={r2_sinan_log:.4f}  R²_orig={r2_sinan_orig:.4f}")
    print(f"    Random Forest:      R²_log={r2_rf_log:.4f}  R²_orig={r2_rf_orig:.4f}")
    if sarima_ok:
        print(f"    SARIMA(1,1,1)(1,1,1,52): R²_log={r2_sarima_log:.4f}  R²_orig={r2_sarima_orig:.4f}")

    print(f"\n  Classification (DF, 4 risk levels):")
    print(f"    XGBoost Baseline: F1_macro={f1_cls_base:.4f}")
    print(f"    XGBoost Tuned:    F1_macro={f1_cls_tuned:.4f}")
    print(f"    Random Forest:    F1_macro={f1_cls_rf:.4f}")

    metrics_csv = pd.DataFrame({
        "Model": ["XGB_Baseline", "XGB_Tuned", "XGB_SINAN_only", "RandomForest", "SARIMA"],
        "R2_log": [r2_base_log, r2_tuned_log, r2_sinan_log, r2_rf_log, r2_sarima_log],
        "R2_orig": [r2_base_orig, r2_tuned_orig, r2_sinan_orig, r2_rf_orig, r2_sarima_orig],
    })
    metrics_csv.to_csv(FIG_DIR / "metrics_comparison.csv", index=False)

    total_figs = len(list(FIG_DIR.glob("*.png")))
    total_size = sum(f.stat().st_size for f in FIG_DIR.glob("*.png")) / 1024 / 1024
    print(f"\n  Total: {total_figs} figures, {total_size:.1f} MB")
    print(f"  Location: {FIG_DIR}")


if __name__ == "__main__":
    main()

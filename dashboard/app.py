"""
Streamlit Dashboard for Dengue Forecast Visualization.

Usage:
    streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_LABELS = {0: "Baixo", 1: "Medio", 2: "Alto", 3: "Surto"}
RISK_COLORS = {"Baixo": "#4CAF50", "Medio": "#FFC107", "Alto": "#FF9800", "Surto": "#F44336"}

UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}

META_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]
TARGET_REG = "notificacoes_t4"
TARGET_CLS = "risco_surto_t4"

st.set_page_config(
    page_title="Dengue Forecast Dashboard",
    page_icon="🦟",
    layout="wide",
)


@st.cache_data
def load_test_data():
    path = DATA_DIR / "model_ready" / "test.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["uf"] = df["ibge_municipio"].str[:2].map(UF_MAP)
    return df


@st.cache_resource
def load_models():
    models = {}
    reg_path = MODELS_DIR / "nacional" / "xgb_reg_nacional.ubj"
    cls_path = MODELS_DIR / "nacional" / "xgb_cls_nacional.ubj"

    if reg_path.exists():
        m = xgb.XGBRegressor()
        m.load_model(str(reg_path))
        models["reg"] = m
    if cls_path.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(cls_path))
        models["cls"] = m
    return models


@st.cache_data
def load_nacional_metrics():
    path = MODELS_DIR / "nacional" / "metrics_nacional.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_uf_metrics():
    path = MODELS_DIR / "nacional" / "metrics_por_uf.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def get_features(df):
    drop = [c for c in META_COLS + [TARGET_REG, TARGET_CLS, "uf", "pred_log", "pred_orig", "pred_cls", "risco"] if c in df.columns]
    return df.drop(columns=drop, errors="ignore")


def main():
    st.title("Dashboard de Previsao de Dengue")
    st.markdown("Modelo XGBoost Nacional — Previsao de notificacoes e nivel de risco")

    df_test = load_test_data()
    models = load_models()
    metrics = load_nacional_metrics()
    uf_metrics = load_uf_metrics()

    if df_test is None:
        st.error("Dados de teste nao encontrados em data/model_ready/test.parquet")
        return

    if not models:
        st.warning("Modelos nao encontrados em models/nacional/. Execute scripts/train_xgboost_nacional.py primeiro.")

    # ── Sidebar ──
    st.sidebar.header("Filtros")
    ufs_available = sorted(df_test["uf"].dropna().unique())
    selected_uf = st.sidebar.selectbox("Estado (UF)", ["Todos"] + ufs_available)
    anos_available = sorted(df_test["ano"].unique())
    selected_ano = st.sidebar.selectbox("Ano", ["Todos"] + anos_available)

    # filter data
    df_filtered = df_test.copy()
    if selected_uf != "Todos":
        df_filtered = df_filtered[df_filtered["uf"] == selected_uf]
    if selected_ano != "Todos":
        df_filtered = df_filtered[df_filtered["ano"] == selected_ano]

    # ── Metrics Overview ──
    st.header("Metricas do Modelo Nacional")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² (log)", f"{metrics['R2_log']:.4f}")
        col2.metric("R² (original)", f"{metrics['R2_orig']:.4f}")
        col3.metric("MAE (orig)", f"{metrics.get('MAE_orig', 0):.1f}")
        col4.metric("Municipios", f"{metrics.get('n_municipios', 5565):,}")
    else:
        st.info("Metricas nacionais nao disponiveis. Execute o treinamento primeiro.")

    # ── Predictions ──
    if models and "reg" in models:
        st.header("Previsoes")

        X = get_features(df_filtered)
        pred_log = models["reg"].predict(X)
        pred_orig = np.expm1(np.maximum(pred_log, 0))
        df_filtered = df_filtered.copy()
        df_filtered["pred_orig"] = pred_orig

        if "cls" in models:
            pred_cls = models["cls"].predict(X)
            df_filtered["pred_cls"] = pred_cls
            df_filtered["risco"] = df_filtered["pred_cls"].map(CLASS_LABELS)
        else:
            conditions = [
                pred_orig <= 1,
                pred_orig <= 5,
                pred_orig <= 20,
                pred_orig > 20,
            ]
            levels = [0, 1, 2, 3]
            df_filtered["pred_cls"] = np.select(conditions, levels, default=0)
            df_filtered["risco"] = df_filtered["pred_cls"].map(CLASS_LABELS)

        # ── Alerts panel ──
        st.subheader("Alertas de Surto")
        surto_df = df_filtered[df_filtered["pred_cls"] >= 2].copy()
        surto_df = surto_df.nlargest(50, "pred_orig")

        if len(surto_df) > 0:
            alert_cols = ["ibge_municipio", "uf", "ano", "semana_epidemiologica", "pred_orig", "risco"]
            display_df = surto_df[alert_cols].copy()
            display_df.columns = ["IBGE", "UF", "Ano", "Semana", "Notif. Previstas", "Risco"]
            display_df["Notif. Previstas"] = display_df["Notif. Previstas"].round(0).astype(int)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum alerta de surto para os filtros selecionados.")

        # ── Time series for selected municipality ──
        st.subheader("Serie Temporal por Municipio")

        top_munic = df_filtered.groupby("ibge_municipio")[TARGET_REG].sum().nlargest(20).index.tolist()
        selected_munic = st.selectbox("Municipio (IBGE)", top_munic)

        if selected_munic:
            df_munic = df_filtered[df_filtered["ibge_municipio"] == selected_munic].sort_values(
                ["ano", "semana_epidemiologica"]
            )
            df_munic["periodo"] = df_munic["ano"].astype(str) + "-S" + df_munic["semana_epidemiologica"].astype(str).str.zfill(2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_munic["periodo"], y=df_munic[TARGET_REG],
                mode="lines", name="Real", line=dict(color="#1976D2", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=df_munic["periodo"], y=df_munic["pred_orig"],
                mode="lines", name="Previsto", line=dict(color="#E53935", width=2, dash="dash"),
            ))
            fig.update_layout(
                title=f"Notificacoes Reais vs Previstas — {selected_munic}",
                xaxis_title="Periodo",
                yaxis_title="Notificacoes",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Risk distribution ──
        st.subheader("Distribuicao de Risco")
        risk_counts = df_filtered["risco"].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            title="Distribuicao de Niveis de Risco",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ── Per-UF Performance ──
    if uf_metrics is not None:
        st.header("Desempenho por UF")
        uf_sorted = uf_metrics.sort_values("R2_log", ascending=False)
        fig_uf = px.bar(
            uf_sorted, x="UF", y="R2_log",
            color="R2_log",
            color_continuous_scale=["red", "yellow", "green"],
            title="R² (log) por Estado",
            labels={"R2_log": "R² (log scale)", "UF": "Estado"},
        )
        fig_uf.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_uf, use_container_width=True)

    # ── Raw data explorer ──
    st.header("Explorador de Dados")
    st.write(f"Total de linhas (filtradas): {len(df_filtered):,}")
    if st.checkbox("Mostrar dados brutos"):
        st.dataframe(df_filtered.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

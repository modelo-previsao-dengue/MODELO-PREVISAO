"""
FastAPI for dengue prediction — serves XGBoost models (DF and Nacional).

Usage:
    uvicorn api.main:app --reload --port 8000
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="Dengue Forecast API",
    description="Previsão de notificações e nível de risco de dengue por município",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_LABELS = {0: "baixo", 1: "medio", 2: "alto", 3: "surto"}
RISK_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#FF9800", 3: "#F44336"}

models: dict = {}
df_test: Optional[pd.DataFrame] = None
meta_cols = ["ibge_municipio", "ano", "semana_epidemiologica"]
target_reg = "notificacoes_t4"
target_cls = "risco_surto_t4"


def load_models():
    global models, df_test

    nac_reg_path = MODELS_DIR / "nacional" / "xgb_reg_nacional.ubj"
    nac_cls_path = MODELS_DIR / "nacional" / "xgb_cls_nacional.ubj"

    if nac_reg_path.exists():
        m = xgb.XGBRegressor()
        m.load_model(str(nac_reg_path))
        models["reg_nacional"] = m

    if nac_cls_path.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(nac_cls_path))
        models["cls_nacional"] = m

    test_path = DATA_DIR / "model_ready" / "test.parquet"
    if test_path.exists():
        df_test = pd.read_parquet(test_path)


@app.on_event("startup")
def startup():
    load_models()


def get_features(df):
    drop = [c for c in meta_cols + [target_reg, target_cls] if c in df.columns]
    return df.drop(columns=drop)


class PredictionResponse(BaseModel):
    ibge_municipio: str
    ano: int
    semana_epidemiologica: int
    notificacoes_previstas: float
    risco: str
    risco_nivel: int
    risco_cor: str


class AlertResponse(BaseModel):
    ibge_municipio: str
    municipio_nome: Optional[str] = None
    uf: Optional[str] = None
    semana: int
    ano: int
    notificacoes_previstas: float
    risco: str
    risco_nivel: int


class MunicipioStats(BaseModel):
    ibge_municipio: str
    total_test_rows: int
    media_notificacoes: float
    max_notificacoes: float
    pct_surto: float


UF_MAP = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}


@app.get("/")
def root():
    return {
        "service": "Dengue Forecast API",
        "models_loaded": list(models.keys()),
        "test_rows": len(df_test) if df_test is not None else 0,
    }


@app.get("/predict/{ibge_municipio}", response_model=list[PredictionResponse])
def predict(
    ibge_municipio: str,
    ano: Optional[int] = None,
    semana: Optional[int] = None,
):
    if df_test is None:
        raise HTTPException(500, "Test data not loaded")
    if "reg_nacional" not in models:
        raise HTTPException(500, "Nacional regression model not loaded")

    mask = df_test["ibge_municipio"] == ibge_municipio
    if ano:
        mask &= df_test["ano"] == ano
    if semana:
        mask &= df_test["semana_epidemiologica"] == semana

    subset = df_test[mask]
    if len(subset) == 0:
        raise HTTPException(404, f"No data for municipality {ibge_municipio}")

    X = get_features(subset)
    pred_log = models["reg_nacional"].predict(X)
    pred_orig = np.expm1(np.maximum(pred_log, 0))

    if "cls_nacional" in models:
        pred_cls = models["cls_nacional"].predict(X)
    else:
        pred_cls = np.where(pred_orig <= 1, 0, np.where(pred_orig <= 5, 1, np.where(pred_orig <= 20, 2, 3)))

    results = []
    for i, (_, row) in enumerate(subset.iterrows()):
        nivel = int(pred_cls[i])
        results.append(PredictionResponse(
            ibge_municipio=ibge_municipio,
            ano=int(row["ano"]),
            semana_epidemiologica=int(row["semana_epidemiologica"]),
            notificacoes_previstas=round(float(pred_orig[i]), 1),
            risco=CLASS_LABELS[nivel],
            risco_nivel=nivel,
            risco_cor=RISK_COLORS[nivel],
        ))

    return results


@app.get("/alerts", response_model=list[AlertResponse])
def alerts(
    min_risk: int = Query(2, ge=0, le=3, description="Minimum risk level (0=baixo, 3=surto)"),
    ano: Optional[int] = None,
    semana: Optional[int] = None,
    uf: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    if df_test is None or "reg_nacional" not in models:
        raise HTTPException(500, "Models or data not loaded")

    subset = df_test.copy()
    if ano:
        subset = subset[subset["ano"] == ano]
    if semana:
        subset = subset[subset["semana_epidemiologica"] == semana]
    if uf:
        subset = subset[subset["ibge_municipio"].str[:2].map(UF_MAP) == uf.upper()]

    if len(subset) == 0:
        return []

    X = get_features(subset)
    pred_log = models["reg_nacional"].predict(X)
    pred_orig = np.expm1(np.maximum(pred_log, 0))
    if "cls_nacional" in models:
        pred_cls = models["cls_nacional"].predict(X)
    else:
        pred_cls = np.where(pred_orig <= 1, 0, np.where(pred_orig <= 5, 1, np.where(pred_orig <= 20, 2, 3)))

    subset = subset.copy()
    subset["pred_cls"] = pred_cls
    subset["pred_orig"] = pred_orig

    high_risk = subset[subset["pred_cls"] >= min_risk].nlargest(limit, "pred_orig")

    results = []
    for _, row in high_risk.iterrows():
        nivel = int(row["pred_cls"])
        uf_code = UF_MAP.get(str(row["ibge_municipio"])[:2], "??")
        results.append(AlertResponse(
            ibge_municipio=str(row["ibge_municipio"]),
            uf=uf_code,
            semana=int(row["semana_epidemiologica"]),
            ano=int(row["ano"]),
            notificacoes_previstas=round(float(row["pred_orig"]), 1),
            risco=CLASS_LABELS[nivel],
            risco_nivel=nivel,
        ))

    return results


@app.get("/municipios", response_model=list[MunicipioStats])
def municipios(
    uf: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    if df_test is None:
        raise HTTPException(500, "Test data not loaded")

    subset = df_test
    if uf:
        subset = subset[subset["ibge_municipio"].str[:2].map(UF_MAP) == uf.upper()]

    stats = subset.groupby("ibge_municipio").agg(
        total_test_rows=(target_reg, "count"),
        media_notificacoes=(target_reg, "mean"),
        max_notificacoes=(target_reg, "max"),
    ).reset_index()

    if target_cls in subset.columns:
        surto_pct = subset.groupby("ibge_municipio")[target_cls].apply(
            lambda x: (x >= 3).mean()
        ).reset_index(name="pct_surto")
        stats = stats.merge(surto_pct, on="ibge_municipio", how="left")
    else:
        stats["pct_surto"] = 0.0

    stats = stats.nlargest(limit, "max_notificacoes")
    return [MunicipioStats(**row) for row in stats.to_dict("records")]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {k: True for k in models},
        "data_loaded": df_test is not None,
    }

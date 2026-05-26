from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title='SINAN TCC2 API', version='1.0.0')

GOLD_ROOT = Path(os.getenv('SINAN_GOLD_ROOT', 'MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/official_dense'))
FEATURE_CATALOG = Path(os.getenv('SINAN_FEATURE_CATALOG', 'MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/feature_catalog.csv'))


@lru_cache(maxsize=1)
def load_all_gold() -> pd.DataFrame:
    parts = sorted(GOLD_ROOT.glob('year=*/*.parquet'))
    if not parts:
        raise HTTPException(status_code=500, detail='Camada gold nao encontrada.')
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


@lru_cache(maxsize=1)
def load_feature_catalog() -> pd.DataFrame:
    if not FEATURE_CATALOG.exists():
        raise HTTPException(status_code=500, detail='Catalogo de features nao encontrado.')
    return pd.read_csv(FEATURE_CATALOG)


@app.get('/health')
def health() -> dict[str, object]:
    parts = sorted(GOLD_ROOT.glob('year=*/*.parquet'))
    return {
        'status': 'ok',
        'gold_root': str(GOLD_ROOT),
        'feature_catalog': str(FEATURE_CATALOG),
        'gold_parts': len(parts),
    }

@app.get('/v1/features/catalog')
def feature_catalog() -> list[dict[str, object]]:
    return load_feature_catalog().to_dict(orient='records')

@app.get('/v1/series/{ibge_municipio}')
def municipality_series(ibge_municipio: str, start: Optional[str] = Query(None), end: Optional[str] = Query(None)) -> list[dict[str, object]]:
    df = load_all_gold()
    df['ibge_municipio'] = df['ibge_municipio'].astype(str)
    out = df.loc[df['ibge_municipio'] == str(ibge_municipio)].copy()
    if start is not None:
        out = out.loc[out['ano_semana'].astype(str) >= str(start)]
    if end is not None:
        out = out.loc[out['ano_semana'].astype(str) <= str(end)]
    if out.empty:
        raise HTTPException(status_code=404, detail='Municipio nao encontrado na camada gold.')
    out = out.sort_values(['ano', 'semana_epidemiologica'])
    return out.to_dict(orient='records')

@app.get('/v1/top-weeks')
def top_weeks(year: Optional[int] = Query(None), limit: int = Query(20, ge=1, le=100)) -> list[dict[str, object]]:
    df = load_all_gold()
    if year is not None:
        df = df.loc[df['ano'] == int(year)]
    grouped = df.groupby(['ano_semana', 'ano', 'semana_epidemiologica'], as_index=False)['notificacoes'].sum()
    grouped = grouped.sort_values('notificacoes', ascending=False).head(limit)
    return grouped.to_dict(orient='records')

#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    import psycopg
except Exception as exc:
    raise SystemExit('psycopg nao instalado: ' + str(exc))

gold_root = Path(os.environ['SINAN_GOLD_ROOT'])
conn = psycopg.connect(
    host=os.environ['SUPABASE_DB_HOST'],
    port=os.environ.get('SUPABASE_DB_PORT', '5432'),
    dbname=os.environ['SUPABASE_DB_NAME'],
    user=os.environ['SUPABASE_DB_USER'],
    password=os.environ['SUPABASE_DB_PASSWORD'],
)
parts = sorted(gold_root.glob('year=*/*.parquet'))
with conn, conn.cursor() as cur:
    for part in parts:
        df = pd.read_parquet(part)
        rows = [tuple(item) for item in df.itertuples(index=False, name=None)]
        if not rows:
            continue
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        sql = f'INSERT INTO sinan_gold_weekly ({columns}) VALUES ({placeholders}) ON CONFLICT (ibge_municipio, ano_semana) DO NOTHING'
        cur.executemany(sql, rows)

print('Carga concluida.')

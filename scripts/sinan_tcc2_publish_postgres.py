#!/usr/bin/env python3
"""
Carrega a camada Gold oficial, catalogo de features e manifesto da execucao em PostgreSQL/Supabase.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "sinan"
DEFAULT_VERSION = "sinan_tcc2_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publica a trilha SINAN TCC2 em PostgreSQL/Supabase")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--truncate", action="store_true")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def get_connection():
    try:
        import psycopg
    except Exception as exc:
        raise SystemExit("psycopg nao instalado: " + str(exc))

    return psycopg.connect(
        host=require_env("SUPABASE_DB_HOST"),
        port=os.getenv("SUPABASE_DB_PORT", "5432"),
        dbname=require_env("SUPABASE_DB_NAME"),
        user=require_env("SUPABASE_DB_USER"),
        password=require_env("SUPABASE_DB_PASSWORD"),
    )


def load_dataframe(cur, table: str, df: pd.DataFrame, conflict_target: str | None = None) -> None:
    rows = [tuple(item) for item in df.itertuples(index=False, name=None)]
    if not rows:
        return
    columns = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    if conflict_target:
        sql += f" ON CONFLICT ({conflict_target}) DO NOTHING"
    cur.executemany(sql, rows)


def main() -> None:
    args = parse_args()
    version = args.version
    gold_root = DATA_ROOT / "gold" / version / "official_dense"
    feature_catalog = DATA_ROOT / "gold" / version / "feature_catalog.csv"
    run_manifest = DATA_ROOT / "governance" / version / "run_manifest.json"

    gold_parts = sorted(gold_root.glob("year=*/*.parquet"))
    if not gold_parts:
        raise SystemExit(f"Nenhum parquet gold encontrado em {gold_root}")
    if not feature_catalog.exists():
        raise SystemExit(f"Catalogo de features ausente: {feature_catalog}")
    if not run_manifest.exists():
        raise SystemExit(f"Manifesto ausente: {run_manifest}")

    manifest_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    feature_df = pd.read_csv(feature_catalog).rename(columns={"group": "feature_group"})

    with get_connection() as conn, conn.cursor() as cur:
        if args.truncate:
            cur.execute("TRUNCATE TABLE sinan_gold_weekly")
            cur.execute("TRUNCATE TABLE sinan_feature_catalog")
            cur.execute("TRUNCATE TABLE sinan_run_manifest")

        for part in gold_parts:
            df = pd.read_parquet(part)
            load_dataframe(cur, "sinan_gold_weekly", df, conflict_target="ibge_municipio, ano_semana")
            print(f"loaded {part.name}")

        load_dataframe(cur, "sinan_feature_catalog", feature_df, conflict_target="feature")
        cur.execute(
            """
            INSERT INTO sinan_run_manifest (version, generated_at, schema_version, run_manifest)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (version) DO UPDATE
            SET generated_at = EXCLUDED.generated_at,
                schema_version = EXCLUDED.schema_version,
                run_manifest = EXCLUDED.run_manifest
            """,
            (
                manifest_payload["version"],
                manifest_payload["ended_at_utc"],
                manifest_payload["schema_version"],
                json.dumps(manifest_payload, ensure_ascii=False),
            ),
        )
    print("Carga concluida.")


if __name__ == "__main__":
    main()

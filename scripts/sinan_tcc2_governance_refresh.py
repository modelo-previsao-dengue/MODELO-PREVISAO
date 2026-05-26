#!/usr/bin/env python3
"""
Gera artefatos adicionais de governanca para uma versao oficial da trilha SINAN TCC2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "sinan"
REFERENCE_PATH = ROOT / "data" / "reference" / "ibge_municipios_api.json"
DEFAULT_VERSION = "sinan_tcc2_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atualiza relatorios de governanca da trilha SINAN TCC2")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    return parser.parse_args()


def ibge_row_to_uf_region(row: dict) -> tuple[str, str]:
    if row.get("microrregiao") and row["microrregiao"].get("mesorregiao"):
        uf_payload = row["microrregiao"]["mesorregiao"]["UF"]
        return str(uf_payload["sigla"]), str(uf_payload["regiao"]["sigla"])
    if row.get("regiao-imediata") and row["regiao-imediata"].get("regiao-intermediaria"):
        uf_payload = row["regiao-imediata"]["regiao-intermediaria"]["UF"]
        return str(uf_payload["sigla"]), str(uf_payload["regiao"]["sigla"])
    raise ValueError(f"Estrutura territorial inesperada no IBGE: {row}")


def load_ibge_reference() -> pd.DataFrame:
    data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    rows = []
    for row in data:
        uf, regiao = ibge_row_to_uf_region(row)
        rows.append(
            {
                "ibge_municipio": str(row["id"]),
                "municipio_ibge": str(row["nome"]),
                "uf_ibge": uf,
                "regiao_ibge": regiao,
            }
        )
    return pd.DataFrame(rows).sort_values("ibge_municipio").reset_index(drop=True)


def load_silver(version: str) -> pd.DataFrame:
    silver_root = DATA_ROOT / "silver" / version / "official_observed"
    parts = sorted(silver_root.glob("year=*/*.parquet"))
    if not parts:
        raise FileNotFoundError(f"Nenhum parquet silver encontrado em {silver_root}")
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


def main() -> None:
    args = parse_args()
    governance_root = DATA_ROOT / "governance" / args.version
    governance_root.mkdir(parents=True, exist_ok=True)

    silver_df = load_silver(args.version)
    ibge_df = load_ibge_reference()

    temporal = (
        silver_df.groupby("ano", as_index=False)
        .agg(
            municipio_semana_rows=("ano_semana", "count"),
            municipios_ativos=("ibge_municipio", "nunique"),
            notificacoes_total=("notificacoes", "sum"),
            semana_min=("ano_semana", "min"),
            semana_max=("ano_semana", "max"),
        )
        .sort_values("ano")
        .reset_index(drop=True)
    )
    temporal_csv = governance_root / "coverage_temporal.csv"
    temporal_json = governance_root / "coverage_temporal.json"
    temporal.to_csv(temporal_csv, index=False)
    temporal_json.write_text(json.dumps(temporal.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    municipal = (
        silver_df.groupby(["ibge_municipio", "municipio", "uf", "regiao"], as_index=False)
        .agg(
            rows_observed=("ano_semana", "count"),
            first_ano_semana=("ano_semana", "min"),
            last_ano_semana=("ano_semana", "max"),
            total_notifications=("notificacoes", "sum"),
        )
    )
    municipal["ibge_municipio"] = municipal["ibge_municipio"].astype(str)
    municipal["observed_in_silver"] = 1

    coverage = ibge_df.merge(municipal, on="ibge_municipio", how="left")
    coverage["municipio"] = coverage["municipio"].fillna(coverage["municipio_ibge"])
    coverage["uf"] = coverage["uf"].fillna(coverage["uf_ibge"])
    coverage["regiao"] = coverage["regiao"].fillna(coverage["regiao_ibge"])
    coverage["observed_in_silver"] = coverage["observed_in_silver"].fillna(0).astype(int)
    for column in ["rows_observed", "total_notifications"]:
        coverage[column] = coverage[column].fillna(0).astype(int)

    municipal_csv = governance_root / "coverage_municipal.csv"
    municipal_json = governance_root / "coverage_municipal.json"
    coverage.to_csv(municipal_csv, index=False)
    municipal_json.write_text(json.dumps(coverage.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")

    observed_count = int(coverage["observed_in_silver"].sum())
    official_count = int(len(coverage))
    summary = {
        "version": args.version,
        "official_municipalities_ibge": official_count,
        "municipalities_observed_in_silver": observed_count,
        "municipalities_without_valid_records": official_count - observed_count,
        "municipal_coverage_pct": round((observed_count / official_count) * 100, 4) if official_count else 0.0,
        "temporal_min_ano_semana": str(silver_df["ano_semana"].min()),
        "temporal_max_ano_semana": str(silver_df["ano_semana"].max()),
    }
    summary_path = governance_root / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(temporal_csv)
    print(municipal_csv)
    print(summary_path)


if __name__ == "__main__":
    main()

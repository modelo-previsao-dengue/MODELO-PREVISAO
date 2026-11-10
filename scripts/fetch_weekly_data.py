#!/usr/bin/env python3
"""Baixa dados recentes do SINAN (OpenDataSUS) e INMET (API Tempo) para atualização semanal.

Uso:
    python3 scripts/fetch_weekly_data.py              # baixa ano corrente
    python3 scripts/fetch_weekly_data.py --year 2026  # baixa ano específico
    python3 scripts/fetch_weekly_data.py --dry-run    # mostra sem baixar

Fontes:
    SINAN: https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/
    INMET: https://apitempo.inmet.gov.br/estacao/dados/{inicio}/{fim}
"""
import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "sinan"
INMET_BRONZE = BASE_DIR / "data" / "inmet" / "bronze" / "hourly"

SINAN_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/DENGBR{year_short:02d}.json.zip"
INMET_BDMEP = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"

INMET_CANONICAL = [
    "data", "hora_utc", "precipitacao_mm", "pressao_mbar",
    "pressao_max_mbar", "pressao_min_mbar", "radiacao_kj_m2",
    "temp_inst_c", "temp_orvalho_c", "temp_max_c", "temp_min_c",
    "temp_orvalho_max_c", "temp_orvalho_min_c",
    "umidade_max_pct", "umidade_min_pct", "umidade_inst_pct",
    "vento_dir_graus", "vento_rajada_ms", "vento_vel_ms",
]

INMET_API_MAP = {
    "DT_MEDICAO": "data",
    "HR_MEDICAO": "hora_utc",
    "CHUVA": "precipitacao_mm",
    "PRE_INS": "pressao_mbar",
    "PRE_MAX": "pressao_max_mbar",
    "PRE_MIN": "pressao_min_mbar",
    "RAD_GLO": "radiacao_kj_m2",
    "TEM_INS": "temp_inst_c",
    "PTO_INS": "temp_orvalho_c",
    "TEM_MAX": "temp_max_c",
    "TEM_MIN": "temp_min_c",
    "PTO_MAX": "temp_orvalho_max_c",
    "PTO_MIN": "temp_orvalho_min_c",
    "UMD_MAX": "umidade_max_pct",
    "UMD_MIN": "umidade_min_pct",
    "UMD_INS": "umidade_inst_pct",
    "VEN_DIR": "vento_dir_graus",
    "VEN_RAJ": "vento_rajada_ms",
    "VEN_VEL": "vento_vel_ms",
}


def fetch_url(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "DengueCast-Pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_sinan_year(year, dry_run=False):
    year_short = year % 100
    url = SINAN_URL.format(year_short=year_short)
    dest = RAW_DIR / f"DENGBR{year_short:02d}.json.zip"

    print(f"  SINAN {year}: {url}")
    if dry_run:
        print(f"    [DRY-RUN] salvaria em {dest}")
        return True

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch_url(url)
        dest.write_bytes(data)
        print(f"    OK: {len(data)/1024**2:.1f} MB -> {dest}")
        return True
    except Exception as e:
        print(f"    ERRO: {e}")
        return False


def fetch_inmet_year(year, dry_run=False):
    url = INMET_BDMEP.format(year=year)
    dest = BASE_DIR / "data" / "raw" / "inmet" / f"{year}.zip"

    print(f"  INMET {year}: {url}")
    if dry_run:
        print(f"    [DRY-RUN] salvaria em {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch_url(url, timeout=300)
        if len(data) < 1000:
            print(f"    SKIP: resposta muito pequena ({len(data)} bytes), ano pode nao estar disponivel")
            return False
        dest.write_bytes(data)
        print(f"    OK: {len(data)/1024**2:.1f} MB -> {dest}")

        extract_dir = BASE_DIR / "data" / "raw" / "inmet" / str(year)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(extract_dir)
        n_files = len(list(extract_dir.rglob("*.CSV")) + list(extract_dir.rglob("*.csv")))
        print(f"    Extraido: {n_files} CSVs em {extract_dir}")

        return True
    except Exception as e:
        print(f"    ERRO: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=datetime.now().year)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-sinan", action="store_true")
    ap.add_argument("--skip-inmet", action="store_true")
    args = ap.parse_args()

    year = args.year
    print(f"Fetch semanal — ano {year}")
    print("=" * 60)

    if not args.skip_sinan:
        print("\n[1/2] SINAN (OpenDataSUS)")
        fetch_sinan_year(year, dry_run=args.dry_run)
        if year > 2000:
            fetch_sinan_year(year - 1, dry_run=args.dry_run)

    if not args.skip_inmet:
        print("\n[2/2] INMET (BDMEP Portal)")
        fetch_inmet_year(year, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY-RUN: nenhum dado foi baixado.")
    else:
        print("Fetch completo. Rode a pipeline:")
        print(f"  scripts/rodar_v4.sh 40")


if __name__ == "__main__":
    main()

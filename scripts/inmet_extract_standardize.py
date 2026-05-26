#!/usr/bin/env python3
"""US-001: Extração e padronização dos CSVs INMET (2000-2026)."""

import glob
import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
BRONZE_DIR = BASE_DIR / "data" / "inmet" / "bronze"
HOURLY_DIR = BRONZE_DIR / "hourly"

STANDARDIZED_COLS = [
    "data", "hora_utc", "precipitacao_mm", "pressao_mbar",
    "pressao_max_mbar", "pressao_min_mbar", "radiacao_kj_m2",
    "temp_inst_c", "temp_max_c", "temp_min_c", "temp_orvalho_c",
    "temp_orvalho_inst_c", "umidade_inst_pct", "umidade_max_pct",
    "umidade_min_pct", "vento_dir_graus", "vento_rajada_ms", "vento_vel_ms",
]

MISSING_YEARS = [2009, 2020, 2022]


def find_inmet_zips():
    found = {}
    for fname in os.listdir(DOWNLOADS):
        if not fname.lower().endswith(".zip"):
            continue
        low = fname.lower()
        if "meteorol" in low or "inmet" in low:
            m = re.search(r"(\d{4})", fname)
            if m:
                year = int(m.group(1))
                if year not in found:
                    found[year] = str(DOWNLOADS / fname)
    return dict(sorted(found.items()))


def parse_header(raw_lines):
    meta = {}
    mapping = {
        "REGIAO": "regiao",
        "UF": "uf",
        "ESTACAO": "nome",
        "CODIGO (WMO)": "codigo_wmo",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ALTITUDE": "altitude",
        "DATA DE FUNDACAO": "data_fundacao",
    }
    for line in raw_lines[:8]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";", 1)
        if len(parts) < 2:
            continue
        key = parts[0].replace(":", "").strip()
        val = parts[1].strip()
        if key in mapping:
            meta[mapping[key]] = val
    for field in ["latitude", "longitude", "altitude"]:
        if field in meta:
            try:
                meta[field] = float(meta[field].replace(",", "."))
            except (ValueError, AttributeError):
                meta[field] = None
    return meta


def parse_csv_from_zip(zf, csv_name):
    try:
        raw = zf.read(csv_name).decode("latin1")
    except Exception:
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    lines = raw.split("\n")
    meta = parse_header(lines)

    data_text = "\n".join(lines[8:])
    try:
        df = pd.read_csv(
            io.StringIO(data_text),
            sep=";",
            decimal=",",
            encoding="latin1",
            on_bad_lines="skip",
        )
    except Exception:
        return None, meta

    if df.empty or len(df.columns) < 5:
        return None, meta

    col_map = {}
    for i, col in enumerate(df.columns):
        if i < len(STANDARDIZED_COLS):
            col_map[col] = STANDARDIZED_COLS[i]
    df = df.rename(columns=col_map)

    keep = [c for c in STANDARDIZED_COLS if c in df.columns]
    df = df[keep].copy()

    for col in df.columns:
        if col in ("data", "hora_utc"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -9999, col] = None

    df["codigo_wmo"] = meta.get("codigo_wmo", "")
    df["uf"] = meta.get("uf", "")
    df["regiao"] = meta.get("regiao", "")

    df = df.dropna(subset=["data"])
    df = df[df["data"].astype(str).str.match(r"^\d{4}")]

    return df, meta


def process_year(year, zip_path):
    print(f"  [{year}] Processando {os.path.basename(zip_path)}...")
    stations_meta = []
    frames = []

    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        print(f"    {len(csvs)} CSVs encontrados")

        for csv_name in csvs:
            df, meta = parse_csv_from_zip(zf, csv_name)
            meta["year"] = year
            stations_meta.append(meta)
            if df is not None and len(df) > 0:
                frames.append(df)

    if frames:
        all_data = pd.concat(frames, ignore_index=True)
        all_data["year"] = year
        out_dir = HOURLY_DIR / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(all_data, preserve_index=False)
        pq.write_table(table, out_dir / "data.parquet")
        print(f"    → {len(all_data):,} registros horários salvos")
    else:
        print(f"    → NENHUM dado válido!")
        all_data = pd.DataFrame()

    return stations_meta, len(all_data) if not all_data.empty else 0


def main():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)

    zips = find_inmet_zips()
    print(f"ZIPs encontrados: {len(zips)} anos")
    print(f"Anos faltantes (sem ZIP): {MISSING_YEARS}\n")

    all_stations = []
    inventory_rows = []

    for year, zip_path in zips.items():
        stations_meta, n_records = process_year(year, zip_path)
        for sm in stations_meta:
            sm.setdefault("codigo_wmo", "")
            all_stations.append(sm)
        inventory_rows.append({"year": year, "n_records": n_records, "n_stations": len(stations_meta)})

    # Station metadata table (deduplicated)
    st_df = pd.DataFrame(all_stations)
    if not st_df.empty:
        anos_por_estacao = st_df.groupby("codigo_wmo")["year"].apply(
            lambda x: sorted(x.unique().tolist())
        ).reset_index()
        anos_por_estacao.columns = ["codigo_wmo", "anos_com_dados"]
        anos_por_estacao["anos_com_dados"] = anos_por_estacao["anos_com_dados"].apply(
            lambda x: ",".join(str(y) for y in x)
        )

        st_unique = st_df.drop_duplicates(subset=["codigo_wmo"], keep="last")
        cols = ["codigo_wmo", "nome", "uf", "regiao", "latitude", "longitude", "altitude", "data_fundacao"]
        cols = [c for c in cols if c in st_unique.columns]
        st_unique = st_unique[cols].copy()
        st_unique = st_unique.merge(anos_por_estacao, on="codigo_wmo", how="left")
        st_unique.to_csv(BRONZE_DIR / "estacoes_inmet.csv", index=False)
        print(f"\nEstações únicas: {len(st_unique)}")

    # Inventory
    inv_df = pd.DataFrame(inventory_rows)
    inv_df.to_csv(BRONZE_DIR / "inventory.csv", index=False)
    print(f"\nInventário por ano:")
    print(inv_df.to_string(index=False))

    total_records = inv_df["n_records"].sum()
    print(f"\nTotal de registros horários: {total_records:,}")
    print(f"Anos processados: {len(zips)}")
    print(f"Anos faltantes: {MISSING_YEARS}")


if __name__ == "__main__":
    main()

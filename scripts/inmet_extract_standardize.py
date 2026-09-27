#!/usr/bin/env python3
"""US-000/US-001: Extracao e padronizacao dos CSVs INMET.

O mapeamento de colunas e feito pelo NOME do cabecalho, nao pela posicao.
A versao anterior usava uma lista posicional cujo indice 8 em diante divergia
do arquivo real: gravava ponto de orvalho como `temp_max_c`, temperatura
maxima como `temp_min_c`, orvalho minimo como `umidade_inst_pct`, umidade
relativa como `vento_dir_graus`, e descartava a velocidade do vento.

Uso:
    python3 scripts/inmet_extract_standardize.py --years 2018-2024
    python3 scripts/inmet_extract_standardize.py --years 2019,2021 --source-dir /caminho
"""

import argparse
import io
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / "data" / "inmet" / "bronze"
HOURLY_DIR = BRONZE_DIR / "hourly"
REPORT_DIR = BRONZE_DIR / "reports"

# Candidatos para o diretorio com os ZIPs/pastas anuais do INMET.
# Pode ser sobrescrito por --source-dir ou pela variavel INMET_SOURCE_DIR.
SOURCE_CANDIDATES = [
    Path.home() / "Downloads",
    Path("/mnt/c/Users/Thiago/Downloads"),
]

# Colunas canonicas do Bronze, na ordem em que aparecem no CSV do INMET.
CANONICAL_COLS = [
    "data", "hora_utc", "precipitacao_mm", "pressao_mbar",
    "pressao_max_mbar", "pressao_min_mbar", "radiacao_kj_m2",
    "temp_inst_c", "temp_orvalho_c", "temp_max_c", "temp_min_c",
    "temp_orvalho_max_c", "temp_orvalho_min_c",
    "umidade_max_pct", "umidade_min_pct", "umidade_inst_pct",
    "vento_dir_graus", "vento_rajada_ms", "vento_vel_ms",
]

NUMERIC_COLS = [c for c in CANONICAL_COLS if c not in ("data", "hora_utc")]

# Faixas fisicamente plausiveis, usadas para validar o mapeamento apos a
# extracao. Sao amplas de proposito: servem para pegar troca de coluna, nao
# para controle de qualidade fino.
PLAUSIBLE_RANGES = {
    "precipitacao_mm": (0, 200),
    "pressao_mbar": (600, 1100),
    "pressao_max_mbar": (600, 1100),
    "pressao_min_mbar": (600, 1100),
    "radiacao_kj_m2": (0, 30000),
    "temp_inst_c": (-20, 50),
    "temp_orvalho_c": (-40, 40),
    "temp_max_c": (-20, 50),
    "temp_min_c": (-20, 50),
    "temp_orvalho_max_c": (-40, 40),
    "temp_orvalho_min_c": (-40, 40),
    "umidade_max_pct": (0, 100),
    "umidade_min_pct": (0, 100),
    "umidade_inst_pct": (0, 100),
    "vento_dir_graus": (0, 360),
    "vento_rajada_ms": (0, 100),
    "vento_vel_ms": (0, 100),
}


def normalize(text):
    """Uppercase sem acento, com o '?' de mojibake tratado como coringa.

    Os arquivos de 2019 em diante vem com o cabecalho corrompido
    ('REGI?O', 'ESTAC?O'), entao a comparacao ignora esse caractere.
    """
    if text is None:
        return ""
    txt = unicodedata.normalize("NFKD", str(text))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = txt.upper().replace("?", "")
    return re.sub(r"\s+", " ", txt).strip()


def map_column(header):
    """Mapeia um cabecalho do CSV do INMET para o nome canonico do Bronze."""
    n = normalize(header)
    if not n:
        return None
    if n.startswith("DATA"):
        return "data"
    if n.startswith("HORA"):
        return "hora_utc"
    if "PRECIPITACAO" in n:
        return "precipitacao_mm"
    if "PRESSAO" in n:
        if "MAX" in n:
            return "pressao_max_mbar"
        if "MIN" in n:
            return "pressao_min_mbar"
        return "pressao_mbar"
    if "RADIACAO" in n:
        return "radiacao_kj_m2"
    # As variantes de orvalho precisam ser testadas antes das de temperatura,
    # porque 'TEMPERATURA ORVALHO MAX' tambem casa com 'MAX'.
    if "ORVALHO" in n:
        if "MAX" in n:
            return "temp_orvalho_max_c"
        if "MIN" in n:
            return "temp_orvalho_min_c"
        return "temp_orvalho_c"
    if "TEMPERATURA" in n:
        if "MAXIMA" in n:
            return "temp_max_c"
        if "MINIMA" in n:
            return "temp_min_c"
        if "BULBO SECO" in n or "TEMPERATURA DO AR" in n:
            return "temp_inst_c"
        return None
    if "UMIDADE" in n:
        if "MAX" in n:
            return "umidade_max_pct"
        if "MIN" in n:
            return "umidade_min_pct"
        return "umidade_inst_pct"
    if "VENTO" in n:
        if "DIRECAO" in n:
            return "vento_dir_graus"
        if "RAJADA" in n:
            return "vento_rajada_ms"
        if "VELOCIDADE" in n:
            return "vento_vel_ms"
    return None


# Prefixos, e nao nomes exatos: o cabecalho de 2019 em diante vem com o
# acento corrompido ('REGI?O', 'ESTAC?O'), e 'DATA DE FUNDACAO' aparece com e
# sem o sufixo '(YYYY-MM-DD)'. 'DATA DE FUND' precede 'DATA' na ordem de teste.
HEADER_META_PREFIXES = [
    ("CODIGO", "codigo_wmo"),
    ("LATITUDE", "latitude"),
    ("LONGITUDE", "longitude"),
    ("ALTITUDE", "altitude"),
    ("DATA DE FUND", "data_fundacao"),
    ("REGI", "regiao"),
    ("ESTA", "nome"),
    ("UF", "uf"),
]


def parse_header(raw_lines):
    meta = {}
    for line in raw_lines[:8]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";", 1)
        if len(parts) < 2:
            continue
        key = normalize(parts[0]).replace(":", "").strip()
        for prefix, field in HEADER_META_PREFIXES:
            if key.startswith(prefix):
                meta[field] = parts[1].strip()
                break
    for field in ["latitude", "longitude", "altitude"]:
        if field in meta:
            try:
                meta[field] = float(str(meta[field]).replace(",", "."))
            except (ValueError, AttributeError):
                meta[field] = None
    return meta


def parse_csv(raw_bytes):
    try:
        raw = raw_bytes.decode("latin1")
    except Exception:
        raw = raw_bytes.decode("utf-8", errors="replace")

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
        return None, meta, {}

    if df.empty or len(df.columns) < 5:
        return None, meta, {}

    col_map = {}
    unmapped = []
    for col in df.columns:
        canon = map_column(col)
        if canon is None:
            if normalize(col).startswith("UNNAMED"):
                continue
            unmapped.append(str(col))
            continue
        # Se dois cabecalhos casarem com o mesmo canonico, fica o primeiro.
        if canon not in col_map.values():
            col_map[col] = canon

    df = df.rename(columns=col_map)
    keep = [c for c in CANONICAL_COLS if c in df.columns]
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

    diag = {
        "mapped": set(col_map.values()),
        "unmapped": unmapped,
        "missing": [c for c in CANONICAL_COLS if c not in df.columns],
    }
    return df, meta, diag


def iter_year_sources(year, source_dir):
    """Rende (nome, bytes) de cada CSV do ano, seja de ZIP ou de pasta."""
    zip_candidates = [source_dir / f"{year}.zip"]
    for fname in sorted(os.listdir(source_dir)):
        low = fname.lower()
        if low.endswith(".zip") and str(year) in fname and (
            "meteorol" in low or "inmet" in low
        ):
            zip_candidates.append(source_dir / fname)

    for zp in zip_candidates:
        if zp.exists():
            with zipfile.ZipFile(zp) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                for n in names:
                    yield n, zf.read(n)
            return

    year_dir = source_dir / str(year)
    if year_dir.is_dir():
        for p in sorted(year_dir.rglob("*")):
            if p.suffix.lower() == ".csv":
                yield p.name, p.read_bytes()
        return

    raise FileNotFoundError(f"Nenhum ZIP nem pasta para {year} em {source_dir}")


def column_stats(df, year):
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            rows.append({"year": year, "coluna": col, "n_valid": 0,
                         "media": None, "min": None, "max": None,
                         "fora_da_faixa_pct": None})
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        lo, hi = PLAUSIBLE_RANGES[col]
        out_of_range = (
            float(((s < lo) | (s > hi)).mean() * 100) if len(s) else None
        )
        rows.append({
            "year": year,
            "coluna": col,
            "n_valid": int(len(s)),
            "media": round(float(s.mean()), 3) if len(s) else None,
            "min": round(float(s.min()), 3) if len(s) else None,
            "max": round(float(s.max()), 3) if len(s) else None,
            "fora_da_faixa_pct": round(out_of_range, 3) if out_of_range is not None else None,
        })
    return rows


def read_existing_bronze(year):
    year_dir = HOURLY_DIR / f"year={year}"
    if not year_dir.exists():
        return None
    try:
        return pq.read_table(year_dir).to_pandas()
    except Exception:
        return None


def process_year(year, source_dir):
    print(f"  [{year}] Processando...")
    stations_meta = []
    frames = []
    unmapped_all = set()
    missing_all = set()
    n_csvs = 0

    for csv_name, raw in iter_year_sources(year, source_dir):
        n_csvs += 1
        df, meta, diag = parse_csv(raw)
        meta["year"] = year
        stations_meta.append(meta)
        if diag:
            unmapped_all.update(diag.get("unmapped", []))
            missing_all.update(diag.get("missing", []))
        if df is not None and len(df) > 0:
            frames.append(df)

    print(f"    {n_csvs} CSVs lidos")
    if unmapped_all:
        print(f"    ATENCAO cabecalhos nao mapeados: {sorted(unmapped_all)[:5]}")
    if missing_all:
        print(f"    ATENCAO colunas canonicas ausentes: {sorted(missing_all)}")

    if not frames:
        print("    -> NENHUM dado valido!")
        return stations_meta, 0, None

    all_data = pd.concat(frames, ignore_index=True)
    all_data["year"] = year

    out_dir = HOURLY_DIR / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(all_data, preserve_index=False),
        out_dir / "data.parquet",
    )
    print(f"    -> {len(all_data):,} registros horarios salvos")
    return stations_meta, len(all_data), all_data


def parse_years(spec):
    years = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            years.update(range(int(a), int(b) + 1))
        else:
            years.add(int(chunk))
    return sorted(years)


def resolve_source_dir(explicit):
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"--source-dir inexistente: {p}")
        return p
    env = os.environ.get("INMET_SOURCE_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in SOURCE_CANDIDATES:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Nenhum diretorio de origem encontrado. Use --source-dir ou INMET_SOURCE_DIR."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2018-2024",
                    help="Anos a extrair, ex.: '2018-2024' ou '2019,2021'")
    ap.add_argument("--source-dir", default=None,
                    help="Diretorio com os ZIPs/pastas anuais do INMET")
    args = ap.parse_args()

    years = parse_years(args.years)
    source_dir = resolve_source_dir(args.source_dir)
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Origem: {source_dir}")
    print(f"Anos: {years}\n")

    # Estatisticas do Bronze atual, antes de sobrescrever.
    before_rows = []
    for year in years:
        old = read_existing_bronze(year)
        if old is not None:
            before_rows.extend(column_stats(old, year))
        del old

    all_stations = []
    inventory_rows = []
    after_rows = []

    for year in years:
        stations_meta, n_records, new_df = process_year(year, source_dir)
        for sm in stations_meta:
            sm.setdefault("codigo_wmo", "")
            all_stations.append(sm)
        inventory_rows.append({
            "year": year, "n_records": n_records, "n_stations": len(stations_meta),
        })
        if new_df is not None:
            after_rows.extend(column_stats(new_df, year))
        del new_df

    write_station_metadata(all_stations)
    update_inventory(inventory_rows)
    write_mapping_report(before_rows, after_rows)

    total = sum(r["n_records"] for r in inventory_rows)
    print(f"\nTotal de registros horarios (anos processados): {total:,}")


def write_station_metadata(all_stations):
    st_df = pd.DataFrame(all_stations)
    if st_df.empty:
        return

    path = BRONZE_DIR / "estacoes_inmet.csv"
    anos = st_df.groupby("codigo_wmo")["year"].apply(
        lambda x: sorted(x.unique().tolist())
    ).reset_index()
    anos.columns = ["codigo_wmo", "anos_com_dados"]
    anos["anos_com_dados"] = anos["anos_com_dados"].apply(
        lambda x: ",".join(str(y) for y in x)
    )

    cols = ["codigo_wmo", "nome", "uf", "regiao", "latitude", "longitude",
            "altitude", "data_fundacao"]
    st_unique = st_df.drop_duplicates(subset=["codigo_wmo"], keep="last")
    st_unique = st_unique[[c for c in cols if c in st_unique.columns]].copy()
    st_unique = st_unique.merge(anos, on="codigo_wmo", how="left")

    # Reprocessar so alguns anos nao pode apagar estacoes dos demais.
    if path.exists():
        prev = pd.read_csv(path, dtype={"codigo_wmo": str})
        keep = prev[~prev["codigo_wmo"].isin(st_unique["codigo_wmo"])]
        st_unique = pd.concat([keep, st_unique], ignore_index=True)

    st_unique = st_unique.sort_values("codigo_wmo")
    st_unique.to_csv(path, index=False)
    print(f"\nEstacoes no catalogo: {len(st_unique)}")


def update_inventory(inventory_rows):
    path = BRONZE_DIR / "inventory.csv"
    inv = pd.DataFrame(inventory_rows)
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[~prev["year"].isin(inv["year"])]
        inv = pd.concat([prev, inv], ignore_index=True)
    inv = inv.sort_values("year")
    inv.to_csv(path, index=False)


def write_mapping_report(before_rows, after_rows):
    """Relatorio antes/depois exigido pelo FR-2."""
    if not after_rows:
        return

    after = pd.DataFrame(after_rows)
    out_csv = REPORT_DIR / "us000_mapeamento_colunas_antes_depois.csv"

    if before_rows:
        before = pd.DataFrame(before_rows)
        cmp_df = before.merge(
            after, on=["year", "coluna"], suffixes=("_antes", "_depois"), how="outer"
        )
        cmp_df["mudou_media"] = (
            cmp_df["media_antes"].round(2) != cmp_df["media_depois"].round(2)
        )
    else:
        cmp_df = after.copy()
        cmp_df["mudou_media"] = None

    cmp_df.to_csv(out_csv, index=False)
    print(f"\nRelatorio antes/depois: {out_csv}")

    fora = after[after["fora_da_faixa_pct"].fillna(0) > 1.0]
    if len(fora):
        print("\nATENCAO colunas com >1% dos valores fora da faixa plausivel:")
        print(fora[["year", "coluna", "media", "min", "max",
                    "fora_da_faixa_pct"]].to_string(index=False))
    else:
        print("Validacao de faixa: todas as colunas dentro do esperado.")


if __name__ == "__main__":
    main()

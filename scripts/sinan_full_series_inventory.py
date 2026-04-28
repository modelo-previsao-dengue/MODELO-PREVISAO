#!/usr/bin/env python3
"""
Inventario completo da serie oficial SINAN/Dengue para o Brasil.

Objetivo:
- baixar recursos oficiais faltantes
- contar registros reais por ano sem materializar toda a serie em memoria
- organizar um consolidado simples e auditavel
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sinan"
PROCESSED_DIR = ROOT / "data" / "processed" / "sinan"
YEARS = list(range(2000, 2027))

URL_CANDIDATES = [
    ("json", "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/DENGBR{year_short:02d}.json.zip"),
    ("csv", "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR{year_short:02d}.csv.zip"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventario completo da serie SINAN/Dengue")
    parser.add_argument("--download-missing", action="store_true", help="Baixa arquivos faltantes")
    parser.add_argument("--force-download", action="store_true", help="Rebaixa arquivos mesmo se ja existirem")
    parser.add_argument("--start-year", type=int, default=2000, help="Ano inicial")
    parser.add_argument("--end-year", type=int, default=2026, help="Ano final")
    return parser.parse_args()


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def iter_json_array(handle: io.TextIOBase, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, object]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    reached_eof = False

    while True:
        if not reached_eof and len(buffer) < chunk_size:
            chunk = handle.read(chunk_size)
            if chunk == "":
                reached_eof = True
            else:
                buffer += chunk

        if not started:
            buffer = buffer.lstrip()
            if not buffer:
                if reached_eof:
                    return
                continue
            if buffer[0] != "[":
                raise ValueError("JSON inesperado: esperado array no topo do arquivo.")
            buffer = buffer[1:]
            started = True

        progressed = False
        while True:
            buffer = buffer.lstrip()
            if not buffer:
                break
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                progressed = True
                continue
            try:
                obj, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield obj
            buffer = buffer[index:]
            progressed = True

        if reached_eof:
            buffer = buffer.strip()
            if buffer in {"", "]"}:
                return
            if not progressed:
                raise ValueError("Fim inesperado do arquivo JSON durante parsing incremental.")


def year_to_short(year: int) -> int:
    return year % 100


def local_candidates(year: int) -> List[Path]:
    short = year_to_short(year)
    return [
        RAW_DIR / f"DENGBR{short:02d}.json.zip",
        RAW_DIR / f"DENGBR{short:02d}.csv.zip",
    ]


def find_existing_file(year: int) -> Optional[Path]:
    for path in local_candidates(year):
        if path.exists():
            return path
    return None


def download_year(year: int, force: bool = False) -> Path:
    existing = find_existing_file(year)
    if existing and not force:
        print(f"[download] reutilizando {existing.name}", flush=True)
        return existing

    for fmt, template in URL_CANDIDATES:
        url = template.format(year_short=year_to_short(year))
        output_path = RAW_DIR / f"DENGBR{year_to_short(year):02d}.{fmt}.zip"
        print(f"[download] {year}: tentando {url}", flush=True)
        response = requests.get(url, stream=True, timeout=600)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        with output_path.open("wb") as fp:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fp.write(chunk)
        print(f"[download] salvo em {output_path}", flush=True)
        return output_path

    raise FileNotFoundError(f"Nenhum recurso oficial suportado encontrado para o ano {year}.")


def count_records_json(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if not members:
            raise ValueError(f"Nenhum JSON encontrado em {zip_path.name}")
        with archive.open(members[0]) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
            count = 0
            for _ in iter_json_array(text_handle):
                count += 1
    return count


def count_records_csv(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"Nenhum CSV encontrado em {zip_path.name}")
        with archive.open(members[0]) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8", newline="")
            count = -1
            for count, _ in enumerate(text_handle, start=0):
                pass
    return max(count, 0)


def count_records(zip_path: Path) -> tuple[int, str]:
    suffixes = "".join(zip_path.suffixes).lower()
    if ".json.zip" in suffixes:
        return count_records_json(zip_path), "json"
    if ".csv.zip" in suffixes:
        return count_records_csv(zip_path), "csv"
    raise ValueError(f"Formato nao suportado: {zip_path.name}")


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def write_markdown_report(rows: List[Dict[str, object]], output_path: Path) -> None:
    total_records = sum(int(row["records"]) for row in rows if row["status"] == "ok")
    ok_rows = [row for row in rows if row["status"] == "ok"]
    max_row = max(ok_rows, key=lambda row: int(row["records"])) if ok_rows else None
    min_row = min(ok_rows, key=lambda row: int(row["records"])) if ok_rows else None

    lines = [
        "# Inventario Completo - SINAN/Dengue Brasil",
        "",
        "## Escopo",
        "",
        "- Fonte: recursos oficiais do SINAN/Dengue no portal de dados abertos da saude",
        "- Escala: Brasil inteiro",
        f"- Anos analisados: {rows[0]['year']} a {rows[-1]['year']}" if rows else "- Anos analisados: N/A",
        f"- Anos com contagem concluida: {len(ok_rows)}",
        f"- Registros totais confirmados: {total_records:,}".replace(",", "."),
        "",
    ]

    if max_row:
        lines.append(f"- Maior volume anual: {max_row['year']} com {int(max_row['records']):,} registros".replace(",", "."))
    if min_row:
        lines.append(f"- Menor volume anual: {min_row['year']} com {int(min_row['records']):,} registros".replace(",", "."))

    lines.extend(
        [
            "",
            "## Tabela anual",
            "",
            "| Ano | Formato | Registros | Tamanho bruto | Status |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for row in rows:
        records = "" if row["records"] is None else f"{int(row['records']):,}".replace(",", ".")
        size = row["file_size_human"] or ""
        lines.append(f"| {row['year']} | {row['format'] or ''} | {records} | {size} | {row['status']} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_directories()

    years = list(range(args.start_year, args.end_year + 1))
    rows: List[Dict[str, object]] = []

    for year in years:
        try:
            zip_path = find_existing_file(year)
            if zip_path is None:
                if args.download_missing:
                    zip_path = download_year(year, force=args.force_download)
                else:
                    rows.append(
                        {
                            "year": year,
                            "format": None,
                            "records": None,
                            "file_name": None,
                            "file_size_bytes": None,
                            "file_size_human": None,
                            "status": "missing",
                        }
                    )
                    print(f"[missing] {year}: arquivo nao encontrado", flush=True)
                    continue

            records, fmt = count_records(zip_path)
            size_bytes = zip_path.stat().st_size
            row = {
                "year": year,
                "format": fmt,
                "records": records,
                "file_name": zip_path.name,
                "file_size_bytes": size_bytes,
                "file_size_human": human_size(size_bytes),
                "status": "ok",
            }
            rows.append(row)
            print(f"[count] {year}: {records} registros em {zip_path.name}", flush=True)
        except Exception as exc:
            rows.append(
                {
                    "year": year,
                    "format": None,
                    "records": None,
                    "file_name": None,
                    "file_size_bytes": None,
                    "file_size_human": None,
                    "status": f"error: {exc}",
                }
            )
            print(f"[error] {year}: {exc}", flush=True)

    csv_path = PROCESSED_DIR / f"sinan_brasil_inventory_{years[0]}_{years[-1]}.csv"
    json_path = PROCESSED_DIR / f"sinan_brasil_inventory_{years[0]}_{years[-1]}.json"
    md_path = PROCESSED_DIR / f"sinan_brasil_inventory_{years[0]}_{years[-1]}.md"

    headers = ["year", "format", "records", "file_name", "file_size_bytes", "file_size_human", "status"]
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join(str(row.get(key, "") if row.get(key, "") is not None else "") for key in headers))
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(rows, md_path)

    print("[done] inventario gerado:", flush=True)
    print(f"  - {csv_path}", flush=True)
    print(f"  - {json_path}", flush=True)
    print(f"  - {md_path}", flush=True)


if __name__ == "__main__":
    main()

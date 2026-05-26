#!/usr/bin/env python3
"""
Pipeline oficial da trilha SINAN do TCC2.

Escopo:
- bronze oficial com inventario e checksums dos brutos anuais
- silver nacional em municipio-semana observado
- gold nacional em municipio-semana denso para modelagem
- governanca, lineage, quality, clusterizacao e feature selection
- camada operacional e contrato API-ready
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sinan_pipeline import (
    ALARM_COLUMNS,
    COMORBIDITY_COLUMNS,
    GRAVE_COLUMNS,
    HEMORRHAGIC_COLUMNS,
    RACE_MAP,
    SYMPTOM_COLUMNS,
    age_code_to_years,
    canonical_string,
    iter_json_array,
    only_digits,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sinan"
DATA_ROOT = ROOT / "data" / "sinan"
REFERENCE_DIR = ROOT / "data" / "reference"
BRONZE_ROOT = DATA_ROOT / "bronze"
SILVER_ROOT = DATA_ROOT / "silver"
GOLD_ROOT = DATA_ROOT / "gold"
GOVERNANCE_ROOT = DATA_ROOT / "governance"
SERVING_ROOT = DATA_ROOT / "serving"
DOCS_DIR = ROOT / "docs"

SCRIPT_VERSION = "sinan_tcc2_v2"
SCHEMA_VERSION = "1.0.0"
IBGE_API_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
IBGE_CACHE_PATH = REFERENCE_DIR / "ibge_municipios_api.json"
SOURCE_URL_TEMPLATE = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/DENGBR{year_short:02d}.json.zip"
SOURCE_CATALOG_URL = "https://opendatasus.saude.gov.br/pt_BR/dataset/arboviroses-dengue"
SOURCE_DICTIONARY_URL = "https://opendatasus.saude.gov.br/dataset/4d5e5d44-58a8-4d67-b8aa-4ef1e4b00a1c/resource/ccc60f03-2834-49a3-98f7-7ee76f50a316/download/dic_dados_dengue.pdf"
RANDOM_STATE = 42
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2026
DEFAULT_ANALYTICS_SAMPLE = 200_000
DEFAULT_CLUSTER_SAMPLE = 60_000
DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 6
FULL_ASSIGNMENTS_PART_PREFIX = "part"
PROGRESS_EVERY = 50_000

MISSING_CODES = {"", "NAN", "NONE"}
GESTANTE_CODES = {"1", "2", "3", "4"}
CRITERIO_MAP = {
    "prop_criterio_lab": {"1"},
    "prop_criterio_clinico_epi": {"2"},
    "prop_criterio_clinico": {"3"},
}
CLASSI_FIN_MAP = {
    "prop_confirmado_provavel": {"1", "10", "11", "12"},
    "prop_descartado": {"2"},
    "prop_inconclusivo": {"8"},
    "prop_dengue_alarme": {"11"},
    "prop_dengue_grave": {"12"},
    "prop_chikungunya": {"13"},
}
COUNT_METRIC_FIELDS = {
    "prop_hospitalizado": "qt_hospitalizados",
    "prop_obito_agravo": "qt_obitos_agravo",
    "prop_obito_outras_causas": "qt_obitos_outras_causas",
    "prop_confirmado_provavel": "qt_confirmados_provaveis",
    "prop_descartado": "qt_descartados",
    "prop_inconclusivo": "qt_inconclusivos",
    "prop_dengue_alarme": "qt_dengue_alarme",
    "prop_dengue_grave": "qt_dengue_grave",
    "prop_chikungunya": "qt_chikungunya",
}

YES_NO_FIELD_GROUPS = {
    "sintoma": SYMPTOM_COLUMNS,
    "comorbidade": COMORBIDITY_COLUMNS,
    "alarme": ALARM_COLUMNS,
    "grave": GRAVE_COLUMNS,
    "hemorragico": HEMORRHAGIC_COLUMNS,
}

BASE_MEAN_FEATURES = [
    "idade_media_anos",
    "atraso_notificacao_medio_dias",
    "prop_sexo_masc",
    "prop_sexo_fem",
    "prop_gestante",
    "prop_hospitalizado",
    "prop_obito_agravo",
    "prop_obito_outras_causas",
    "prop_criterio_lab",
    "prop_criterio_clinico_epi",
    "prop_criterio_clinico",
    "prop_confirmado_provavel",
    "prop_descartado",
    "prop_inconclusivo",
    "prop_dengue_alarme",
    "prop_dengue_grave",
    "prop_chikungunya",
]
BASE_MEAN_FEATURES.extend([f"prop_raca_{value}" for value in RACE_MAP.values()])
for group_name, source_columns in YES_NO_FIELD_GROUPS.items():
    for source_column in source_columns:
        BASE_MEAN_FEATURES.append(f"prop_{source_column.lower()}")
BASE_MEAN_FEATURES = list(dict.fromkeys(BASE_MEAN_FEATURES))

SILVER_METRIC_COLUMNS = [
    "notificacoes",
    "qt_hospitalizados",
    "qt_obitos_agravo",
    "qt_obitos_outras_causas",
    "qt_confirmados_provaveis",
    "qt_descartados",
    "qt_inconclusivos",
    "qt_dengue_alarme",
    "qt_dengue_grave",
    "qt_chikungunya",
]
SILVER_METRIC_COLUMNS.extend(BASE_MEAN_FEATURES)

FEATURE_GROUP_DESCRIPTIONS = {
    "identificacao": "Chaves e atributos de calendario da linha municipio-semana.",
    "volume": "Medidas de volume de notificacoes e desfechos agregados.",
    "proporcao_demografica": "Distribuicoes agregadas de sexo, raca e gestacao.",
    "proporcao_clinica": "Proporcoes agregadas de sintomas, comorbidades, alarme e gravidade.",
    "temporal": "Features temporais derivadas por municipio.",
    "indice_composto": "Indices clinico-epidemiologicos compostos para modelagem.",
    "label_auxiliar": "Rotulos auxiliares para alerta e analise exploratoria.",
}


@dataclass
class RunPaths:
    version: str
    bronze: Path
    silver: Path
    gold: Path
    governance: Path
    serving: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline oficial SINAN TCC2")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--version", default=SCRIPT_VERSION)
    parser.add_argument("--refresh-ibge-cache", action="store_true")
    parser.add_argument("--skip-bronze", action="store_true")
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-analytics", action="store_true")
    parser.add_argument("--skip-serving", action="store_true")
    parser.add_argument("--analytics-sample-size", type=int, default=DEFAULT_ANALYTICS_SAMPLE)
    parser.add_argument("--cluster-sample-size", type=int, default=DEFAULT_CLUSTER_SAMPLE)
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    return parser.parse_args()


def ensure_directories(paths: RunPaths) -> None:
    for path in [
        RAW_DIR,
        REFERENCE_DIR,
        paths.bronze,
        paths.silver,
        paths.gold,
        paths.governance,
        paths.serving,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_run_paths(version: str) -> RunPaths:
    return RunPaths(
        version=version,
        bronze=BRONZE_ROOT / version,
        silver=SILVER_ROOT / version,
        gold=GOLD_ROOT / version,
        governance=GOVERNANCE_ROOT / version,
        serving=SERVING_ROOT / version,
    )


def year_to_short(year: int) -> int:
    return year % 100


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fetch_json(url: str) -> object:
    raw = urllib.request.urlopen(url, timeout=120).read()
    if raw[:2] == b"\x1f\x8b":
        return json.loads(gzip.decompress(raw).decode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def ibge_row_to_uf_region(row: Dict[str, object]) -> tuple[str, str]:
    if row.get("microrregiao") and row["microrregiao"].get("mesorregiao"):
        uf_payload = row["microrregiao"]["mesorregiao"]["UF"]
        return str(uf_payload["sigla"]), str(uf_payload["regiao"]["sigla"])
    if row.get("regiao-imediata") and row["regiao-imediata"].get("regiao-intermediaria"):
        uf_payload = row["regiao-imediata"]["regiao-intermediaria"]["UF"]
        return str(uf_payload["sigla"]), str(uf_payload["regiao"]["sigla"])
    raise ValueError(f"Estrutura territorial inesperada na linha do IBGE: {row}")


def load_ibge_reference(cache_path: Path, refresh: bool = False) -> tuple[Dict[str, Dict[str, object]], Dict[str, str]]:
    if refresh or not cache_path.exists():
        data = fetch_json(IBGE_API_URL)
        write_json(cache_path, data)
    else:
        data = read_json(cache_path)

    by_code: Dict[str, Dict[str, object]] = {}
    by_prefix6: Dict[str, str] = {}
    for row in data:
        code = str(row["id"])
        uf, regiao = ibge_row_to_uf_region(row)
        by_code[code] = {
            "ibge_municipio": code,
            "municipio": row["nome"],
            "uf": uf,
            "regiao": regiao,
        }
        by_prefix6[code[:6]] = code
    return by_code, by_prefix6


def safe_date_from_text(value: object) -> Optional[date]:
    text = str(value or "").strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        return date(int(text[:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        return None


def normalize_week_text(value: object, preferred_year: Optional[int] = None) -> Optional[str]:
    text = str(value or "").strip()
    if "." in text:
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None

    def valid(year: int, week: int) -> bool:
        return 1900 <= year <= 2100 and 1 <= week <= 53

    candidates: List[tuple[int, int]] = []
    if len(digits) == 6:
        candidates.append((int(digits[:4]), int(digits[4:])))
        candidates.append((int(digits[2:]), int(digits[:2])))
    elif len(digits) == 5:
        candidates.append((int(digits[:4]), int(digits[4])))
        candidates.append((int(digits[1:]), int(digits[0])))

    valid_candidates = [(year, week) for year, week in candidates if valid(year, week)]
    if not valid_candidates:
        return None

    if preferred_year is not None:
        exact_matches = [(year, week) for year, week in valid_candidates if year == preferred_year]
        if exact_matches:
            year, week = exact_matches[0]
            return f"{year:04d}{week:02d}"

        close_matches = [
            (year, week)
            for year, week in valid_candidates
            if abs(year - preferred_year) <= 1
        ]
        if close_matches:
            close_matches.sort(key=lambda item: (abs(item[0] - preferred_year), item[1]))
            year, week = close_matches[0]
            return f"{year:04d}{week:02d}"

    if len(valid_candidates) == 1:
        year, week = valid_candidates[0]
        return f"{year:04d}{week:02d}"

    week_first_candidates = [(year, week) for year, week in valid_candidates if f"{week:02d}{year:04d}" == digits]
    if week_first_candidates:
        year, week = week_first_candidates[0]
        return f"{year:04d}{week:02d}"

    year, week = valid_candidates[0]
    return f"{year:04d}{week:02d}"


def derive_ano_semana(record: Dict[str, object], source_year: int) -> tuple[Optional[str], Optional[str]]:
    dt_notific = safe_date_from_text(record.get("DT_NOTIFIC"))
    dt_sin_pri = safe_date_from_text(record.get("DT_SIN_PRI"))

    week_fields = [
        ("SEM_NOT", dt_notific.year if dt_notific else source_year),
        ("SEM_PRI", dt_sin_pri.year if dt_sin_pri else source_year),
    ]
    for field, preferred_year in week_fields:
        normalized = normalize_week_text(record.get(field), preferred_year=preferred_year)
        if normalized:
            derived_year = int(normalized[:4])
            if abs(derived_year - source_year) <= 1:
                return normalized, field

    date_fields = [
        ("DT_NOTIFIC", dt_notific),
        ("DT_SIN_PRI", dt_sin_pri),
    ]
    for field, parsed in date_fields:
        if parsed:
            derived_year = int(parsed.strftime("%G"))
            if abs(derived_year - source_year) <= 1:
                return parsed.strftime("%G%V"), f"{field}_iso"
    return None, None


def ano_semana_to_week_start(ano_semana: str) -> date:
    year = int(ano_semana[:4])
    week = int(ano_semana[4:])
    jan4 = date(year, 1, 4)
    first_week_start = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return first_week_start + timedelta(days=(week - 1) * 7)


def fast_yes_no(value: object) -> float:
    code = canonical_string(value)
    if code == "1":
        return 1.0
    if code in {"0", "2"}:
        return 0.0
    return np.nan


def scalar_indicator(value: object, accepted_values: Iterable[str]) -> float:
    normalized = canonical_string(value)
    if normalized in MISSING_CODES:
        return np.nan
    return 1.0 if normalized in {canonical_string(item) for item in accepted_values} else 0.0


def compute_record_signature(record: Dict[str, object]) -> int:
    canonical = json.dumps(record, sort_keys=False, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def resolve_municipality(
    record: Dict[str, object],
    ibge_by_code: Dict[str, Dict[str, object]],
    ibge_by_prefix6: Dict[str, str],
) -> tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
    uf_tokens = {
        canonical_string(record.get("SG_UF")),
        canonical_string(record.get("UF")),
        canonical_string(record.get("SG_UF_NOT")),
    }
    municipality_candidates = [
        ("id_mn_resi", only_digits(record.get("ID_MN_RESI"))),
        ("id_municip", only_digits(record.get("ID_MUNICIP"))),
        ("comuninf", only_digits(record.get("COMUNINF"))),
    ]

    for source_field, digits in municipality_candidates:
        if not digits:
            continue
        resolution = None
        code7 = None
        normalized = digits[:7]

        if len(normalized) == 7 and normalized in ibge_by_code:
            code7 = normalized
            resolution = "direct_7d"
        elif len(digits) >= 6 and digits[:6] in ibge_by_prefix6:
            code7 = ibge_by_prefix6[digits[:6]]
            resolution = "prefix_6d_to_7d"
        elif digits.startswith("53") and ("DF" in uf_tokens or "53" in uf_tokens):
            code7 = "5300108"
            resolution = "df_prefix_fallback"

        if code7:
            ref = ibge_by_code[code7]
            return (
                {
                    "ibge_municipio": ref["ibge_municipio"],
                    "municipio": ref["municipio"],
                    "uf": ref["uf"],
                    "regiao": ref["regiao"],
                    "municipio_resolution": resolution,
                    "municipio_source_field": source_field,
                    "municipio_source_value": digits,
                },
                None,
            )

    unresolved = {
        "id_mn_resi": str(record.get("ID_MN_RESI") or ""),
        "id_municip": str(record.get("ID_MUNICIP") or ""),
        "comuninf": str(record.get("COMUNINF") or ""),
        "sg_uf": str(record.get("SG_UF") or ""),
        "uf": str(record.get("UF") or ""),
        "sg_uf_not": str(record.get("SG_UF_NOT") or ""),
    }
    return None, unresolved


def compute_feature_values(record: Dict[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}

    idade_anos = age_code_to_years(record.get("NU_IDADE_N"))
    if idade_anos == idade_anos:
        result["idade_media_anos"] = float(idade_anos)

    dt_notific = safe_date_from_text(record.get("DT_NOTIFIC"))
    dt_sin_pri = safe_date_from_text(record.get("DT_SIN_PRI"))
    if dt_notific and dt_sin_pri:
        result["atraso_notificacao_medio_dias"] = float((dt_notific - dt_sin_pri).days)

    sexo = canonical_string(record.get("CS_SEXO"))
    if sexo not in MISSING_CODES:
        result["prop_sexo_masc"] = 1.0 if sexo == "M" else 0.0
        result["prop_sexo_fem"] = 1.0 if sexo == "F" else 0.0

    gestante = canonical_string(record.get("CS_GESTANT"))
    if gestante not in MISSING_CODES:
        result["prop_gestante"] = 1.0 if gestante in GESTANTE_CODES else 0.0

    raca = canonical_string(record.get("CS_RACA"))
    if raca not in MISSING_CODES:
        for code, label in RACE_MAP.items():
            result[f"prop_raca_{label}"] = 1.0 if raca == code else 0.0

    hospitalizado = fast_yes_no(record.get("HOSPITALIZ"))
    if hospitalizado == hospitalizado:
        result["prop_hospitalizado"] = hospitalizado

    evolucao = canonical_string(record.get("EVOLUCAO"))
    if evolucao not in MISSING_CODES:
        result["prop_obito_agravo"] = 1.0 if evolucao == "2" else 0.0
        result["prop_obito_outras_causas"] = 1.0 if evolucao == "3" else 0.0

    criterio = canonical_string(record.get("CRITERIO"))
    if criterio not in MISSING_CODES:
        for field_name, accepted_values in CRITERIO_MAP.items():
            result[field_name] = 1.0 if criterio in accepted_values else 0.0

    classi_fin = canonical_string(record.get("CLASSI_FIN"))
    if classi_fin not in MISSING_CODES:
        for field_name, accepted_values in CLASSI_FIN_MAP.items():
            result[field_name] = 1.0 if classi_fin in accepted_values else 0.0

    for source_columns in YES_NO_FIELD_GROUPS.values():
        for source_column in source_columns:
            value = fast_yes_no(record.get(source_column))
            if value == value:
                result[f"prop_{source_column.lower()}"] = value

    return result


def build_empty_bucket(municipality_info: Dict[str, object], ano_semana: str) -> Dict[str, object]:
    bucket: Dict[str, object] = {
        "ibge_municipio": municipality_info["ibge_municipio"],
        "municipio": municipality_info["municipio"],
        "uf": municipality_info["uf"],
        "regiao": municipality_info["regiao"],
        "ano_semana": ano_semana,
        "municipio_resolution": municipality_info["municipio_resolution"],
        "municipio_source_field": municipality_info["municipio_source_field"],
        "notificacoes": 0,
    }
    for field_name in COUNT_METRIC_FIELDS.values():
        bucket[field_name] = 0.0
    return bucket


def aggregate_year_to_silver(
    year: int,
    paths: RunPaths,
    ibge_by_code: Dict[str, Dict[str, object]],
    ibge_by_prefix6: Dict[str, str],
) -> tuple[pd.DataFrame, Dict[str, object], List[Dict[str, object]]]:
    zip_path = RAW_DIR / f"DENGBR{year_to_short(year):02d}.json.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Arquivo bruto oficial nao encontrado: {zip_path}")

    buckets: Dict[Tuple[str, str], Dict[str, object]] = {}
    seen_signatures: set[int] = set()
    discarded_examples: List[Dict[str, object]] = []
    discarded_examples_limit = 200

    stats = {
        "year": year,
        "file_name": zip_path.name,
        "source_url": SOURCE_URL_TEMPLATE.format(year_short=year_to_short(year)),
        "file_size_bytes": int(zip_path.stat().st_size),
        "sha256": sha256_file(zip_path),
        "total_records": 0,
        "duplicate_records_removed": 0,
        "invalid_week_records": 0,
        "invalid_municipality_records": 0,
        "valid_records": 0,
        "unique_municipio_semana_rows": 0,
        "municipality_resolution_counts": defaultdict(int),
        "week_source_counts": defaultdict(int),
    }

    with zipfile.ZipFile(zip_path) as archive:
        json_members = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if not json_members:
            raise ValueError(f"Nenhum JSON encontrado em {zip_path.name}")
        member_name = json_members[0]
        stats["json_member"] = member_name

        with archive.open(member_name) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
            for record in iter_json_array(text_handle):
                stats["total_records"] += 1
                if stats["total_records"] % PROGRESS_EVERY == 0:
                    print(
                        f"[silver-progress] {year}: {stats['total_records']:,} registros lidos".replace(",", "."),
                        flush=True,
                    )

                signature = compute_record_signature(record)
                if signature in seen_signatures:
                    stats["duplicate_records_removed"] += 1
                    continue
                seen_signatures.add(signature)

                ano_semana, week_source = derive_ano_semana(record, source_year=year)
                if not ano_semana:
                    stats["invalid_week_records"] += 1
                    if len(discarded_examples) < discarded_examples_limit:
                        discarded_examples.append(
                            {
                                "year": year,
                                "discard_reason": "invalid_week",
                                "sem_pri": str(record.get("SEM_PRI") or ""),
                                "sem_not": str(record.get("SEM_NOT") or ""),
                                "dt_sin_pri": str(record.get("DT_SIN_PRI") or ""),
                                "dt_notific": str(record.get("DT_NOTIFIC") or ""),
                            }
                        )
                    continue

                municipality_info, unresolved = resolve_municipality(record, ibge_by_code, ibge_by_prefix6)
                if municipality_info is None:
                    stats["invalid_municipality_records"] += 1
                    if len(discarded_examples) < discarded_examples_limit:
                        row = {"year": year, "discard_reason": "invalid_municipality"}
                        row.update(unresolved or {})
                        discarded_examples.append(row)
                    continue

                stats["valid_records"] += 1
                stats["municipality_resolution_counts"][municipality_info["municipio_resolution"]] += 1
                stats["week_source_counts"][week_source or "unknown"] += 1

                key = (municipality_info["ibge_municipio"], ano_semana)
                bucket = buckets.get(key)
                if bucket is None:
                    bucket = build_empty_bucket(municipality_info, ano_semana)
                    buckets[key] = bucket

                bucket["notificacoes"] = int(bucket["notificacoes"]) + 1

                feature_values = compute_feature_values(record)
                for feature_name, value in feature_values.items():
                    if value is None or np.isnan(value):
                        continue
                    bucket[f"{feature_name}__sum"] = float(bucket.get(f"{feature_name}__sum", 0.0)) + float(value)
                    bucket[f"{feature_name}__count"] = float(bucket.get(f"{feature_name}__count", 0.0)) + 1.0
                    if feature_name in COUNT_METRIC_FIELDS:
                        bucket[COUNT_METRIC_FIELDS[feature_name]] = float(bucket.get(COUNT_METRIC_FIELDS[feature_name], 0.0)) + float(value)

    rows: List[Dict[str, object]] = []
    for (_, ano_semana), bucket in sorted(buckets.items()):
        row: Dict[str, object] = {
            "ibge_municipio": bucket["ibge_municipio"],
            "municipio": bucket["municipio"],
            "uf": bucket["uf"],
            "regiao": bucket["regiao"],
            "ano_semana": ano_semana,
            "ano": int(ano_semana[:4]),
            "semana_epidemiologica": int(ano_semana[4:]),
            "week_start": ano_semana_to_week_start(ano_semana).isoformat(),
            "source_year": year,
            "municipio_resolution": bucket["municipio_resolution"],
            "municipio_source_field": bucket["municipio_source_field"],
            "notificacoes": int(bucket["notificacoes"]),
        }
        for metric_name in COUNT_METRIC_FIELDS.values():
            row[metric_name] = int(round(float(bucket.get(metric_name, 0.0))))
        for feature_name in BASE_MEAN_FEATURES:
            sum_value = bucket.get(f"{feature_name}__sum")
            count_value = float(bucket.get(f"{feature_name}__count", 0.0))
            row[feature_name] = (float(sum_value) / count_value) if sum_value is not None and count_value else np.nan
        rows.append(row)

    silver_df = pd.DataFrame(rows)
    if not silver_df.empty:
        silver_df = silver_df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"]).reset_index(drop=True)
        silver_df["ibge_municipio"] = silver_df["ibge_municipio"].astype(str)
        silver_df["ano_semana"] = silver_df["ano_semana"].astype(str)

    stats["unique_municipio_semana_rows"] = int(len(silver_df))
    stats["municipality_resolution_counts"] = dict(stats["municipality_resolution_counts"])
    stats["week_source_counts"] = dict(stats["week_source_counts"])

    year_dir = paths.silver / "official_observed" / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    silver_path = year_dir / "part-000.parquet"
    silver_df.to_parquet(silver_path, index=False)
    stats["silver_path"] = str(silver_path)
    return silver_df, stats, discarded_examples


def write_bronze_inventory(paths: RunPaths, year_stats: List[Dict[str, object]]) -> Dict[str, str]:
    inventory_dir = paths.bronze / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(year_stats).sort_values("year").reset_index(drop=True)

    csv_path = inventory_dir / "sinan_bronze_inventory.csv"
    json_path = inventory_dir / "sinan_bronze_inventory.json"
    md_path = inventory_dir / "sinan_bronze_inventory.md"

    df.to_csv(csv_path, index=False)
    write_json(json_path, df.to_dict(orient="records"))

    lines = [
        "# Inventario Bronze Oficial - SINAN TCC2",
        "",
        f"- Versao: `{paths.version}`",
        f"- Arquivos anuais avaliados: `{len(df)}`",
        f"- Registros brutos lidos: `{int(df['total_records'].sum()):,}`".replace(",", "."),
        f"- Duplicatas exatas removidas na silver: `{int(df['duplicate_records_removed'].sum()):,}`".replace(",", "."),
        f"- Registros validos para a silver: `{int(df['valid_records'].sum()):,}`".replace(",", "."),
        "",
        "| ano | arquivo | tamanho_bytes | sha256 | registros | duplicatas | semana_invalida | municipio_invalido | validos | linhas_silver |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(
            f"| {row['year']} | {row['file_name']} | {row['file_size_bytes']} | {row['sha256']} | "
            f"{row['total_records']} | {row['duplicate_records_removed']} | {row['invalid_week_records']} | "
            f"{row['invalid_municipality_records']} | {row['valid_records']} | {row['unique_municipio_semana_rows']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def compute_global_missingness(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(frames), ignore_index=True)
    rows: List[Dict[str, object]] = []
    total = len(combined)
    for column in combined.columns:
        missing = int(combined[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(combined[column].dtype),
                "missing_count": missing,
                "missing_pct": round((missing / total) * 100, 4) if total else 0.0,
                "unique_values": int(combined[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def build_schema_payload(df: pd.DataFrame, dataset_name: str, primary_key: List[str], granularity: str) -> Dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "schema_version": SCHEMA_VERSION,
        "granularity": granularity,
        "primary_key": primary_key,
        "columns": [
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "nullable": bool(df[column].isna().any()),
            }
            for column in df.columns
        ],
    }


def build_silver_outputs(
    paths: RunPaths,
    silver_frames: List[pd.DataFrame],
    year_stats: List[Dict[str, object]],
    discarded_examples: List[Dict[str, object]],
) -> Dict[str, object]:
    combined = pd.concat(silver_frames, ignore_index=True) if silver_frames else pd.DataFrame()
    combined = combined.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"]).reset_index(drop=True)

    schema_payload = build_schema_payload(
        combined,
        dataset_name="silver_observed_municipio_semana",
        primary_key=["ibge_municipio", "ano_semana"],
        granularity="municipio-semana observado",
    )
    schema_path = paths.governance / "schema_silver.json"
    write_json(schema_path, schema_payload)

    missingness = compute_global_missingness([combined]) if not combined.empty else pd.DataFrame()
    missingness_path = paths.governance / "missingness_silver.csv"
    missingness.to_csv(missingness_path, index=False)

    discard_df = pd.DataFrame(discarded_examples)
    discard_path = paths.governance / "discard_report_silver.csv"
    discard_df.to_csv(discard_path, index=False)

    municipalities = int(combined["ibge_municipio"].nunique()) if not combined.empty else 0
    quality_summary = {
        "dataset": "silver_observed_municipio_semana",
        "version": paths.version,
        "schema_version": SCHEMA_VERSION,
        "record_counts": {
            "rows": int(len(combined)),
            "municipalities": municipalities,
            "years": sorted({int(value) for value in combined["ano"].unique()}) if not combined.empty else [],
            "total_notifications": int(combined["notificacoes"].sum()) if not combined.empty else 0,
        },
        "coverage": {
            "min_ano_semana": str(combined["ano_semana"].min()) if not combined.empty else None,
            "max_ano_semana": str(combined["ano_semana"].max()) if not combined.empty else None,
            "ufs": sorted(combined["uf"].dropna().astype(str).unique().tolist()) if not combined.empty else [],
        },
        "discard_summary": {
            "invalid_week_records": int(sum(item["invalid_week_records"] for item in year_stats)),
            "invalid_municipality_records": int(sum(item["invalid_municipality_records"] for item in year_stats)),
            "duplicate_records_removed": int(sum(item["duplicate_records_removed"] for item in year_stats)),
        },
    }
    quality_path = paths.governance / "quality_summary_silver.json"
    write_json(quality_path, quality_summary)

    lineage_payload = {
        "dataset": "silver_observed_municipio_semana",
        "version": paths.version,
        "sources": [
            {
                "type": "sinan_raw_json_zip",
                "year": item["year"],
                "path": str(RAW_DIR / item["file_name"]),
                "source_url": item["source_url"],
                "sha256": item["sha256"],
            }
            for item in year_stats
        ],
        "transformations": [
            "parsing incremental de arrays JSON oficiais",
            "reconciliacao territorial com tabela oficial de municipios do IBGE",
            "remocao de duplicidades exatas por assinatura canonica do registro",
            "validacao e derivacao de semana epidemiologica",
            "agregacao para municipio-semana observado",
        ],
    }
    lineage_path = paths.governance / "lineage_silver.json"
    write_json(lineage_path, lineage_payload)

    return {
        "combined": combined,
        "schema_path": str(schema_path),
        "missingness_path": str(missingness_path),
        "discard_path": str(discard_path),
        "quality_path": str(quality_path),
        "lineage_path": str(lineage_path),
    }


def compute_indices(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    symptom_columns = [f"prop_{column.lower()}" for column in SYMPTOM_COLUMNS if f"prop_{column.lower()}" in out.columns]
    comorbidity_columns = [f"prop_{column.lower()}" for column in COMORBIDITY_COLUMNS if f"prop_{column.lower()}" in out.columns]
    alarm_columns = [f"prop_{column.lower()}" for column in ALARM_COLUMNS if f"prop_{column.lower()}" in out.columns]
    severe_columns = [f"prop_{column.lower()}" for column in GRAVE_COLUMNS if f"prop_{column.lower()}" in out.columns]
    hemorrhagic_columns = [f"prop_{column.lower()}" for column in HEMORRHAGIC_COLUMNS if f"prop_{column.lower()}" in out.columns]

    out["indice_sintomas"] = out[symptom_columns].mean(axis=1, skipna=True) if symptom_columns else 0.0
    out["indice_comorbidades"] = out[comorbidity_columns].mean(axis=1, skipna=True) if comorbidity_columns else 0.0
    out["indice_alarme"] = out[alarm_columns].mean(axis=1, skipna=True) if alarm_columns else 0.0
    out["indice_gravidade"] = out[severe_columns].mean(axis=1, skipna=True) if severe_columns else 0.0
    out["indice_hemorragico"] = out[hemorrhagic_columns].mean(axis=1, skipna=True) if hemorrhagic_columns else 0.0
    out["indice_carga_clinica"] = out[
        [
            "indice_sintomas",
            "indice_comorbidades",
            "indice_alarme",
            "indice_gravidade",
            "indice_hemorragico",
        ]
    ].mean(axis=1, skipna=True)
    out["indice_desfecho_severo"] = out[
        [
            "prop_hospitalizado",
            "prop_dengue_alarme",
            "prop_dengue_grave",
            "prop_obito_agravo",
        ]
    ].mean(axis=1, skipna=True)
    out["indice_confirmacao"] = out["prop_confirmado_provavel"] - out["prop_inconclusivo"]
    return out


def build_gold_calendar(silver_df: pd.DataFrame) -> pd.DataFrame:
    calendar = (
        silver_df[["ano_semana", "ano", "semana_epidemiologica", "week_start"]]
        .drop_duplicates()
        .sort_values(["ano", "semana_epidemiologica"])
        .reset_index(drop=True)
    )
    calendar["week_start"] = pd.to_datetime(calendar["week_start"], errors="coerce")
    return calendar


def engineer_municipality_series(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values(["ano", "semana_epidemiologica"]).reset_index(drop=True).copy()
    notifications = group["notificacoes"].astype(float)
    group["week_of_year_sin"] = np.sin(2 * np.pi * group["semana_epidemiologica"].astype(float) / 52.0)
    group["week_of_year_cos"] = np.cos(2 * np.pi * group["semana_epidemiologica"].astype(float) / 52.0)
    group["is_zero_notification_week"] = (group["notificacoes"] == 0).astype(int)

    for lag in [1, 2, 3, 4, 8, 12]:
        group[f"notificacoes_lag_{lag}"] = notifications.shift(lag).fillna(0.0)

    shifted = notifications.shift(1).fillna(0.0)
    for window in [3, 4, 8, 12]:
        group[f"notificacoes_media_movel_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        group[f"notificacoes_min_movel_{window}"] = shifted.rolling(window=window, min_periods=1).min()
        group[f"notificacoes_max_movel_{window}"] = shifted.rolling(window=window, min_periods=1).max()

    group["notificacoes_diff_1"] = notifications.diff(1).fillna(0.0)
    group["notificacoes_diff_4"] = notifications.diff(4).fillna(0.0)
    group["notificacoes_pct_change_1"] = notifications.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    group["notificacoes_pct_change_4"] = notifications.pct_change(4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    group["notificacoes_aceleracao_1"] = group["notificacoes_diff_1"].diff(1).fillna(0.0)

    divisor_4 = group["notificacoes_media_movel_4"].replace(0.0, np.nan)
    divisor_8 = group["notificacoes_media_movel_8"].replace(0.0, np.nan)
    group["notificacoes_razao_media_4"] = (notifications / divisor_4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    group["notificacoes_razao_media_8"] = (notifications / divisor_8).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    q75 = float(notifications.quantile(0.75)) if len(group) else 0.0
    q90 = float(notifications.quantile(0.90)) if len(group) else 0.0
    pct75 = float(group["notificacoes_pct_change_1"].quantile(0.75)) if len(group) else 0.0
    rolling_std_8 = shifted.rolling(window=8, min_periods=2).std().fillna(0.0)

    has_notifications = notifications > 0
    group["label_alerta_q75"] = ((notifications >= q75) & has_notifications).astype(int)
    group["label_alerta_q90"] = ((notifications >= q90) & has_notifications).astype(int)
    group["label_crescimento_acelerado"] = (
        (group["notificacoes_pct_change_1"] >= pct75) & has_notifications
    ).astype(int)
    group["label_surto_local"] = (
        (notifications >= (group["notificacoes_media_movel_8"] + 2.0 * rolling_std_8))
        & has_notifications
    ).astype(int)
    group["label_semana_critica"] = (
        has_notifications
        & (
            (group["label_alerta_q90"] == 1)
            | (group["label_surto_local"] == 1)
            | (group["prop_dengue_grave"].fillna(0.0) >= 0.05)
        )
    ).astype(int)
    return group


def write_gold_dataset(
    paths: RunPaths,
    silver_df: pd.DataFrame,
    analytics_sample_size: int,
) -> Dict[str, object]:
    calendar = build_gold_calendar(silver_df)
    municipality_reference = (
        silver_df[["ibge_municipio", "municipio", "uf", "regiao"]]
        .drop_duplicates()
        .sort_values(["ibge_municipio"])
        .reset_index(drop=True)
    )

    observed_by_municipality = {
        municipality: frame.copy()
        for municipality, frame in silver_df.groupby("ibge_municipio", sort=True)
    }

    gold_official_root = paths.gold / "official_dense"
    gold_official_root.mkdir(parents=True, exist_ok=True)
    part_counters: Dict[int, int] = defaultdict(int)
    yearly_rows: Dict[int, int] = defaultdict(int)
    sample_frames: List[pd.DataFrame] = []
    batch_rows: List[pd.DataFrame] = []
    batch_limit = 120
    missing_counts: Dict[str, int] = defaultdict(int)
    total_gold_rows = 0

    def flush_batch() -> None:
        nonlocal batch_rows, total_gold_rows
        if not batch_rows:
            return
        batch = pd.concat(batch_rows, ignore_index=True)
        total_gold_rows += int(len(batch))
        for column in batch.columns:
            missing_counts[column] += int(batch[column].isna().sum())
        active_sample = batch.loc[batch["notificacoes"] > 0].copy()
        if len(active_sample):
            take = min(max(1000, analytics_sample_size // 40), len(active_sample))
            sample_frames.append(active_sample.sample(n=take, random_state=RANDOM_STATE))
        for year, year_df in batch.groupby("ano", sort=True):
            year_dir = gold_official_root / f"year={int(year)}"
            year_dir.mkdir(parents=True, exist_ok=True)
            path = year_dir / f"{FULL_ASSIGNMENTS_PART_PREFIX}-{part_counters[int(year)]:05d}.parquet"
            year_df.to_parquet(path, index=False)
            part_counters[int(year)] += 1
            yearly_rows[int(year)] += int(len(year_df))
        batch_rows = []

    for municipality_row in municipality_reference.to_dict(orient="records"):
        municipality = municipality_row["ibge_municipio"]
        observed = observed_by_municipality[municipality].copy()
        merged = calendar.merge(
            observed.drop(columns=["ano", "semana_epidemiologica", "week_start"]),
            on="ano_semana",
            how="left",
        )
        merged["ibge_municipio"] = municipality
        merged["municipio"] = municipality_row["municipio"]
        merged["uf"] = municipality_row["uf"]
        merged["regiao"] = municipality_row["regiao"]
        merged["municipio_resolution"] = merged["municipio_resolution"].fillna("dense_zero_fill")
        merged["municipio_source_field"] = merged["municipio_source_field"].fillna("calendar_fill")
        merged["source_year"] = merged["source_year"].fillna(merged["ano"]).astype(int)

        fill_zero_columns = ["notificacoes"]
        fill_zero_columns.extend(metric for metric in COUNT_METRIC_FIELDS.values())
        fill_zero_columns.extend(BASE_MEAN_FEATURES)
        for column in fill_zero_columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(0.0)

        merged = compute_indices(merged)
        engineered = engineer_municipality_series(merged)
        batch_rows.append(engineered)
        if len(batch_rows) >= batch_limit:
            flush_batch()
    flush_batch()

    sample = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if len(sample) > analytics_sample_size:
        sample = sample.sample(n=analytics_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    gold_schema_source = sample if len(sample) else silver_df.head(0)
    schema_payload = build_schema_payload(
        gold_schema_source,
        dataset_name="gold_dense_municipio_semana",
        primary_key=["ibge_municipio", "ano_semana"],
        granularity="municipio-semana denso",
    )
    schema_path = paths.governance / "schema_gold.json"
    write_json(schema_path, schema_payload)

    missingness_rows: List[Dict[str, object]] = []
    for column in gold_schema_source.columns:
        missing_count = missing_counts.get(column, 0)
        missingness_rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / total_gold_rows) * 100, 4) if total_gold_rows else 0.0,
            }
        )
    missingness_df = pd.DataFrame(missingness_rows).sort_values(["missing_pct", "column"], ascending=[False, True]).reset_index(drop=True)
    missingness_path = paths.governance / "missingness_gold.csv"
    missingness_df.to_csv(missingness_path, index=False)

    quality_summary = {
        "dataset": "gold_dense_municipio_semana",
        "version": paths.version,
        "schema_version": SCHEMA_VERSION,
        "record_counts": {
            "rows": int(total_gold_rows),
            "municipalities": int(len(municipality_reference)),
            "years": sorted(int(year) for year in yearly_rows),
            "rows_by_year": {str(year): int(value) for year, value in sorted(yearly_rows.items())},
        },
        "sampling": {
            "analytics_sample_rows": int(len(sample)),
            "analytics_sample_only_active_weeks": True,
        },
    }
    quality_path = paths.governance / "quality_summary_gold.json"
    write_json(quality_path, quality_summary)

    lineage_payload = {
        "dataset": "gold_dense_municipio_semana",
        "version": paths.version,
        "upstream_dataset": str(paths.silver / "official_observed"),
        "transformations": [
            "densificacao da serie municipio-semana sobre o calendario nacional observado",
            "zero fill para semanas sem notificacoes por municipio",
            "engenharia temporal por municipio",
            "construcao de indices clinico-epidemiologicos compostos",
            "geracao de labels auxiliares para alertas",
        ],
    }
    lineage_path = paths.governance / "lineage_gold.json"
    write_json(lineage_path, lineage_payload)

    return {
        "gold_root": gold_official_root,
        "schema_path": str(schema_path),
        "missingness_path": str(missingness_path),
        "quality_path": str(quality_path),
        "lineage_path": str(lineage_path),
        "calendar": calendar,
        "sample": sample,
        "total_gold_rows": total_gold_rows,
        "municipality_reference": municipality_reference,
        "yearly_rows": yearly_rows,
    }


def build_feature_catalog(sample: pd.DataFrame, paths: RunPaths) -> str:
    rows: List[Dict[str, object]] = []
    if sample.empty:
        catalog_df = pd.DataFrame(columns=["feature", "group", "dtype", "description"])
    else:
        for column in sample.columns:
            if column in {"ibge_municipio", "municipio", "uf", "regiao", "ano_semana", "week_start", "municipio_resolution", "municipio_source_field"}:
                group = "identificacao"
            elif column.startswith("qt_") or column == "notificacoes":
                group = "volume"
            elif column.startswith("prop_raca_") or column.startswith("prop_sexo_") or column == "prop_gestante":
                group = "proporcao_demografica"
            elif column.startswith("prop_"):
                group = "proporcao_clinica"
            elif column.startswith("notificacoes_") or column.startswith("week_of_year_") or column == "is_zero_notification_week":
                group = "temporal"
            elif column.startswith("indice_"):
                group = "indice_composto"
            elif column.startswith("label_"):
                group = "label_auxiliar"
            else:
                group = "identificacao"
            rows.append(
                {
                    "feature": column,
                    "group": group,
                    "dtype": str(sample[column].dtype),
                    "description": FEATURE_GROUP_DESCRIPTIONS.get(group, ""),
                }
            )
        catalog_df = pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)

    catalog_path = paths.gold / "feature_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    return str(catalog_path)


def choose_best_k(feature_matrix: np.ndarray, min_k: int, max_k: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if len(feature_matrix) < max_k + 1:
        max_k = max(min_k, len(feature_matrix) - 1)
    for k in range(min_k, max_k + 1):
        if k < 2:
            continue
        model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=4096, n_init=20)
        labels = model.fit_predict(feature_matrix)
        eval_size = min(len(feature_matrix), 12_000)
        eval_matrix = feature_matrix[:eval_size]
        eval_labels = labels[:eval_size]
        score = silhouette_score(eval_matrix, eval_labels) if len(set(eval_labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": float(score), "inertia": float(model.inertia_)})
    return pd.DataFrame(rows).sort_values(["silhouette", "k"], ascending=[False, True]).reset_index(drop=True)


def numeric_feature_columns_for_clustering(sample: pd.DataFrame) -> List[str]:
    excluded = {
        "ano",
        "semana_epidemiologica",
        "source_year",
        "ibge_municipio",
        "ano_semana",
        "week_start",
        "municipio",
        "uf",
        "regiao",
        "municipio_resolution",
        "municipio_source_field",
        "label_alerta_q75",
        "label_alerta_q90",
        "label_crescimento_acelerado",
        "label_surto_local",
        "label_semana_critica",
    }
    return [
        column
        for column in sample.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(sample[column])
    ]


def fit_cluster_model(
    sample: pd.DataFrame,
    min_k: int,
    max_k: int,
    mode: str,
) -> Dict[str, object]:
    numeric_columns = numeric_feature_columns_for_clustering(sample)
    if mode == "profile":
        numeric_columns = [column for column in numeric_columns if not column.startswith("notificacoes") and column != "notificacoes"]

    feature_frame = sample[numeric_columns].copy()
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.loc[:, feature_frame.nunique(dropna=False) > 1]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)
    evaluation = choose_best_k(scaled, min_k=min_k, max_k=max_k)
    best_k = int(evaluation.iloc[0]["k"])
    model = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, batch_size=4096, n_init=30)
    labels = model.fit_predict(scaled)

    labeled = sample[["ibge_municipio", "ano_semana", "notificacoes"]].copy()
    labeled["cluster_id"] = labels

    if mode == "intensity":
        ranking_metric = labeled.groupby("cluster_id")["notificacoes"].median().sort_values().index.tolist()
        label_names = [
            "muito_baixa_transmissao",
            "baixa_transmissao",
            "transicao",
            "alta_transmissao",
            "pico_epidemico",
            "extremo_epidemico",
        ]
    else:
        summary_frame = sample.copy()
        summary_frame["cluster_id"] = labels
        ranking_metric = (
            summary_frame.groupby("cluster_id")["indice_carga_clinica"]
            .median()
            .sort_values()
            .index.tolist()
        )
        label_names = [
            "perfil_basal",
            "perfil_leve",
            "perfil_moderado",
            "perfil_alterado",
            "perfil_agudo",
            "perfil_extremo",
        ]

    label_map = {cluster_id: label_names[min(index, len(label_names) - 1)] for index, cluster_id in enumerate(ranking_metric)}
    labeled["cluster_label"] = labeled["cluster_id"].map(label_map)

    return {
        "mode": mode,
        "feature_columns": feature_frame.columns.tolist(),
        "scaler": scaler,
        "model": model,
        "evaluation": evaluation,
        "sample_assignments": labeled,
        "label_map": {str(key): value for key, value in label_map.items()},
    }


def run_feature_selection(sample: pd.DataFrame, assignments: pd.DataFrame, mode: str) -> pd.DataFrame:
    merged = sample.merge(assignments[["ibge_municipio", "ano_semana", "cluster_id"]], on=["ibge_municipio", "ano_semana"], how="inner")
    target = merged["cluster_id"].astype(int)
    numeric_columns = numeric_feature_columns_for_clustering(merged)
    if mode == "profile":
        numeric_columns = [column for column in numeric_columns if not column.startswith("notificacoes") and column != "notificacoes"]

    feature_frame = merged[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_frame = feature_frame.loc[:, feature_frame.nunique(dropna=False) > 1]

    anova_scores, anova_pvalues = f_classif(feature_frame, target)
    mi_scores = mutual_info_classif(feature_frame, target, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(feature_frame, target)
    rf_scores = rf.feature_importances_

    ranking = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "anova_f": anova_scores,
            "anova_pvalue": anova_pvalues,
            "mutual_information": mi_scores,
            "random_forest_importance": rf_scores,
        }
    )
    for column in ["anova_f", "mutual_information", "random_forest_importance"]:
        series = ranking[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        min_value = float(series.min())
        max_value = float(series.max())
        if math.isclose(max_value, min_value):
            ranking[f"{column}_norm"] = 0.0
        else:
            ranking[f"{column}_norm"] = (series - min_value) / (max_value - min_value)
    ranking["composite_score"] = ranking[
        ["anova_f_norm", "mutual_information_norm", "random_forest_importance_norm"]
    ].mean(axis=1)
    ranking = ranking.sort_values(["composite_score", "anova_f"], ascending=[False, False]).reset_index(drop=True)
    ranking["selected"] = False
    ranking.loc[ranking.index < min(25, len(ranking)), "selected"] = True
    return ranking


def assign_clusters_to_full_gold(
    paths: RunPaths,
    model_bundle: Dict[str, object],
) -> tuple[str, str]:
    mode = model_bundle["mode"]
    assignments_root = paths.gold / "analytics" / f"cluster_assignments_{mode}_full"
    assignments_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    feature_columns = model_bundle["feature_columns"]
    scaler: StandardScaler = model_bundle["scaler"]
    model: MiniBatchKMeans = model_bundle["model"]
    label_map = {int(key): value for key, value in model_bundle["label_map"].items()}

    for year_dir in sorted((paths.gold / "official_dense").glob("year=*")):
        year = int(year_dir.name.split("=", 1)[1])
        target_dir = assignments_root / f"year={year}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for part_index, part_path in enumerate(sorted(year_dir.glob("*.parquet"))):
            df = pd.read_parquet(part_path)
            features = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scaled = scaler.transform(features)
            labels = model.predict(scaled)
            out = df[["ibge_municipio", "ano_semana", "ano", "semana_epidemiologica", "notificacoes"]].copy()
            out["cluster_id"] = labels
            out["cluster_label"] = out["cluster_id"].map(label_map)
            out_path = target_dir / f"{FULL_ASSIGNMENTS_PART_PREFIX}-{part_index:05d}.parquet"
            out.to_parquet(out_path, index=False)

            chunk_summary = (
                out.groupby(["cluster_id", "cluster_label"], as_index=False)
                .agg(
                    rows=("ano_semana", "count"),
                    media_notificacoes=("notificacoes", "mean"),
                    mediana_notificacoes=("notificacoes", "median"),
                )
                .assign(year=year)
            )
            summary_rows.extend(chunk_summary.to_dict(orient="records"))

    summary_df = (
        pd.DataFrame(summary_rows)
        .groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            rows=("rows", "sum"),
            media_notificacoes=("media_notificacoes", "mean"),
            mediana_notificacoes=("mediana_notificacoes", "mean"),
        )
        .sort_values(["media_notificacoes", "cluster_id"])
        .reset_index(drop=True)
    )
    summary_path = paths.gold / "analytics" / f"cluster_summary_{mode}.csv"
    summary_df.to_csv(summary_path, index=False)
    return str(assignments_root), str(summary_path)


def build_eda_outputs(
    paths: RunPaths,
    silver_df: pd.DataFrame,
    gold_sample: pd.DataFrame,
    cluster_bundles: List[Dict[str, object]],
) -> Dict[str, str]:
    analytics_dir = paths.gold / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    yearly_summary = (
        silver_df.groupby("ano", as_index=False)
        .agg(
            notificacoes_total=("notificacoes", "sum"),
            notificacoes_media=("notificacoes", "mean"),
            notificacoes_mediana=("notificacoes", "median"),
            notificacoes_max=("notificacoes", "max"),
            municipios_ativos=("ibge_municipio", "nunique"),
        )
        .sort_values("ano")
        .reset_index(drop=True)
    )
    yearly_summary_path = analytics_dir / "eda_yearly_summary.csv"
    yearly_summary.to_csv(yearly_summary_path, index=False)

    national_weeks = (
        silver_df.groupby(["ano_semana", "ano", "semana_epidemiologica"], as_index=False)
        .agg(notificacoes_total=("notificacoes", "sum"))
        .sort_values("notificacoes_total", ascending=False)
        .reset_index(drop=True)
    )
    peak_weeks_path = analytics_dir / "eda_peak_weeks.csv"
    national_weeks.head(30).to_csv(peak_weeks_path, index=False)

    numeric_columns = [column for column in gold_sample.columns if pd.api.types.is_numeric_dtype(gold_sample[column])]
    descriptive_stats = gold_sample[numeric_columns].describe().transpose().reset_index().rename(columns={"index": "feature"})
    descriptive_stats_path = analytics_dir / "eda_descriptive_stats.csv"
    descriptive_stats.to_csv(descriptive_stats_path, index=False)

    correlation_target_columns = [
        column
        for column in numeric_columns
        if column not in {"ano", "semana_epidemiologica", "source_year"}
    ]
    correlations = []
    target = gold_sample["notificacoes"].astype(float)
    for column in correlation_target_columns:
        if column == "notificacoes":
            continue
        series = gold_sample[column].astype(float)
        corr = target.corr(series)
        if corr == corr:
            correlations.append({"feature": column, "pearson_notificacoes": float(corr)})
    correlations_df = pd.DataFrame(correlations).sort_values("pearson_notificacoes", ascending=False).reset_index(drop=True)
    correlations_path = analytics_dir / "eda_correlations.csv"
    correlations_df.to_csv(correlations_path, index=False)

    cluster_summary_frames = []
    for bundle in cluster_bundles:
        summary_path = paths.gold / "analytics" / f"cluster_summary_{bundle['mode']}.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            summary_df.insert(0, "mode", bundle["mode"])
            cluster_summary_frames.append(summary_df)
    cluster_summary_df = pd.concat(cluster_summary_frames, ignore_index=True) if cluster_summary_frames else pd.DataFrame()
    cluster_summary_path = analytics_dir / "eda_cluster_summary.csv"
    cluster_summary_df.to_csv(cluster_summary_path, index=False)

    summary_payload = {
        "version": paths.version,
        "notificacoes_totais": int(silver_df["notificacoes"].sum()),
        "municipios_cobertos": int(silver_df["ibge_municipio"].nunique()),
        "periodo": {
            "min_ano_semana": str(silver_df["ano_semana"].min()),
            "max_ano_semana": str(silver_df["ano_semana"].max()),
        },
        "pico_nacional": national_weeks.head(1).to_dict(orient="records"),
    }
    summary_path = analytics_dir / "eda_summary.json"
    write_json(summary_path, summary_payload)

    report_lines = [
        "# Relatorio EDA SINAN TCC2",
        "",
        f"- Versao oficial: `{paths.version}`",
        f"- Municipios cobertos: `{int(silver_df['ibge_municipio'].nunique()):,}`".replace(",", "."),
        f"- Notificacoes totais aproveitadas: `{int(silver_df['notificacoes'].sum()):,}`".replace(",", "."),
        f"- Janela: `{silver_df['ano_semana'].min()}` ate `{silver_df['ano_semana'].max()}`",
        "",
        "## Maiores semanas nacionais por notificacoes",
        "",
    ]
    for row in national_weeks.head(10).to_dict(orient="records"):
        report_lines.append(
            f"- {row['ano_semana']}: {int(row['notificacoes_total']):,} notificacoes".replace(",", ".")
        )
    report_path = analytics_dir / "relatorio_eda.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "yearly_summary": str(yearly_summary_path),
        "peak_weeks": str(peak_weeks_path),
        "descriptive_stats": str(descriptive_stats_path),
        "correlations": str(correlations_path),
        "cluster_summary": str(cluster_summary_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }


def build_operational_outputs(
    paths: RunPaths,
    gold_schema_path: Path,
    feature_catalog_path: Path,
) -> Dict[str, str]:
    sql_dir = paths.serving / "sql"
    api_dir = paths.serving / "api"
    docs_dir = paths.serving / "docs"
    for path in [sql_dir, api_dir, docs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    schema = read_json(gold_schema_path)
    sql_columns = []
    for column in schema["columns"]:
        name = column["name"]
        dtype = column["dtype"]
        if "int" in dtype:
            sql_type = "BIGINT"
        elif "float" in dtype:
            sql_type = "DOUBLE PRECISION"
        elif "datetime" in dtype:
            sql_type = "TIMESTAMP"
        else:
            sql_type = "TEXT"
        nullability = "" if column["nullable"] else " NOT NULL"
        sql_columns.append(f"    {name} {sql_type}{nullability}")

    sql_lines = [
        "CREATE TABLE IF NOT EXISTS sinan_gold_weekly (",
        ",\n".join(sql_columns) + ",",
        "    PRIMARY KEY (ibge_municipio, ano_semana)",
        ");",
        "",
        "CREATE TABLE IF NOT EXISTS sinan_feature_catalog (",
        "    feature TEXT PRIMARY KEY,",
        "    feature_group TEXT NOT NULL,",
        "    dtype TEXT NOT NULL,",
        "    description TEXT",
        ");",
        "",
        "CREATE TABLE IF NOT EXISTS sinan_run_manifest (",
        "    version TEXT PRIMARY KEY,",
        "    generated_at TIMESTAMP NOT NULL,",
        "    schema_version TEXT NOT NULL,",
        "    run_manifest JSONB NOT NULL",
        ");",
    ]
    sql_path = sql_dir / "sinan_gold_schema.sql"
    sql_path.write_text("\n".join(sql_lines), encoding="utf-8")

    env_lines = [
        "SINAN_GOLD_ROOT=MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/official_dense",
        "SINAN_FEATURE_CATALOG=MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/feature_catalog.csv",
        "SUPABASE_DB_HOST=",
        "SUPABASE_DB_PORT=5432",
        "SUPABASE_DB_NAME=",
        "SUPABASE_DB_USER=",
        "SUPABASE_DB_PASSWORD=",
        "R2_ENDPOINT_URL=",
        "R2_ACCESS_KEY_ID=",
        "R2_SECRET_ACCESS_KEY=",
        "R2_BUCKET=",
        "R2_PREFIX=sinan/tcc2/v2",
    ]
    env_path = paths.serving / ".env.example"
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    publish_lines = [
        "# Publicacao Operacional - SINAN TCC2",
        "",
        "## Object storage",
        "",
        "- publicar o diretório `official_dense/` em bucket compatível com S3",
        "- preservar particionamento `year=YYYY`",
        "- publicar junto `feature_catalog.csv`, `schema_gold.json` e `run_manifest.json`",
        "",
        "## Banco operacional",
        "",
        "1. aplicar `sql/sinan_gold_schema.sql` no PostgreSQL/Supabase",
        "2. carregar a camada gold a partir dos parquet por ano",
        "3. carregar `feature_catalog.csv` na tabela `sinan_feature_catalog`",
        "4. registrar `run_manifest.json` em `sinan_run_manifest`",
        "5. opcionalmente usar os scripts de automacao em `MODELO-PREVISAO/scripts/`",
        "",
        "## Variaveis esperadas",
        "",
        "- `SUPABASE_DB_*`",
        "- `R2_*`",
        "",
        "## Scripts recomendados",
        "",
        "- `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2 --dry-run`",
        "- `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2`",
        "- `python3 scripts/sinan_tcc2_publish_postgres.py --version sinan_tcc2_v2`",
        "- `python3 scripts/sinan_tcc2_publish_postgres.py --version sinan_tcc2_v2 --truncate`",
        "",
        "## Contrato API sugerido",
        "",
        "- `GET /health`",
        "- `GET /v1/series/{ibge_municipio}?start=YYYYWW&end=YYYYWW`",
        "- `GET /v1/features/catalog`",
        "- `GET /v1/top-weeks?year=YYYY&limit=N`",
    ]
    publish_doc_path = docs_dir / "sinan_operational_publish.md"
    publish_doc_path.write_text("\n".join(publish_lines), encoding="utf-8")

    api_lines = [
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "import os",
        "",
        "import pandas as pd",
        "from fastapi import FastAPI, HTTPException, Query",
        "",
        "app = FastAPI(title='SINAN TCC2 API', version='1.0.0')",
        "",
        "GOLD_ROOT = Path(os.getenv('SINAN_GOLD_ROOT', 'MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/official_dense'))",
        "FEATURE_CATALOG = Path(os.getenv('SINAN_FEATURE_CATALOG', 'MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/feature_catalog.csv'))",
        "",
        "def load_all_gold() -> pd.DataFrame:",
        "    parts = sorted(GOLD_ROOT.glob('year=*/*.parquet'))",
        "    if not parts:",
        "        raise HTTPException(status_code=500, detail='Camada gold nao encontrada.')",
        "    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)",
        "",
        "@app.get('/health')",
        "def health() -> dict[str, str]:",
        "    return {'status': 'ok'}",
        "",
        "@app.get('/v1/features/catalog')",
        "def feature_catalog() -> list[dict[str, object]]:",
        "    if not FEATURE_CATALOG.exists():",
        "        raise HTTPException(status_code=500, detail='Catalogo de features nao encontrado.')",
        "    return pd.read_csv(FEATURE_CATALOG).to_dict(orient='records')",
        "",
        "@app.get('/v1/series/{ibge_municipio}')",
        "def municipality_series(ibge_municipio: str, start: str | None = Query(None), end: str | None = Query(None)) -> list[dict[str, object]]:",
        "    df = load_all_gold()",
        "    df['ibge_municipio'] = df['ibge_municipio'].astype(str)",
        "    out = df.loc[df['ibge_municipio'] == str(ibge_municipio)].copy()",
        "    if start is not None:",
        "        out = out.loc[out['ano_semana'].astype(str) >= str(start)]",
        "    if end is not None:",
        "        out = out.loc[out['ano_semana'].astype(str) <= str(end)]",
        "    if out.empty:",
        "        raise HTTPException(status_code=404, detail='Municipio nao encontrado na camada gold.')",
        "    out = out.sort_values(['ano', 'semana_epidemiologica'])",
        "    return out.to_dict(orient='records')",
        "",
        "@app.get('/v1/top-weeks')",
        "def top_weeks(year: int | None = Query(None), limit: int = Query(20, ge=1, le=100)) -> list[dict[str, object]]:",
        "    df = load_all_gold()",
        "    if year is not None:",
        "        df = df.loc[df['ano'] == int(year)]",
        "    grouped = df.groupby(['ano_semana', 'ano', 'semana_epidemiologica'], as_index=False)['notificacoes'].sum()",
        "    grouped = grouped.sort_values('notificacoes', ascending=False).head(limit)",
        "    return grouped.to_dict(orient='records')",
    ]
    api_path = api_dir / "sinan_api.py"
    api_path.write_text("\n".join(api_lines) + "\n", encoding="utf-8")

    load_script_lines = [
        "#!/usr/bin/env python3",
        "from __future__ import annotations",
        "",
        "import os",
        "from pathlib import Path",
        "",
        "import pandas as pd",
        "",
        "try:",
        "    import psycopg",
        "except Exception as exc:",
        "    raise SystemExit('psycopg nao instalado: ' + str(exc))",
        "",
        "gold_root = Path(os.environ['SINAN_GOLD_ROOT'])",
        "conn = psycopg.connect(",
        "    host=os.environ['SUPABASE_DB_HOST'],",
        "    port=os.environ.get('SUPABASE_DB_PORT', '5432'),",
        "    dbname=os.environ['SUPABASE_DB_NAME'],",
        "    user=os.environ['SUPABASE_DB_USER'],",
        "    password=os.environ['SUPABASE_DB_PASSWORD'],",
        ")",
        "parts = sorted(gold_root.glob('year=*/*.parquet'))",
        "with conn, conn.cursor() as cur:",
        "    for part in parts:",
        "        df = pd.read_parquet(part)",
        "        rows = [tuple(item) for item in df.itertuples(index=False, name=None)]",
        "        if not rows:",
        "            continue",
        "        columns = ', '.join(df.columns)",
        "        placeholders = ', '.join(['%s'] * len(df.columns))",
        "        sql = f'INSERT INTO sinan_gold_weekly ({columns}) VALUES ({placeholders}) ON CONFLICT (ibge_municipio, ano_semana) DO NOTHING'",
        "        cur.executemany(sql, rows)",
        "",
        "print('Carga concluida.')",
    ]
    load_script_path = api_dir / "load_gold_to_postgres.py"
    load_script_path.write_text("\n".join(load_script_lines) + "\n", encoding="utf-8")

    contract_payload = {
        "version": paths.version,
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
                "response": {"status": "ok"},
            },
            {
                "method": "GET",
                "path": "/v1/series/{ibge_municipio}",
                "params": {"start": "YYYYWW", "end": "YYYYWW"},
                "description": "Serie municipio-semana completa da camada gold.",
            },
            {
                "method": "GET",
                "path": "/v1/features/catalog",
                "description": "Catalogo oficial das features publicadas.",
            },
            {
                "method": "GET",
                "path": "/v1/top-weeks",
                "params": {"year": "YYYY", "limit": "1-100"},
                "description": "Semanas nacionais de maior volume.",
            },
        ],
        "feature_catalog_path": str(feature_catalog_path),
    }
    contract_path = docs_dir / "api_contract.json"
    write_json(contract_path, contract_payload)

    return {
        "sql_schema": str(sql_path),
        "env_example": str(env_path),
        "publish_doc": str(publish_doc_path),
        "api_app": str(api_path),
        "load_script": str(load_script_path),
        "api_contract": str(contract_path),
    }


def write_run_manifest(paths: RunPaths, payload: Dict[str, object]) -> str:
    manifest_path = paths.governance / "run_manifest.json"
    write_json(manifest_path, payload)
    return str(manifest_path)


def write_final_run_report(
    paths: RunPaths,
    silver_df: pd.DataFrame,
    year_stats: List[Dict[str, object]],
    gold_info: Optional[Dict[str, object]],
    eda_outputs: Optional[Dict[str, str]],
    cluster_outputs: List[Dict[str, str]],
) -> str:
    report_path = paths.governance / "final_run_report.md"
    total_duplicates = int(sum(item["duplicate_records_removed"] for item in year_stats))
    total_invalid_week = int(sum(item["invalid_week_records"] for item in year_stats))
    total_invalid_municipality = int(sum(item["invalid_municipality_records"] for item in year_stats))

    lines = [
        "# Relatorio Final da Trilha SINAN TCC2",
        "",
        f"- Versao oficial: `{paths.version}`",
        f"- Janela: `{int(silver_df['ano'].min())}` ate `{int(silver_df['ano'].max())}`",
        f"- Municipios cobertos na silver: `{int(silver_df['ibge_municipio'].nunique()):,}`".replace(",", "."),
        f"- Notificacoes aproveitadas na silver: `{int(silver_df['notificacoes'].sum()):,}`".replace(",", "."),
        f"- Duplicatas exatas removidas: `{total_duplicates:,}`".replace(",", "."),
        f"- Registros descartados por semana invalida: `{total_invalid_week:,}`".replace(",", "."),
        f"- Registros descartados por municipio invalido: `{total_invalid_municipality:,}`".replace(",", "."),
    ]
    if gold_info is not None:
        lines.append(f"- Linhas finais da gold densa: `{int(gold_info['total_gold_rows']):,}`".replace(",", "."))
    if eda_outputs is not None:
        lines.extend(
            [
                "",
                "## Artefatos principais",
                "",
                f"- Silver observada: `{paths.silver / 'official_observed'}`",
                f"- Gold densa: `{paths.gold / 'official_dense'}`",
                f"- Relatorio EDA: `{eda_outputs['report']}`",
            ]
        )
    if cluster_outputs:
        lines.extend(
            [
                "",
                "## Clusterizacao",
                "",
            ]
        )
        for output in cluster_outputs:
            lines.append(f"- `{output['mode']}`: `{output['summary_path']}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("Ano inicial maior que ano final.")

    paths = build_run_paths(args.version)
    ensure_directories(paths)
    ibge_by_code, ibge_by_prefix6 = load_ibge_reference(IBGE_CACHE_PATH, refresh=args.refresh_ibge_cache)

    years = list(range(args.start_year, args.end_year + 1))
    started_at = datetime.utcnow().isoformat() + "Z"

    silver_frames: List[pd.DataFrame] = []
    year_stats: List[Dict[str, object]] = []
    discarded_examples: List[Dict[str, object]] = []
    bronze_outputs: Dict[str, str] = {}

    if not args.skip_silver:
        for year in years:
            silver_df, stats, discarded = aggregate_year_to_silver(
                year=year,
                paths=paths,
                ibge_by_code=ibge_by_code,
                ibge_by_prefix6=ibge_by_prefix6,
            )
            silver_frames.append(silver_df)
            year_stats.append(stats)
            discarded_examples.extend(discarded)
            print(
                f"[silver] {year}: {stats['total_records']} registros, "
                f"{stats['valid_records']} validos, {stats['unique_municipio_semana_rows']} linhas silver",
                flush=True,
            )
        bronze_outputs = write_bronze_inventory(paths, year_stats)
    else:
        for year_dir in sorted((paths.silver / "official_observed").glob("year=*")):
            part_path = year_dir / "part-000.parquet"
            if part_path.exists():
                silver_frames.append(pd.read_parquet(part_path))
        inventory_df = maybe_read_csv(paths.bronze / "inventory" / "sinan_bronze_inventory.csv")
        if not inventory_df.empty:
            year_stats = inventory_df.to_dict(orient="records")
            bronze_outputs = {
                "csv": str(paths.bronze / "inventory" / "sinan_bronze_inventory.csv"),
                "json": str(paths.bronze / "inventory" / "sinan_bronze_inventory.json"),
                "markdown": str(paths.bronze / "inventory" / "sinan_bronze_inventory.md"),
            }
        discard_df = maybe_read_csv(paths.governance / "discard_report_silver.csv")
        if not discard_df.empty:
            discarded_examples = discard_df.to_dict(orient="records")

    silver_outputs = build_silver_outputs(paths, silver_frames, year_stats, discarded_examples)
    silver_df = silver_outputs["combined"]

    gold_info: Optional[Dict[str, object]] = None
    feature_catalog_path = None
    if not args.skip_gold:
        gold_info = write_gold_dataset(
            paths=paths,
            silver_df=silver_df,
            analytics_sample_size=args.analytics_sample_size,
        )
        feature_catalog_path = build_feature_catalog(gold_info["sample"], paths)

    cluster_outputs: List[Dict[str, str]] = []
    eda_outputs: Optional[Dict[str, str]] = None
    if not args.skip_analytics and gold_info is not None:
        analytics_root = paths.gold / "analytics"
        analytics_root.mkdir(parents=True, exist_ok=True)
        gold_sample = gold_info["sample"].copy()
        if len(gold_sample) > args.cluster_sample_size:
            cluster_sample = gold_sample.sample(n=args.cluster_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        else:
            cluster_sample = gold_sample.reset_index(drop=True)

        cluster_bundles: List[Dict[str, object]] = []
        for mode in ["intensity", "profile"]:
            bundle = fit_cluster_model(cluster_sample, min_k=args.min_k, max_k=args.max_k, mode=mode)
            evaluation_path = analytics_root / f"cluster_evaluation_{mode}.csv"
            sample_assignments_path = analytics_root / f"cluster_assignments_{mode}_sample.csv"
            bundle["evaluation"].to_csv(evaluation_path, index=False)
            bundle["sample_assignments"].to_csv(sample_assignments_path, index=False)

            ranking = run_feature_selection(cluster_sample, bundle["sample_assignments"], mode=mode)
            ranking_path = analytics_root / f"feature_ranking_{mode}.csv"
            selected_path = analytics_root / f"selected_features_{mode}.json"
            ranking.to_csv(ranking_path, index=False)
            write_json(
                selected_path,
                {
                    "mode": mode,
                    "selected_features": ranking.loc[ranking["selected"], "feature"].tolist(),
                    "top_25": ranking.head(25).to_dict(orient="records"),
                },
            )

            full_assignments_root, summary_path = assign_clusters_to_full_gold(paths, bundle)
            cluster_outputs.append(
                {
                    "mode": mode,
                    "evaluation_path": str(evaluation_path),
                    "sample_assignments_path": str(sample_assignments_path),
                    "ranking_path": str(ranking_path),
                    "selected_path": str(selected_path),
                    "full_assignments_root": full_assignments_root,
                    "summary_path": summary_path,
                }
            )
            cluster_bundles.append(bundle)

        eda_outputs = build_eda_outputs(paths, silver_df, gold_sample, cluster_bundles)

    serving_outputs: Dict[str, str] = {}
    if not args.skip_serving and gold_info is not None and feature_catalog_path is not None:
        serving_outputs = build_operational_outputs(
            paths=paths,
            gold_schema_path=Path(gold_info["schema_path"]),
            feature_catalog_path=Path(feature_catalog_path),
        )

    ended_at = datetime.utcnow().isoformat() + "Z"
    run_manifest = {
        "version": paths.version,
        "schema_version": SCHEMA_VERSION,
        "script": str(Path(__file__).resolve()),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "years": years,
        "official_outputs": {
            "bronze_inventory": bronze_outputs,
            "silver_root": str(paths.silver / "official_observed"),
            "gold_root": str(paths.gold / "official_dense") if gold_info is not None else None,
            "feature_catalog": feature_catalog_path,
            "governance_root": str(paths.governance),
            "serving_root": str(paths.serving),
        },
        "quality_outputs": {
            "silver_quality": silver_outputs["quality_path"],
            "gold_quality": gold_info["quality_path"] if gold_info is not None else None,
        },
        "analytics_outputs": cluster_outputs,
        "eda_outputs": eda_outputs,
        "serving_outputs": serving_outputs,
    }
    run_manifest_path = write_run_manifest(paths, run_manifest)
    final_report_path = write_final_run_report(paths, silver_df, year_stats, gold_info, eda_outputs, cluster_outputs)

    print("[done] pipeline oficial SINAN TCC2 concluida", flush=True)
    print(f"  - run_manifest: {run_manifest_path}", flush=True)
    print(f"  - final_report: {final_report_path}", flush=True)


if __name__ == "__main__":
    main()

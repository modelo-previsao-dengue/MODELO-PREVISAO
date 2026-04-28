#!/usr/bin/env python3
"""
Pipeline de coleta, processamento, clusterização e seleção de atributos do SINAN.

Fluxo:
1. Resume o que o TCC1 efetivamente utilizou.
2. Baixa os arquivos anuais completos do SINAN (fonte oficial OpenDataSUS/S3).
3. Extrai o recorte analítico configurado.
4. Gera atributos semanais a partir dos microdados do SINAN.
5. Executa clusterização das semanas epidemiológicas.
6. Ranqueia e seleciona atributos que melhor separam os clusters.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sinan"
PROCESSED_DIR = ROOT / "data" / "processed" / "sinan"
TCC1_DIR = ROOT.parent / "TCC1-DOCS"

SOURCE_URL_TEMPLATE = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/DENGBR{year_short:02d}.json.zip"
SOURCE_PORTAL_URL = "https://portalsinan.saude.gov.br/dados-epidemiologicos-SINAN"
SOURCE_BASEDOSDADOS_URL = "https://basedosdados.org/dataset/f51134c2-5ab9-4bbc-882f-f1034603147a"
SOURCE_DICTIONARY_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/dic_dados_dengue.pdf"

DEFAULT_YEARS = [2022, 2023, 2024]
FULL_DENGUE_YEARS = list(range(2000, 2027))
DEFAULT_UF = "DF"
DEFAULT_MUNICIPALITY_CODE = "5300108"
DEFAULT_MUNICIPALITY_NAME = "Brasilia"
DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 6
RANDOM_STATE = 42
UF_CODE_MAP = {
    "RO": "11",
    "AC": "12",
    "AM": "13",
    "RR": "14",
    "PA": "15",
    "AP": "16",
    "TO": "17",
    "MA": "21",
    "PI": "22",
    "CE": "23",
    "RN": "24",
    "PB": "25",
    "PE": "26",
    "AL": "27",
    "SE": "28",
    "BA": "29",
    "MG": "31",
    "ES": "32",
    "RJ": "33",
    "SP": "35",
    "PR": "41",
    "SC": "42",
    "RS": "43",
    "MS": "50",
    "MT": "51",
    "GO": "52",
    "DF": "53",
}

BASE_COLUMNS = [
    "DT_NOTIFIC",
    "SEM_NOT",
    "NU_ANO",
    "SG_UF_NOT",
    "ID_MUNICIP",
    "DT_SIN_PRI",
    "SEM_PRI",
    "ANO_NASC",
    "NU_IDADE_N",
    "CS_SEXO",
    "CS_GESTANT",
    "CS_RACA",
    "CS_ESCOL_N",
    "SG_UF",
    "ID_MN_RESI",
    "ID_RG_RESI",
    "ID_PAIS",
    "DT_INVEST",
    "FEBRE",
    "MIALGIA",
    "CEFALEIA",
    "EXANTEMA",
    "VOMITO",
    "NAUSEA",
    "DOR_COSTAS",
    "CONJUNTVIT",
    "ARTRITE",
    "ARTRALGIA",
    "PETEQUIA_N",
    "LEUCOPENIA",
    "LACO",
    "DOR_RETRO",
    "DIABETES",
    "HEMATOLOG",
    "HEPATOPAT",
    "RENAL",
    "HIPERTENSA",
    "ACIDO_PEPT",
    "AUTO_IMUNE",
    "RES_CHIKS1",
    "RES_CHIKS2",
    "RESUL_PRNT",
    "RESUL_SORO",
    "RESUL_NS1",
    "RESUL_VI_N",
    "RESUL_PCR_",
    "SOROTIPO",
    "HISTOPA_N",
    "IMUNOH_N",
    "HOSPITALIZ",
    "DT_INTERNA",
    "UF",
    "MUNICIPIO",
    "TPAUTOCTO",
    "COUFINF",
    "COPAISINF",
    "COMUNINF",
    "CLASSI_FIN",
    "CRITERIO",
    "DOENCA_TRA",
    "CLINC_CHIK",
    "EVOLUCAO",
    "DT_OBITO",
    "DT_ENCERRA",
    "ALRM_HIPOT",
    "ALRM_PLAQ",
    "ALRM_VOM",
    "ALRM_SANG",
    "ALRM_HEMAT",
    "ALRM_ABDOM",
    "ALRM_LETAR",
    "ALRM_HEPAT",
    "ALRM_LIQ",
    "DT_ALRM",
    "GRAV_PULSO",
    "GRAV_CONV",
    "GRAV_ENCH",
    "GRAV_INSUF",
    "GRAV_TAQUI",
    "GRAV_EXTRE",
    "GRAV_HIPOT",
    "GRAV_HEMAT",
    "GRAV_MELEN",
    "GRAV_METRO",
    "GRAV_SANG",
    "GRAV_AST",
    "GRAV_MIOC",
    "GRAV_CONSC",
    "GRAV_ORGAO",
    "DT_GRAV",
    "MANI_HEMOR",
    "EPISTAXE",
    "GENGIVO",
    "METRO",
    "PETEQUIAS",
    "HEMATURA",
    "SANGRAM",
    "LACO_N",
    "PLASMATICO",
    "EVIDENCIA",
    "PLAQ_MENOR",
    "CON_FHD",
    "COMPLICA",
    "TP_SISTEMA",
    "NDUPLIC_N",
    "DT_DIGITA",
    "CS_FLXRET",
    "FLXRECEBI",
    "MIGRADO_W",
]

SYMPTOM_COLUMNS = [
    "FEBRE",
    "MIALGIA",
    "CEFALEIA",
    "EXANTEMA",
    "VOMITO",
    "NAUSEA",
    "DOR_COSTAS",
    "CONJUNTVIT",
    "ARTRITE",
    "ARTRALGIA",
    "PETEQUIA_N",
    "LEUCOPENIA",
    "LACO",
    "DOR_RETRO",
]

COMORBIDITY_COLUMNS = [
    "DIABETES",
    "HEMATOLOG",
    "HEPATOPAT",
    "RENAL",
    "HIPERTENSA",
    "ACIDO_PEPT",
    "AUTO_IMUNE",
]

ALARM_COLUMNS = [
    "ALRM_HIPOT",
    "ALRM_PLAQ",
    "ALRM_VOM",
    "ALRM_SANG",
    "ALRM_HEMAT",
    "ALRM_ABDOM",
    "ALRM_LETAR",
    "ALRM_HEPAT",
    "ALRM_LIQ",
]

GRAVE_COLUMNS = [
    "GRAV_PULSO",
    "GRAV_CONV",
    "GRAV_ENCH",
    "GRAV_INSUF",
    "GRAV_TAQUI",
    "GRAV_EXTRE",
    "GRAV_HIPOT",
    "GRAV_HEMAT",
    "GRAV_MELEN",
    "GRAV_METRO",
    "GRAV_SANG",
    "GRAV_AST",
    "GRAV_MIOC",
    "GRAV_CONSC",
    "GRAV_ORGAO",
]

HEMORRHAGIC_COLUMNS = [
    "MANI_HEMOR",
    "EPISTAXE",
    "GENGIVO",
    "METRO",
    "PETEQUIAS",
    "HEMATURA",
    "SANGRAM",
    "LACO_N",
    "PLASMATICO",
    "EVIDENCIA",
    "PLAQ_MENOR",
    "CON_FHD",
]

YES_NO_COLUMNS = sorted(
    set(
        SYMPTOM_COLUMNS
        + COMORBIDITY_COLUMNS
        + ALARM_COLUMNS
        + GRAVE_COLUMNS
        + HEMORRHAGIC_COLUMNS
        + ["HOSPITALIZ", "DOENCA_TRA", "TPAUTOCTO"]
    )
)

DATE_COLUMNS = [
    "DT_NOTIFIC",
    "DT_SIN_PRI",
    "DT_INVEST",
    "DT_INTERNA",
    "DT_OBITO",
    "DT_ENCERRA",
    "DT_ALRM",
    "DT_GRAV",
    "DT_DIGITA",
]

RACE_MAP = {
    "1": "branca",
    "2": "preta",
    "3": "amarela",
    "4": "parda",
    "5": "indigena",
}


@dataclass
class Scope:
    years: List[int]
    uf: str
    municipality_code: str
    municipality_name: str
    full_brazil: bool = False

    @property
    def slug(self) -> str:
        if self.full_brazil:
            return f"brasil_{self.years[0]}_{self.years[-1]}"
        if self.uf and not self.municipality_code and not self.municipality_name:
            return f"{slugify(self.uf)}_{self.years[0]}_{self.years[-1]}"
        return f"{slugify(self.municipality_name)}_{self.years[0]}_{self.years[-1]}"

    @property
    def municipality_code_prefix(self) -> str:
        digits = only_digits(self.municipality_code)
        return digits[:6] if digits else ""

    @property
    def has_any_filter(self) -> bool:
        return self.full_brazil or bool(self.uf or self.municipality_code or self.municipality_name)


def slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
        .replace(" ", "_")
    )


def only_digits(value: object) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def canonical_string(value: object) -> str:
    return str(value or "").strip().upper()


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def year_to_short(year: int) -> int:
    return year % 100


def build_tcc1_scope_summary() -> Dict[str, object]:
    summary: Dict[str, object] = {
        "source_repo": str(TCC1_DIR),
        "scope_inferred": {
            "period": "2022-2024",
            "municipality_code": "5300108",
            "municipality_name": "Brasilia",
            "uf": "DF",
            "method": "InfoDengue aggregate series",
        },
        "notes": [
            "TCC1 usou serie agregada da API InfoDengue, nao microdados brutos do SINAN.",
            "A documentacao descreve Distrito Federal, mas o codigo efetivamente usa o geocode 5300108 (Brasilia).",
        ],
    }

    info_dengue_path = TCC1_DIR / "data_processed" / "sinan_2022_2024.csv"
    merged_path = TCC1_DIR / "data_processed" / "inmet_sinan_merged.csv"

    if info_dengue_path.exists():
        df = pd.read_csv(info_dengue_path)
        summary["tcc1_info_dengue_rows"] = int(len(df))
        summary["tcc1_info_dengue_columns"] = list(df.columns)
        if "cases" in df.columns:
            summary["tcc1_total_cases"] = int(df["cases"].sum())
            summary["tcc1_peak_cases"] = int(df["cases"].max())

    if merged_path.exists():
        df = pd.read_csv(merged_path)
        summary["tcc1_merged_rows"] = int(len(df))
        summary["tcc1_merged_columns"] = list(df.columns)

    output_path = PROCESSED_DIR / "tcc1_scope_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def download_year(year: int, force: bool = False) -> Path:
    ensure_directories()
    filename = f"DENGBR{year_to_short(year):02d}.json.zip"
    output_path = RAW_DIR / filename
    if output_path.exists() and not force:
        print(f"[download] reutilizando {output_path.name}", flush=True)
        return output_path

    url = SOURCE_URL_TEMPLATE.format(year_short=year_to_short(year))
    print(f"[download] {year}: {url}", flush=True)
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    with output_path.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fp.write(chunk)

    print(f"[download] salvo em {output_path}", flush=True)
    return output_path


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


def record_matches_scope(record: Dict[str, object], scope: Scope) -> bool:
    if scope.full_brazil or not scope.has_any_filter:
        return True

    target_prefix = scope.municipality_code_prefix
    municipality_candidates = [
        only_digits(record.get("ID_MN_RESI")),
        only_digits(record.get("ID_MUNICIP")),
        only_digits(record.get("COMUNINF")),
        only_digits(record.get("MUNICIPIO")),
    ]
    if target_prefix and any(code.startswith(target_prefix) for code in municipality_candidates if code):
        return True

    uf_res = canonical_string(record.get("SG_UF"))
    uf_not = canonical_string(record.get("UF"))
    allowed_uf_values = {scope.uf.upper(), UF_CODE_MAP.get(scope.uf.upper(), "")}
    allowed_uf_values.discard("")
    uf_match = not scope.uf or not allowed_uf_values or not allowed_uf_values.isdisjoint({uf_res, uf_not})
    if not uf_match:
        return False

    municipality_name = canonical_string(record.get("MUNICIPIO"))
    target_name = canonical_string(scope.municipality_name)
    if municipality_name and target_name and target_name in municipality_name:
        return True

    if scope.uf and not scope.municipality_code and not scope.municipality_name:
        return True

    # No TCC1, o recorte foi descrito como DF, embora o codigo cite Brasilia.
    # Como o SINAN real nem sempre preenche o campo textual do municipio de forma consistente,
    # usamos a UF de residencia como fallback para nao perder casos do DF.
    if scope.uf.upper() == "DF":
        return True

    return False


def summarize_value_counter(counter: Counter, limit: int = 20) -> Dict[str, int]:
    return {key: int(value) for key, value in counter.most_common(limit)}


def extract_scope_from_year(zip_path: Path, year: int, scope: Scope) -> tuple[pd.DataFrame, Dict[str, object]]:
    with zipfile.ZipFile(zip_path) as archive:
        json_members = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if not json_members:
            raise ValueError(f"Nenhum JSON encontrado em {zip_path}")

        member_name = json_members[0]
        total_records = 0
        filtered_records: List[Dict[str, object]] = []
        column_names: set[str] = set()
        class_counter: Counter = Counter()
        criterio_counter: Counter = Counter()

        with archive.open(member_name) as binary_handle:
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
            for record in iter_json_array(text_handle):
                total_records += 1
                column_names.update(record.keys())
                class_counter[canonical_string(record.get("CLASSI_FIN"))] += 1
                criterio_counter[canonical_string(record.get("CRITERIO"))] += 1

                if not record_matches_scope(record, scope):
                    continue

                selected = {column: record.get(column) for column in BASE_COLUMNS}
                selected["SOURCE_YEAR"] = year
                filtered_records.append(selected)

    df = pd.DataFrame(filtered_records)
    metadata = {
        "year": year,
        "zip_path": str(zip_path),
        "json_member": member_name,
        "total_records_in_year_file": int(total_records),
        "filtered_records_for_scope": int(len(df)),
        "available_columns_count": len(column_names),
        "available_columns": sorted(column_names),
        "classificacao_counts_top20": summarize_value_counter(class_counter),
        "criterio_counts_top20": summarize_value_counter(criterio_counter),
    }
    return df, metadata


def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def age_code_to_years(value: object) -> float:
    digits = only_digits(value)
    if not digits:
        return np.nan
    if len(digits) <= 3:
        amount = int(digits)
        if amount <= 130:
            return float(amount)
    digits = digits.zfill(4)
    unit = digits[0]
    amount = int(digits[1:])
    if unit == "4":
        return float(amount)
    if unit == "3":
        return float(amount) / 12.0
    if unit == "2":
        return float(amount) / 365.0
    if unit == "1":
        return float(amount) / (24.0 * 365.0)
    return np.nan


def yes_no_to_float(value: object) -> float:
    code = canonical_string(value)
    if code == "1":
        return 1.0
    if code in {"0", "2"}:
        return 0.0
    return np.nan


def category_indicator(series: pd.Series, accepted_values: Iterable[str]) -> pd.Series:
    accepted = {canonical_string(value) for value in accepted_values}
    normalized = series.astype(str).str.strip().str.upper()
    result = pd.Series(np.where(normalized.isin(accepted), 1.0, 0.0), index=series.index)
    missing_mask = normalized.isin({"", "NAN", "NONE"})
    result.loc[missing_mask] = np.nan
    return result


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in DATE_COLUMNS:
        if column in out.columns:
            out[f"{column}_PARSED"] = parse_date_column(out[column])

    if "NU_IDADE_N" in out.columns:
        out["IDADE_ANOS"] = out["NU_IDADE_N"].map(age_code_to_years)

    if {"DT_NOTIFIC_PARSED", "DT_SIN_PRI_PARSED"}.issubset(out.columns):
        out["ATRASO_NOTIFICACAO_DIAS"] = (
            out["DT_NOTIFIC_PARSED"] - out["DT_SIN_PRI_PARSED"]
        ).dt.days

    sem_pri = out.get("SEM_PRI", pd.Series(index=out.index, dtype=object)).astype(str).str.strip()
    out["ANO_SEMANA"] = sem_pri.where(sem_pri.str.match(r"^\d{6}$"), np.nan)
    out["ANO_EPI"] = pd.to_numeric(out["ANO_SEMANA"].str[:4], errors="coerce")
    out["SEMANA_EPI"] = pd.to_numeric(out["ANO_SEMANA"].str[4:], errors="coerce")

    if "CS_SEXO" in out.columns:
        out["SEXO_MASC"] = category_indicator(out["CS_SEXO"], ["M"])
        out["SEXO_FEM"] = category_indicator(out["CS_SEXO"], ["F"])

    if "CS_GESTANT" in out.columns:
        out["GESTANTE"] = category_indicator(out["CS_GESTANT"], ["1", "2", "3", "4"])

    if "CS_RACA" in out.columns:
        for code, label in RACE_MAP.items():
            out[f"RACA_{label.upper()}"] = category_indicator(out["CS_RACA"], [code])

    if "HOSPITALIZ" in out.columns:
        out["HOSPITALIZADO"] = out["HOSPITALIZ"].map(yes_no_to_float)

    if "EVOLUCAO" in out.columns:
        out["OBITO_AGRAVO"] = category_indicator(out["EVOLUCAO"], ["2"])
        out["OBITO_OUTRAS_CAUSAS"] = category_indicator(out["EVOLUCAO"], ["3"])

    if "CRITERIO" in out.columns:
        out["CRITERIO_LAB"] = category_indicator(out["CRITERIO"], ["1"])
        out["CRITERIO_CLINICO_EPI"] = category_indicator(out["CRITERIO"], ["2"])
        out["CRITERIO_CLINICO"] = category_indicator(out["CRITERIO"], ["3"])

    if "CLASSI_FIN" in out.columns:
        out["CASO_CONFIRMADO_PROVAVEL"] = category_indicator(out["CLASSI_FIN"], ["1", "10", "11", "12"])
        out["CASO_DESCARTADO"] = category_indicator(out["CLASSI_FIN"], ["2"])
        out["CASO_INCONCLUSIVO"] = category_indicator(out["CLASSI_FIN"], ["8"])
        out["CASO_DENGUE_ALARME"] = category_indicator(out["CLASSI_FIN"], ["11"])
        out["CASO_DENGUE_GRAVE"] = category_indicator(out["CLASSI_FIN"], ["12"])
        out["CASO_CHIKUNGUNYA"] = category_indicator(out["CLASSI_FIN"], ["13"])

    for column in YES_NO_COLUMNS:
        if column in out.columns:
            out[f"{column}_FLAG"] = out[column].map(yes_no_to_float)

    return out


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Nao ha registros no recorte filtrado para gerar atributos semanais.")

    required = ["ANO_EPI", "SEMANA_EPI", "ANO_SEMANA"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas epidemiologicas ausentes: {missing}")

    filtered = df.dropna(subset=["ANO_SEMANA"]).copy()
    filtered["ANO_EPI"] = filtered["ANO_EPI"].astype("Int64")
    filtered["SEMANA_EPI"] = filtered["SEMANA_EPI"].astype("Int64")

    feature_columns = [
        "IDADE_ANOS",
        "ATRASO_NOTIFICACAO_DIAS",
        "SEXO_MASC",
        "SEXO_FEM",
        "GESTANTE",
        "HOSPITALIZADO",
        "OBITO_AGRAVO",
        "OBITO_OUTRAS_CAUSAS",
        "CRITERIO_LAB",
        "CRITERIO_CLINICO_EPI",
        "CRITERIO_CLINICO",
        "CASO_CONFIRMADO_PROVAVEL",
        "CASO_DESCARTADO",
        "CASO_INCONCLUSIVO",
        "CASO_DENGUE_ALARME",
        "CASO_DENGUE_GRAVE",
        "CASO_CHIKUNGUNYA",
    ]

    feature_columns.extend([f"RACA_{label.upper()}" for label in RACE_MAP.values()])
    feature_columns.extend([f"{column}_FLAG" for column in SYMPTOM_COLUMNS])
    feature_columns.extend([f"{column}_FLAG" for column in COMORBIDITY_COLUMNS])
    feature_columns.extend([f"{column}_FLAG" for column in ALARM_COLUMNS])
    feature_columns.extend([f"{column}_FLAG" for column in GRAVE_COLUMNS])
    feature_columns.extend([f"{column}_FLAG" for column in HEMORRHAGIC_COLUMNS])

    existing_feature_columns = [column for column in feature_columns if column in filtered.columns]
    counts = filtered.groupby("ANO_SEMANA").size().reset_index(name="NOTIFICACOES")
    means = filtered.groupby("ANO_SEMANA")[existing_feature_columns].mean().reset_index() if existing_feature_columns else filtered[["ANO_SEMANA"]].drop_duplicates()
    weekly = counts.merge(means, on="ANO_SEMANA", how="left")
    weekly["ANO_EPI"] = pd.to_numeric(weekly["ANO_SEMANA"].str[:4], errors="coerce")
    weekly["SEMANA_EPI"] = pd.to_numeric(weekly["ANO_SEMANA"].str[4:], errors="coerce")
    rename_map = {column: column.lower() for column in existing_feature_columns}
    weekly = weekly.rename(columns=rename_map)

    weekly = weekly.sort_values(["ANO_EPI", "SEMANA_EPI"]).reset_index(drop=True)
    return weekly


def choose_kmeans_k(feature_matrix: np.ndarray, min_k: int, max_k: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_samples = feature_matrix.shape[0]
    for k in range(min_k, min(max_k, n_samples - 1) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels) if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": score, "inertia": float(model.inertia_)})

    evaluation = pd.DataFrame(rows)
    if evaluation.empty:
        raise ValueError("Nao ha semanas suficientes para avaliar clusterizacao.")
    return evaluation


def run_clustering(
    weekly: pd.DataFrame,
    min_k: int,
    max_k: int,
    excluded_features: Optional[set[str]] = None,
    descriptive_labels: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded_features = excluded_features or set()
    candidate_features = [
        column
        for column in weekly.columns
        if column not in {"ANO_SEMANA", "ANO_EPI", "SEMANA_EPI"} and column not in excluded_features
    ]
    candidate_features = [column for column in candidate_features if weekly[column].notna().sum() >= max(10, len(weekly) // 3)]

    feature_frame = weekly[candidate_features].copy()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(0.0)
    feature_frame = feature_frame.loc[:, feature_frame.nunique(dropna=False) > 1]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)

    evaluation = choose_kmeans_k(scaled, min_k=min_k, max_k=max_k)
    best_row = evaluation.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    best_k = int(best_row["k"])

    model = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
    labels = model.fit_predict(scaled)

    assignments = weekly[["ANO_SEMANA", "ANO_EPI", "SEMANA_EPI", "NOTIFICACOES"]].copy()
    assignments["CLUSTER"] = labels

    cluster_summary = assignments.groupby("CLUSTER", as_index=False).agg(
        semanas=("ANO_SEMANA", "count"),
        notificacoes_medias=("NOTIFICACOES", "mean"),
        notificacoes_maximas=("NOTIFICACOES", "max"),
    )
    cluster_summary = cluster_summary.sort_values("notificacoes_medias").reset_index(drop=True)
    descriptive_labels = descriptive_labels or ["baixa_transmissao", "transicao", "alta_transmissao", "pico", "extremo", "super_extremo"]
    label_map = {
        row["CLUSTER"]: descriptive_labels[min(index, len(descriptive_labels) - 1)]
        for index, row in cluster_summary.iterrows()
    }
    assignments["CLUSTER_LABEL"] = assignments["CLUSTER"].map(label_map)
    cluster_summary["cluster_label"] = cluster_summary["CLUSTER"].map(label_map)

    return assignments, evaluation, feature_frame.assign(CLUSTER=labels)


def minmax_normalize(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.nunique() <= 1:
        return pd.Series(np.zeros(len(clean)), index=clean.index)
    return (clean - clean.min()) / (clean.max() - clean.min())


def run_feature_selection(feature_frame_with_cluster: pd.DataFrame, dropped_features: Optional[set[str]] = None) -> pd.DataFrame:
    y = feature_frame_with_cluster["CLUSTER"].astype(int)
    x = feature_frame_with_cluster.drop(columns=["CLUSTER"]).copy()
    if dropped_features:
        x = x.drop(columns=[column for column in dropped_features if column in x.columns])
    x = x.fillna(x.median(numeric_only=True)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x = x.loc[:, x.nunique(dropna=False) > 1]

    f_scores, p_values = f_classif(x, y)
    mi_scores = mutual_info_classif(x, y, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    rf.fit(x, y)

    ranking = pd.DataFrame(
        {
            "feature": x.columns,
            "anova_f": f_scores,
            "anova_pvalue": p_values,
            "mutual_information": mi_scores,
            "random_forest_importance": rf.feature_importances_,
        }
    )
    ranking["score_anova_norm"] = minmax_normalize(ranking["anova_f"])
    ranking["score_mi_norm"] = minmax_normalize(ranking["mutual_information"])
    ranking["score_rf_norm"] = minmax_normalize(ranking["random_forest_importance"])
    ranking["composite_score"] = (
        ranking["score_anova_norm"] + ranking["score_mi_norm"] + ranking["score_rf_norm"]
    ) / 3.0
    ranking = ranking.sort_values(["composite_score", "random_forest_importance"], ascending=False).reset_index(drop=True)
    ranking["selected"] = ranking["composite_score"] >= ranking["composite_score"].quantile(0.75)
    return ranking


def summarize_analysis(assignments: pd.DataFrame, cluster_eval: pd.DataFrame, ranking: pd.DataFrame) -> Dict[str, object]:
    best_eval = cluster_eval.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    cluster_distribution = (
        assignments.groupby("CLUSTER_LABEL", as_index=False)
        .agg(
            semanas=("ANO_SEMANA", "count"),
            media_notificacoes=("NOTIFICACOES", "mean"),
            max_notificacoes=("NOTIFICACOES", "max"),
        )
        .sort_values("media_notificacoes")
        .reset_index(drop=True)
    )
    return {
        "best_k": int(best_eval["k"]),
        "best_silhouette": float(best_eval["silhouette"]),
        "cluster_distribution": cluster_distribution,
        "selected_top15": ranking[ranking["selected"]].head(15),
    }


def build_missingness_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = len(df)
    for column in df.columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing,
                "missing_pct": round((missing / total) * 100, 4) if total else 0.0,
                "unique_values": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "missing_pct", "column"], ascending=[True, False, True]).reset_index(drop=True)


def build_data_quality_summary(
    scope: Scope,
    normalized: pd.DataFrame,
    weekly: pd.DataFrame,
    metadata_rows: List[Dict[str, object]],
    intensity_summary: Dict[str, object],
    profile_summary: Dict[str, object],
) -> tuple[Dict[str, object], pd.DataFrame]:
    missingness = pd.concat(
        [
            build_missingness_table(normalized, "normalized_recorte"),
            build_missingness_table(weekly, "weekly_features"),
        ],
        ignore_index=True,
    )

    key_columns = [
        "DT_NOTIFIC",
        "DT_SIN_PRI",
        "SEM_PRI",
        "CS_SEXO",
        "CS_RACA",
        "CLASSI_FIN",
        "CRITERIO",
        "HOSPITALIZ",
        "EVOLUCAO",
    ]
    key_missingness = {
        column: round(float(normalized[column].isna().mean() * 100), 4)
        for column in key_columns
        if column in normalized.columns
    }

    class_counts = normalized["CLASSI_FIN"].astype(str).str.strip().value_counts(dropna=False).head(20).to_dict() if "CLASSI_FIN" in normalized.columns else {}
    criterio_counts = normalized["CRITERIO"].astype(str).str.strip().value_counts(dropna=False).head(20).to_dict() if "CRITERIO" in normalized.columns else {}

    summary = {
        "scope": {
            "scope_type": "full_brazil" if scope.full_brazil else ("uf" if scope.uf and not scope.municipality_code and not scope.municipality_name else "municipality"),
            "municipality_name": scope.municipality_name,
            "municipality_code": scope.municipality_code,
            "uf": scope.uf,
            "years": scope.years,
        },
        "record_counts": {
            "normalized_recorte_rows": int(len(normalized)),
            "weekly_rows": int(len(weekly)),
            "source_year_rows": {str(row["year"]): int(row["filtered_records_for_scope"]) for row in metadata_rows},
        },
        "temporal_coverage": {
            "min_dt_notific": str(normalized["DT_NOTIFIC_PARSED"].min().date()) if "DT_NOTIFIC_PARSED" in normalized.columns and normalized["DT_NOTIFIC_PARSED"].notna().any() else None,
            "max_dt_notific": str(normalized["DT_NOTIFIC_PARSED"].max().date()) if "DT_NOTIFIC_PARSED" in normalized.columns and normalized["DT_NOTIFIC_PARSED"].notna().any() else None,
            "min_ano_semana": str(weekly["ANO_SEMANA"].min()) if "ANO_SEMANA" in weekly.columns and len(weekly) else None,
            "max_ano_semana": str(weekly["ANO_SEMANA"].max()) if "ANO_SEMANA" in weekly.columns and len(weekly) else None,
        },
        "data_integrity": {
            "duplicated_rows_normalized": int(normalized.duplicated().sum()),
            "duplicated_weeks": int(weekly["ANO_SEMANA"].duplicated().sum()) if "ANO_SEMANA" in weekly.columns else 0,
            "key_missingness_pct": key_missingness,
        },
        "distribution_checks": {
            "classi_fin_top20": {str(k): int(v) for k, v in class_counts.items()},
            "criterio_top20": {str(k): int(v) for k, v in criterio_counts.items()},
        },
        "analysis_summary": {
            "intensity_best_k": intensity_summary["best_k"],
            "intensity_best_silhouette": round(float(intensity_summary["best_silhouette"]), 6),
            "profile_best_k": profile_summary["best_k"],
            "profile_best_silhouette": round(float(profile_summary["best_silhouette"]), 6),
        },
    }
    return summary, missingness


def write_processed_readme(scope: Scope, quality_summary: Dict[str, object], outputs: Dict[str, Path]) -> Path:
    output_path = PROCESSED_DIR / "README.md"
    if scope.full_brazil:
        scope_label = "Brasil"
        scope_code = "N/A"
    elif scope.uf and not scope.municipality_code and not scope.municipality_name:
        scope_label = f"UF {scope.uf}"
        scope_code = "N/A"
    else:
        scope_label = f"{scope.municipality_name}/{scope.uf}"
        scope_code = scope.municipality_code

    lines = [
        "# Entrega SINAN Processado",
        "",
        "Esta pasta contem os artefatos finais da task `Coleta e Processamento de Dados - SINAN`.",
        "",
        "## Recorte",
        "",
        f"- Escopo de referencia: {scope_label}",
        f"- Codigo de referencia: {scope_code}",
        f"- Periodo: {scope.years[0]}-{scope.years[-1]}",
        f"- Registros normalizados: {quality_summary['record_counts']['normalized_recorte_rows']:,}".replace(",", "."),
        f"- Semanas epidemiologicas: {quality_summary['record_counts']['weekly_rows']:,}".replace(",", "."),
        "",
        "## Principais arquivos",
        "",
        "- `recorte_*.parquet/csv`: microdados filtrados e normalizados",
        "- `weekly_features_*.csv`: agregacao semanal pronta para modelagem",
        "- `cluster_*`: artefatos de clusterizacao por intensidade e por perfil clinico",
        "- `feature_ranking*` e `selected_features*`: selecao de atributos",
        "- `quality_*`: validacao e qualidade dos dados",
        "- `relatorio_sinan_*.md`: relatorio consolidado para a monografia",
        "",
        "## Reexecucao",
        "",
        "```bash",
        "cd MODELO-PREVISAO",
        "python3 scripts/sinan_pipeline.py --skip-download",
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_quality_outputs(
    scope: Scope,
    quality_summary: Dict[str, object],
    missingness: pd.DataFrame,
    outputs: Dict[str, Path],
) -> Dict[str, Path]:
    quality_json_path = PROCESSED_DIR / f"quality_summary_{scope.slug}.json"
    quality_json_path.write_text(json.dumps(quality_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["quality_summary"] = quality_json_path

    missingness_path = PROCESSED_DIR / f"quality_missingness_{scope.slug}.csv"
    missingness.to_csv(missingness_path, index=False)
    outputs["quality_missingness"] = missingness_path

    manifest_payload = {
        "scope": {
            "scope_type": "full_brazil" if scope.full_brazil else ("uf" if scope.uf and not scope.municipality_code and not scope.municipality_name else "municipality"),
            "municipality_name": scope.municipality_name,
            "municipality_code": scope.municipality_code,
            "uf": scope.uf,
            "years": scope.years,
        },
        "generated_files": {key: str(path) for key, path in sorted(outputs.items())},
    }
    manifest_path = PROCESSED_DIR / f"artifact_manifest_{scope.slug}.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["artifact_manifest"] = manifest_path

    processed_readme_path = write_processed_readme(scope, quality_summary, outputs)
    outputs["processed_readme"] = processed_readme_path
    return outputs


def write_markdown_report(
    scope: Scope,
    tcc1_summary: Dict[str, object],
    metadata_rows: List[Dict[str, object]],
    recorte_df: pd.DataFrame,
    weekly: pd.DataFrame,
    intensity_summary: Dict[str, object],
    profile_summary: Dict[str, object],
) -> Path:
    output_path = PROCESSED_DIR / f"relatorio_sinan_{scope.slug}.md"
    if scope.full_brazil:
        scope_label = "Brasil"
        scope_code = "N/A"
    elif scope.uf and not scope.municipality_code and not scope.municipality_name:
        scope_label = f"UF {scope.uf}"
        scope_code = "N/A"
    else:
        scope_label = f"{scope.municipality_name}/{scope.uf}"
        scope_code = scope.municipality_code

    lines = [
        "# Relatório SINAN - TCC2",
        "",
        "## 1. Fontes",
        "",
        f"- Portal SINAN: {SOURCE_PORTAL_URL}",
        f"- OpenDataSUS / S3: {SOURCE_URL_TEMPLATE.format(year_short=24)}",
        f"- Dicionário de dados: {SOURCE_DICTIONARY_URL}",
        f"- Base dos Dados: {SOURCE_BASEDOSDADOS_URL}",
        "",
        "## 2. O que o TCC1 realmente fez",
        "",
        f"- Método: {tcc1_summary['scope_inferred']['method']}",
        f"- Recorte: {tcc1_summary['scope_inferred']['municipality_name']}/{tcc1_summary['scope_inferred']['uf']}",
        f"- Período: {tcc1_summary['scope_inferred']['period']}",
        f"- Geocódigo usado: {tcc1_summary['scope_inferred']['municipality_code']}",
        "",
        "## 3. Recorte adotado no TCC2",
        "",
        f"- Escopo padrão: {scope_label}",
        f"- Código de referência: {scope_code}",
        f"- Período: {scope.years[0]}-{scope.years[-1]}",
        f"- Microdados filtrados: {len(recorte_df):,}".replace(",", "."),
        f"- Semanas epidemiológicas com atributos: {len(weekly):,}".replace(",", "."),
        "",
        "## 4. Coleta por ano",
        "",
    ]

    for row in metadata_rows:
        lines.append(
            f"- {row['year']}: {row['filtered_records_for_scope']:,} registros do recorte em "
            f"{row['total_records_in_year_file']:,} registros totais".replace(",", ".")
        )

    lines.extend(
        [
            "",
            "## 5. Clusterização por Intensidade",
            "",
            f"- Melhor `k`: {intensity_summary['best_k']}",
            f"- Melhor silhouette: {intensity_summary['best_silhouette']:.4f}",
            "",
            "Distribuição dos clusters:",
            "",
        ]
    )

    for _, row in intensity_summary["cluster_distribution"].iterrows():
        lines.append(
            f"- {row['CLUSTER_LABEL']}: {int(row['semanas'])} semanas, "
            f"média {row['media_notificacoes']:.1f} notificações"
        )

    lines.extend(
        [
            "",
            "## 6. Atributos Selecionados por Intensidade",
            "",
            "Top atributos com maior capacidade de separar os clusters epidemiológicos:",
            "",
        ]
    )

    for _, row in intensity_summary["selected_top15"].iterrows():
        lines.append(f"- {row['feature']}: score composto {row['composite_score']:.4f}")

    lines.extend(
        [
            "",
            "## 7. Clusterização por Perfil Clínico-Epidemiológico",
            "",
            "- Nesta variante, `NOTIFICACOES` foi excluída do espaço de cluster para reduzir dominância da magnitude semanal.",
            f"- Melhor `k`: {profile_summary['best_k']}",
            f"- Melhor silhouette: {profile_summary['best_silhouette']:.4f}",
            "",
            "Distribuição dos clusters:",
            "",
        ]
    )

    for _, row in profile_summary["cluster_distribution"].iterrows():
        lines.append(
            f"- {row['CLUSTER_LABEL']}: {int(row['semanas'])} semanas, "
            f"média {row['media_notificacoes']:.1f} notificações"
        )

    lines.extend(
        [
            "",
            "## 8. Atributos Selecionados por Perfil Clínico",
            "",
            "Top atributos discriminativos quando a clusterização ignora o volume bruto de casos:",
            "",
        ]
    )

    for _, row in profile_summary["selected_top15"].iterrows():
        lines.append(f"- {row['feature']}: score composto {row['composite_score']:.4f}")

    lines.extend(
        [
            "",
            "## 9. Observações metodológicas",
            "",
            "- O pipeline baixa o arquivo anual completo do SINAN e só depois aplica o recorte analítico.",
            "- A clusterização é feita em nível de semana epidemiológica, para preservar comparabilidade com a série temporal do TCC1.",
            "- A seleção de atributos combina ANOVA, informação mútua e importância de Random Forest contra os rótulos de cluster.",
            "- A análise por intensidade captura semanas epidêmicas versus semanas de baixa transmissão.",
            "- A análise por perfil clínico força a interpretação dos sintomas, sinais de alarme e gravidade sem usar o volume de notificações como variável de agrupamento.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_outputs(
    scope: Scope,
    metadata_rows: List[Dict[str, object]],
    recorte_df: pd.DataFrame,
    weekly: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_eval: pd.DataFrame,
    ranking: pd.DataFrame,
    profile_assignments: pd.DataFrame,
    profile_cluster_eval: pd.DataFrame,
    profile_ranking: pd.DataFrame,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}

    metadata_path = PROCESSED_DIR / f"metadata_{scope.slug}.json"
    metadata_path.write_text(json.dumps(metadata_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["metadata"] = metadata_path

    recorte_parquet = PROCESSED_DIR / f"recorte_{scope.slug}.parquet"
    recorte_df.to_parquet(recorte_parquet, index=False)
    outputs["recorte_parquet"] = recorte_parquet

    if len(recorte_df) <= 1_000_000:
        recorte_csv = PROCESSED_DIR / f"recorte_{scope.slug}.csv"
        recorte_df.to_csv(recorte_csv, index=False)
        outputs["recorte_csv"] = recorte_csv

    weekly_path = PROCESSED_DIR / f"weekly_features_{scope.slug}.csv"
    weekly.to_csv(weekly_path, index=False)
    outputs["weekly_features"] = weekly_path

    assignments_path = PROCESSED_DIR / f"cluster_assignments_{scope.slug}.csv"
    assignments.to_csv(assignments_path, index=False)
    outputs["cluster_assignments"] = assignments_path

    cluster_eval_path = PROCESSED_DIR / f"cluster_evaluation_{scope.slug}.csv"
    cluster_eval.to_csv(cluster_eval_path, index=False)
    outputs["cluster_evaluation"] = cluster_eval_path

    ranking_path = PROCESSED_DIR / f"feature_ranking_{scope.slug}.csv"
    ranking.to_csv(ranking_path, index=False)
    outputs["feature_ranking"] = ranking_path

    selected_payload = {
        "scope": {
            "scope_type": "full_brazil" if scope.full_brazil else ("uf" if scope.uf and not scope.municipality_code and not scope.municipality_name else "municipality"),
            "years": scope.years,
            "uf": scope.uf,
            "municipality_code": scope.municipality_code,
            "municipality_name": scope.municipality_name,
        },
        "selected_features": ranking.loc[ranking["selected"], "feature"].tolist(),
        "top_15_features": ranking.head(15).to_dict(orient="records"),
    }
    selected_path = PROCESSED_DIR / f"selected_features_{scope.slug}.json"
    selected_path.write_text(json.dumps(selected_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["selected_features"] = selected_path

    profile_assignments_path = PROCESSED_DIR / f"cluster_assignments_profile_{scope.slug}.csv"
    profile_assignments.to_csv(profile_assignments_path, index=False)
    outputs["cluster_assignments_profile"] = profile_assignments_path

    profile_cluster_eval_path = PROCESSED_DIR / f"cluster_evaluation_profile_{scope.slug}.csv"
    profile_cluster_eval.to_csv(profile_cluster_eval_path, index=False)
    outputs["cluster_evaluation_profile"] = profile_cluster_eval_path

    profile_ranking_path = PROCESSED_DIR / f"feature_ranking_profile_{scope.slug}.csv"
    profile_ranking.to_csv(profile_ranking_path, index=False)
    outputs["feature_ranking_profile"] = profile_ranking_path

    profile_selected_payload = {
        "scope": {
            "scope_type": "full_brazil" if scope.full_brazil else ("uf" if scope.uf and not scope.municipality_code and not scope.municipality_name else "municipality"),
            "years": scope.years,
            "uf": scope.uf,
            "municipality_code": scope.municipality_code,
            "municipality_name": scope.municipality_name,
        },
        "analysis_mode": "clinical_profile",
        "selected_features": profile_ranking.loc[profile_ranking["selected"], "feature"].tolist(),
        "top_15_features": profile_ranking.head(15).to_dict(orient="records"),
    }
    profile_selected_path = PROCESSED_DIR / f"selected_features_profile_{scope.slug}.json"
    profile_selected_path.write_text(json.dumps(profile_selected_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["selected_features_profile"] = profile_selected_path

    return outputs


def run_pipeline(scope: Scope, min_k: int, max_k: int, force_download: bool, skip_download: bool) -> Dict[str, Path]:
    ensure_directories()
    tcc1_summary = build_tcc1_scope_summary()
    cached_normalized_path = PROCESSED_DIR / f"normalized_recorte_{scope.slug}.parquet"

    if skip_download and cached_normalized_path.exists():
        normalized = pd.read_parquet(cached_normalized_path)
        metadata_path = PROCESSED_DIR / f"metadata_{scope.slug}.json"
        metadata_rows = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
        print(f"[cache] reutilizando recorte normalizado em {cached_normalized_path}", flush=True)
    else:
        yearly_frames: List[pd.DataFrame] = []
        metadata_rows: List[Dict[str, object]] = []

        for year in scope.years:
            zip_path = RAW_DIR / f"DENGBR{year_to_short(year):02d}.json.zip"
            if not zip_path.exists() and skip_download:
                raise FileNotFoundError(f"Arquivo {zip_path.name} nao encontrado e --skip-download foi informado.")
            if not skip_download:
                zip_path = download_year(year, force=force_download)

            frame, metadata = extract_scope_from_year(zip_path, year, scope)
            yearly_frames.append(frame)
            metadata_rows.append(metadata)
            print(
                f"[extract] {year}: {metadata['filtered_records_for_scope']} registros do recorte "
                f"em {metadata['total_records_in_year_file']} registros totais"
            , flush=True)

        recorte_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame(columns=BASE_COLUMNS)
        normalized = normalize_dataframe(recorte_df)
        normalized.to_parquet(cached_normalized_path, index=False)
        print(f"[cache] recorte normalizado salvo em {cached_normalized_path}", flush=True)

    duplicate_count = int(normalized.duplicated().sum())
    if duplicate_count:
        normalized = normalized.drop_duplicates().reset_index(drop=True)
        normalized.to_parquet(cached_normalized_path, index=False)
        print(f"[quality] removidas {duplicate_count} linhas duplicadas exatas do recorte normalizado", flush=True)

    weekly = build_weekly_features(normalized)
    assignments, cluster_eval, feature_frame_with_cluster = run_clustering(weekly, min_k=min_k, max_k=max_k)
    ranking = run_feature_selection(feature_frame_with_cluster)
    profile_assignments, profile_cluster_eval, profile_feature_frame_with_cluster = run_clustering(
        weekly,
        min_k=min_k,
        max_k=max_k,
        excluded_features={"NOTIFICACOES"},
        descriptive_labels=["perfil_basal", "perfil_intermediario", "perfil_alterado", "perfil_agudo", "perfil_extremo", "perfil_raro"],
    )
    profile_ranking = run_feature_selection(profile_feature_frame_with_cluster, dropped_features={"NOTIFICACOES"})
    outputs = save_outputs(
        scope,
        metadata_rows,
        normalized,
        weekly,
        assignments,
        cluster_eval,
        ranking,
        profile_assignments,
        profile_cluster_eval,
        profile_ranking,
    )
    intensity_summary = summarize_analysis(assignments, cluster_eval, ranking)
    profile_summary = summarize_analysis(profile_assignments, profile_cluster_eval, profile_ranking)
    quality_summary, missingness = build_data_quality_summary(
        scope=scope,
        normalized=normalized,
        weekly=weekly,
        metadata_rows=metadata_rows,
        intensity_summary=intensity_summary,
        profile_summary=profile_summary,
    )
    outputs = save_quality_outputs(scope, quality_summary, missingness, outputs)
    outputs["report"] = write_markdown_report(
        scope=scope,
        tcc1_summary=tcc1_summary,
        metadata_rows=metadata_rows,
        recorte_df=normalized,
        weekly=weekly,
        intensity_summary=intensity_summary,
        profile_summary=profile_summary,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline SINAN para TCC2")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS, help="Anos do recorte analitico")
    parser.add_argument("--start-year", type=int, help="Ano inicial do recorte analitico")
    parser.add_argument("--end-year", type=int, help="Ano final do recorte analitico")
    parser.add_argument("--full-dengue-series", action="store_true", help="Usa a serie oficial completa de dengue publicada no portal, de 2000 a 2026")
    parser.add_argument("--full-brazil", action="store_true", help="Nao aplica filtro geografico e processa o Brasil inteiro")
    parser.add_argument("--uf", default=DEFAULT_UF, help="UF do recorte")
    parser.add_argument("--municipality-code", default=DEFAULT_MUNICIPALITY_CODE, help="Codigo IBGE do municipio")
    parser.add_argument("--municipality-name", default=DEFAULT_MUNICIPALITY_NAME, help="Nome do municipio")
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K, help="Menor k para avaliar no KMeans")
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K, help="Maior k para avaliar no KMeans")
    parser.add_argument("--force-download", action="store_true", help="Rebaixa os arquivos anuais mesmo se ja existirem")
    parser.add_argument("--skip-download", action="store_true", help="Nao baixa; usa apenas arquivos ja presentes em data/raw/sinan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.full_dengue_series:
        years = FULL_DENGUE_YEARS
    elif args.start_year is not None or args.end_year is not None:
        if args.start_year is None or args.end_year is None:
            raise ValueError("Use --start-year e --end-year juntos.")
        if args.start_year > args.end_year:
            raise ValueError("Ano inicial maior que ano final.")
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = sorted(args.years)

    municipality_code = args.municipality_code
    municipality_name = args.municipality_name
    uf = args.uf.upper()
    if args.full_brazil:
        municipality_code = ""
        municipality_name = ""
        uf = ""

    scope = Scope(
        years=years,
        uf=uf,
        municipality_code=municipality_code,
        municipality_name=municipality_name,
        full_brazil=args.full_brazil,
    )

    outputs = run_pipeline(
        scope=scope,
        min_k=args.min_k,
        max_k=args.max_k,
        force_download=args.force_download,
        skip_download=args.skip_download,
    )

    print("[done] artefatos gerados:", flush=True)
    for key, path in outputs.items():
        print(f"  - {key}: {path}", flush=True)


if __name__ == "__main__":
    main()

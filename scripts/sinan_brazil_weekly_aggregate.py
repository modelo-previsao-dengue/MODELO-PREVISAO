#!/usr/bin/env python3
"""
Agregacao semanal nacional do SINAN/Dengue em escala Brasil.

Este script percorre os arquivos anuais oficiais e gera uma base semanal
consolidada sem materializar todos os microdados em memoria.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import zipfile

import numpy as np
import pandas as pd

from sinan_pipeline import (
    ALARM_COLUMNS,
    COMORBIDITY_COLUMNS,
    GRAVE_COLUMNS,
    HEMORRHAGIC_COLUMNS,
    PROCESSED_DIR,
    RAW_DIR,
    RACE_MAP,
    SYMPTOM_COLUMNS,
    age_code_to_years,
    iter_json_array,
    run_clustering,
    run_feature_selection,
    summarize_analysis,
)


DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2026
SCOPE_SLUG = "brasil_2000_2026"
PROGRESS_EVERY = 500_000
MISSING_CODES = {"", "NAN", "NONE"}
GESTANTE_CODES = {"1", "2", "3", "4"}
CRITERIO_MAP = {
    "criterio_lab": {"1"},
    "criterio_clinico_epi": {"2"},
    "criterio_clinico": {"3"},
}
CLASSI_FIN_MAP = {
    "caso_confirmado_provavel": {"1", "10", "11", "12"},
    "caso_descartado": {"2"},
    "caso_inconclusivo": {"8"},
    "caso_dengue_alarme": {"11"},
    "caso_dengue_grave": {"12"},
    "caso_chikungunya": {"13"},
}

FEATURE_COLUMNS = [
    "idade_anos",
    "atraso_notificacao_dias",
    "sexo_masc",
    "sexo_fem",
    "gestante",
    "hospitalizado",
    "obito_agravo",
    "obito_outras_causas",
    "criterio_lab",
    "criterio_clinico_epi",
    "criterio_clinico",
    "caso_confirmado_provavel",
    "caso_descartado",
    "caso_inconclusivo",
    "caso_dengue_alarme",
    "caso_dengue_grave",
    "caso_chikungunya",
]
FEATURE_COLUMNS.extend([f"raca_{label}" for label in RACE_MAP.values()])
FEATURE_COLUMNS.extend([f"{column.lower()}_flag" for column in SYMPTOM_COLUMNS])
FEATURE_COLUMNS.extend([f"{column.lower()}_flag" for column in COMORBIDITY_COLUMNS])
FEATURE_COLUMNS.extend([f"{column.lower()}_flag" for column in ALARM_COLUMNS])
FEATURE_COLUMNS.extend([f"{column.lower()}_flag" for column in GRAVE_COLUMNS])
FEATURE_COLUMNS.extend([f"{column.lower()}_flag" for column in HEMORRHAGIC_COLUMNS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agregacao semanal nacional do SINAN/Dengue")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR, help="Ano inicial")
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR, help="Ano final")
    parser.add_argument("--scope-slug", default=None, help="Slug de saida para os artefatos")
    parser.add_argument("--aggregate-only", action="store_true", help="Gera apenas agregacao semanal e metadados")
    parser.add_argument("--min-k", type=int, default=2, help="Menor k para clusterizacao")
    parser.add_argument("--max-k", type=int, default=8, help="Maior k para clusterizacao")
    return parser.parse_args()


def parse_iso_date(value: object) -> Optional[date]:
    text = str(value or "").strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        return None
    try:
        return date(int(text[:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        return None


def scalar_indicator(value: object, accepted_values: Iterable[str]) -> float:
    normalized = str(value or "").strip().upper()
    if normalized in MISSING_CODES:
        return np.nan
    return 1.0 if normalized in accepted_values else 0.0


def fast_code(value: object) -> str:
    return str(value or "").strip().upper()


def fast_yes_no(value: object) -> float:
    text = str(value or "").strip()
    if text == "1":
        return 1.0
    if text in {"0", "2"}:
        return 0.0
    return np.nan


def get_ano_semana(record: Dict[str, object]) -> Optional[str]:
    for field in ["SEM_PRI", "SEM_NOT"]:
        value = str(record.get(field) or "").strip()
        if len(value) != 6 or not value.isdigit():
            continue

        year_first = int(value[:4])
        week_last = int(value[4:])
        if 1900 <= year_first <= 2100 and 1 <= week_last <= 53:
            return f"{year_first:04d}{week_last:02d}"

        week_first = int(value[:2])
        year_last = int(value[2:])
        if 1900 <= year_last <= 2100 and 1 <= week_first <= 53:
            return f"{year_last:04d}{week_first:02d}"
    return None


def compute_record_features(record: Dict[str, object]) -> Dict[str, float]:
    result: Dict[str, float] = {}

    idade_anos = age_code_to_years(record.get("NU_IDADE_N"))
    if idade_anos == idade_anos:
        result["idade_anos"] = float(idade_anos)

    dt_notific = parse_iso_date(record.get("DT_NOTIFIC"))
    dt_sin_pri = parse_iso_date(record.get("DT_SIN_PRI"))
    if dt_notific and dt_sin_pri:
        result["atraso_notificacao_dias"] = float((dt_notific - dt_sin_pri).days)

    cs_sexo = fast_code(record.get("CS_SEXO"))
    if cs_sexo not in MISSING_CODES:
        result["sexo_masc"] = 1.0 if cs_sexo == "M" else 0.0
        result["sexo_fem"] = 1.0 if cs_sexo == "F" else 0.0

    cs_gestant = fast_code(record.get("CS_GESTANT"))
    if cs_gestant not in MISSING_CODES:
        result["gestante"] = 1.0 if cs_gestant in GESTANTE_CODES else 0.0

    race_value = fast_code(record.get("CS_RACA"))
    for code, label in RACE_MAP.items():
        if race_value not in MISSING_CODES:
            race_indicator = 1.0 if race_value == code else 0.0
            result[f"raca_{label}"] = race_indicator

    hospitalizado = fast_yes_no(record.get("HOSPITALIZ"))
    if hospitalizado == hospitalizado:
        result["hospitalizado"] = hospitalizado

    evolucao = fast_code(record.get("EVOLUCAO"))
    if evolucao not in MISSING_CODES:
        result["obito_agravo"] = 1.0 if evolucao == "2" else 0.0
        result["obito_outras_causas"] = 1.0 if evolucao == "3" else 0.0

    criterio = fast_code(record.get("CRITERIO"))
    if criterio not in MISSING_CODES:
        for column, accepted in CRITERIO_MAP.items():
            result[column] = 1.0 if criterio in accepted else 0.0

    classi_fin = fast_code(record.get("CLASSI_FIN"))
    if classi_fin not in MISSING_CODES:
        for column, accepted in CLASSI_FIN_MAP.items():
            result[column] = 1.0 if classi_fin in accepted else 0.0

    for group in [SYMPTOM_COLUMNS, COMORBIDITY_COLUMNS, ALARM_COLUMNS, GRAVE_COLUMNS, HEMORRHAGIC_COLUMNS]:
        for source_column in group:
            value = fast_yes_no(record.get(source_column))
            if value == value:
                result[f"{source_column.lower()}_flag"] = value

    return result


def aggregate_weekly(start_year: int, end_year: int) -> tuple[pd.DataFrame, Dict[str, object]]:
    weeks: Dict[str, Dict[str, float]] = defaultdict(dict)
    records_by_year: Dict[int, int] = {}
    valid_week_records_by_year: Dict[int, int] = {}
    invalid_week_records_by_year: Dict[int, int] = {}
    source_files: List[Dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        zip_path = RAW_DIR / f"DENGBR{year % 100:02d}.json.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Arquivo oficial nao encontrado: {zip_path}")

        with zipfile.ZipFile(zip_path) as archive:
            json_members = [name for name in archive.namelist() if name.lower().endswith(".json")]
            if not json_members:
                raise ValueError(f"Nenhum JSON encontrado em {zip_path.name}")

            member_name = json_members[0]
            source_files.append(
                {
                    "year": year,
                    "file_name": zip_path.name,
                    "file_size_bytes": zip_path.stat().st_size,
                    "json_member": member_name,
                }
            )

            total_year = 0
            valid_week_year = 0
            invalid_week_year = 0
            with archive.open(member_name) as binary_handle:
                text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
                for record in iter_json_array(text_handle):
                    total_year += 1
                    if total_year % PROGRESS_EVERY == 0:
                        print(f"[progress] {year}: {total_year} registros lidos", flush=True)
                    ano_semana = get_ano_semana(record)
                    if not ano_semana:
                        invalid_week_year += 1
                        continue

                    valid_week_year += 1
                    bucket = weeks[ano_semana]
                    bucket["NOTIFICACOES"] = bucket.get("NOTIFICACOES", 0.0) + 1.0

                    feature_values = compute_record_features(record)
                    for feature_name, value in feature_values.items():
                        if value is None or np.isnan(value):
                            continue
                        bucket[f"{feature_name}__sum"] = bucket.get(f"{feature_name}__sum", 0.0) + float(value)
                        bucket[f"{feature_name}__count"] = bucket.get(f"{feature_name}__count", 0.0) + 1.0

            records_by_year[year] = total_year
            valid_week_records_by_year[year] = valid_week_year
            invalid_week_records_by_year[year] = invalid_week_year
            print(
                f"[aggregate] {year}: {total_year} registros, {valid_week_year} com semana epidemiologica valida",
                flush=True,
            )

    rows: List[Dict[str, object]] = []
    for ano_semana, stats in weeks.items():
        row: Dict[str, object] = {
            "ANO_SEMANA": ano_semana,
            "NOTIFICACOES": int(stats.get("NOTIFICACOES", 0.0)),
            "ANO_EPI": int(ano_semana[:4]),
            "SEMANA_EPI": int(ano_semana[4:]),
        }
        for feature_name in FEATURE_COLUMNS:
            sum_value = stats.get(f"{feature_name}__sum")
            count_value = stats.get(f"{feature_name}__count", 0.0)
            row[feature_name] = (sum_value / count_value) if sum_value is not None and count_value else np.nan
        rows.append(row)

    weekly = pd.DataFrame(rows).sort_values(["ANO_EPI", "SEMANA_EPI"]).reset_index(drop=True)
    metadata = {
        "scope_slug": SCOPE_SLUG,
        "scope_type": "full_brazil",
        "years": list(range(start_year, end_year + 1)),
        "total_records_by_year": records_by_year,
        "valid_week_records_by_year": valid_week_records_by_year,
        "invalid_week_records_by_year": invalid_week_records_by_year,
        "total_records": int(sum(records_by_year.values())),
        "valid_week_records": int(sum(valid_week_records_by_year.values())),
        "invalid_week_records": int(sum(invalid_week_records_by_year.values())),
        "epidemiological_weeks": int(len(weekly)),
        "source_files": source_files,
    }
    return weekly, metadata


def build_quality_summary(weekly: pd.DataFrame, metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "scope": {
            "scope_type": "full_brazil",
            "label": "Brasil",
            "years": metadata["years"],
        },
        "record_counts": {
            "total_records": metadata["total_records"],
            "valid_week_records": metadata["valid_week_records"],
            "invalid_week_records": metadata["invalid_week_records"],
            "weekly_rows": int(len(weekly)),
        },
        "temporal_coverage": {
            "min_ano_semana": str(weekly["ANO_SEMANA"].min()) if len(weekly) else None,
            "max_ano_semana": str(weekly["ANO_SEMANA"].max()) if len(weekly) else None,
        },
        "data_integrity": {
            "duplicated_weeks": int(weekly["ANO_SEMANA"].duplicated().sum()) if "ANO_SEMANA" in weekly.columns else 0,
            "feature_missingness_pct": {
                column: round(float(weekly[column].isna().mean() * 100), 4)
                for column in weekly.columns
                if column not in {"ANO_SEMANA", "ANO_EPI", "SEMANA_EPI"}
            },
        },
    }


def write_markdown_report(
    weekly: pd.DataFrame,
    metadata: Dict[str, object],
    intensity_summary: Optional[Dict[str, object]],
    profile_summary: Optional[Dict[str, object]],
    output_path: Path,
) -> None:
    sorted_years = sorted(metadata["total_records_by_year"].items(), key=lambda item: item[1], reverse=True)
    lines = [
        "# Relatorio SINAN Brasil - Serie Completa",
        "",
        "## Escopo",
        "",
        "- Fonte: microdados oficiais do SINAN/Dengue",
        "- Escala: Brasil inteiro",
        f"- Periodo: {metadata['years'][0]}-{metadata['years'][-1]}",
        f"- Registros totais lidos: {metadata['total_records']:,}".replace(",", "."),
        f"- Registros com semana epidemiologica valida: {metadata['valid_week_records']:,}".replace(",", "."),
        f"- Semanas epidemiologicas consolidadas: {len(weekly):,}".replace(",", "."),
        "",
        "## Maiores anos por volume bruto",
        "",
    ]
    for year, value in sorted_years[:10]:
        lines.append(f"- {year}: {value:,} registros".replace(",", "."))

    if intensity_summary and profile_summary:
        lines.extend(
            [
                "",
                "## Clusterizacao por intensidade",
                "",
                f"- Melhor k: {intensity_summary['best_k']}",
                f"- Melhor silhouette: {intensity_summary['best_silhouette']:.4f}",
                "",
                "## Clusterizacao por perfil clinico-epidemiologico",
                "",
                f"- Melhor k: {profile_summary['best_k']}",
                f"- Melhor silhouette: {profile_summary['best_silhouette']:.4f}",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("Ano inicial maior que ano final.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scope_slug = args.scope_slug or f"brasil_{args.start_year}_{args.end_year}"
    weekly, metadata = aggregate_weekly(args.start_year, args.end_year)
    metadata["scope_slug"] = scope_slug
    quality_summary = build_quality_summary(weekly, metadata)

    outputs = {
        "weekly_features": PROCESSED_DIR / f"weekly_features_{scope_slug}.csv",
        "metadata": PROCESSED_DIR / f"metadata_{scope_slug}.json",
        "quality_summary": PROCESSED_DIR / f"quality_summary_{scope_slug}.json",
        "report": PROCESSED_DIR / f"relatorio_sinan_{scope_slug}.md",
    }

    weekly.to_csv(outputs["weekly_features"], index=False)
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["quality_summary"].write_text(json.dumps(quality_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.aggregate_only:
        write_markdown_report(weekly, metadata, None, None, outputs["report"])
    else:
        assignments, cluster_eval, feature_frame_with_cluster = run_clustering(
            weekly,
            min_k=args.min_k,
            max_k=args.max_k,
        )
        ranking = run_feature_selection(feature_frame_with_cluster)
        profile_assignments, profile_cluster_eval, profile_feature_frame_with_cluster = run_clustering(
            weekly,
            min_k=args.min_k,
            max_k=args.max_k,
            excluded_features={"NOTIFICACOES"},
            descriptive_labels=["perfil_basal", "perfil_intermediario", "perfil_alterado", "perfil_agudo", "perfil_extremo", "perfil_raro", "perfil_critico", "perfil_residual"],
        )
        profile_ranking = run_feature_selection(profile_feature_frame_with_cluster, dropped_features={"NOTIFICACOES"})

        intensity_summary = summarize_analysis(assignments, cluster_eval, ranking)
        profile_summary = summarize_analysis(profile_assignments, profile_cluster_eval, profile_ranking)

        outputs["cluster_assignments"] = PROCESSED_DIR / f"cluster_assignments_{scope_slug}.csv"
        outputs["cluster_evaluation"] = PROCESSED_DIR / f"cluster_evaluation_{scope_slug}.csv"
        outputs["feature_ranking"] = PROCESSED_DIR / f"feature_ranking_{scope_slug}.csv"
        outputs["selected_features"] = PROCESSED_DIR / f"selected_features_{scope_slug}.json"
        outputs["cluster_assignments_profile"] = PROCESSED_DIR / f"cluster_assignments_profile_{scope_slug}.csv"
        outputs["cluster_evaluation_profile"] = PROCESSED_DIR / f"cluster_evaluation_profile_{scope_slug}.csv"
        outputs["feature_ranking_profile"] = PROCESSED_DIR / f"feature_ranking_profile_{scope_slug}.csv"
        outputs["selected_features_profile"] = PROCESSED_DIR / f"selected_features_profile_{scope_slug}.json"

        assignments.to_csv(outputs["cluster_assignments"], index=False)
        cluster_eval.to_csv(outputs["cluster_evaluation"], index=False)
        ranking.to_csv(outputs["feature_ranking"], index=False)
        profile_assignments.to_csv(outputs["cluster_assignments_profile"], index=False)
        profile_cluster_eval.to_csv(outputs["cluster_evaluation_profile"], index=False)
        profile_ranking.to_csv(outputs["feature_ranking_profile"], index=False)
        outputs["selected_features"].write_text(
            json.dumps(
                {
                    "scope_type": "full_brazil",
                    "scope_slug": scope_slug,
                    "selected_features": ranking.loc[ranking["selected"], "feature"].tolist(),
                    "top_15_features": ranking.head(15).to_dict(orient="records"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        outputs["selected_features_profile"].write_text(
            json.dumps(
                {
                    "scope_type": "full_brazil",
                    "scope_slug": scope_slug,
                    "analysis_mode": "clinical_profile",
                    "selected_features": profile_ranking.loc[profile_ranking["selected"], "feature"].tolist(),
                    "top_15_features": profile_ranking.head(15).to_dict(orient="records"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_markdown_report(weekly, metadata, intensity_summary, profile_summary, outputs["report"])

    print("[done] artefatos gerados:", flush=True)
    for key, path in outputs.items():
        print(f"  - {key}: {path}", flush=True)


if __name__ == "__main__":
    main()

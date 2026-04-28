#!/usr/bin/env python3
"""
Consolida agregacoes semanais nacionais em uma unica base Brasil e executa
clusterizacao, selecao de atributos e relatorio final.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from sinan_brazil_weekly_aggregate import build_quality_summary, write_markdown_report
from sinan_pipeline import PROCESSED_DIR, run_clustering, run_feature_selection, summarize_analysis


DEFAULT_CHUNKS = [
    "brasil_2000_2010",
    "brasil_2011_2016",
    "brasil_2017_2021",
    "brasil_2022_2026",
]
FINAL_SCOPE = "brasil_2000_2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidacao final do SINAN Brasil semanal")
    parser.add_argument("--chunks", nargs="+", default=DEFAULT_CHUNKS, help="Slugs parciais a consolidar")
    parser.add_argument("--scope-slug", default=FINAL_SCOPE, help="Slug final do consolidado")
    parser.add_argument("--min-k", type=int, default=2, help="Menor k para clusterizacao")
    parser.add_argument("--max-k", type=int, default=8, help="Maior k para clusterizacao")
    return parser.parse_args()


def normalize_ano_semana(value: object) -> Optional[str]:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    if not text.isdigit():
        return None

    def valid(year: int, week: int) -> bool:
        return 2000 <= year <= 2026 and 1 <= week <= 53

    if len(text) == 6:
        year_first = int(text[:4])
        week_last = int(text[4:])
        if valid(year_first, week_last):
            return f"{year_first:04d}{week_last:02d}"

        week_first = int(text[:2])
        year_last = int(text[2:])
        if valid(year_last, week_first):
            return f"{year_last:04d}{week_first:02d}"

    if len(text) == 5:
        week_first = int(text[0])
        year_last = int(text[1:])
        if valid(year_last, week_first):
            return f"{year_last:04d}{week_first:02d}"

        year_first = int(text[:4])
        week_last = int(text[4])
        if valid(year_first, week_last):
            return f"{year_first:04d}{week_last:02d}"

    if len(text) == 4:
        week = int(text[:2])
        year = 2000 + int(text[2:])
        if valid(year, week):
            return f"{year:04d}{week:02d}"

    if len(text) == 3:
        week = int(text[0])
        year = 2000 + int(text[1:])
        if valid(year, week):
            return f"{year:04d}{week:02d}"

    return None


def merge_duplicate_weeks(weekly: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    feature_columns = [column for column in weekly.columns if column not in {"ANO_SEMANA", "ANO_EPI", "SEMANA_EPI", "NOTIFICACOES"}]
    weekly = weekly.copy()
    weekly["_ANO_SEMANA_NORMALIZADO"] = weekly["ANO_SEMANA"].map(normalize_ano_semana)
    dropped_rows = int(weekly["_ANO_SEMANA_NORMALIZADO"].isna().sum())
    dropped_notifications = int(weekly.loc[weekly["_ANO_SEMANA_NORMALIZADO"].isna(), "NOTIFICACOES"].sum())
    weekly = weekly.loc[weekly["_ANO_SEMANA_NORMALIZADO"].notna()].copy()
    weekly["ANO_SEMANA"] = weekly["_ANO_SEMANA_NORMALIZADO"]
    weekly = weekly.drop(columns=["_ANO_SEMANA_NORMALIZADO"])
    weekly["ANO_EPI"] = weekly["ANO_SEMANA"].str[:4].astype(int)
    weekly["SEMANA_EPI"] = weekly["ANO_SEMANA"].str[4:].astype(int)

    rows: List[Dict[str, object]] = []
    for ano_semana, group in weekly.groupby("ANO_SEMANA", sort=True):
        row: Dict[str, object] = {
            "ANO_SEMANA": ano_semana,
            "ANO_EPI": int(ano_semana[:4]),
            "SEMANA_EPI": int(ano_semana[4:]),
            "NOTIFICACOES": int(group["NOTIFICACOES"].sum()),
        }
        weights = group["NOTIFICACOES"].astype(float)
        for column in feature_columns:
            valid = group[column].notna()
            if valid.any():
                row[column] = float((group.loc[valid, column].astype(float) * weights.loc[valid]).sum() / weights.loc[valid].sum())
            else:
                row[column] = pd.NA
        rows.append(row)

    merged = pd.DataFrame(rows).sort_values(["ANO_EPI", "SEMANA_EPI"]).reset_index(drop=True)
    diagnostics = {
        "dropped_invalid_ano_semana_rows": dropped_rows,
        "dropped_invalid_ano_semana_notifications": dropped_notifications,
    }
    return merged, diagnostics


def main() -> None:
    args = parse_args()
    weekly_frames: List[pd.DataFrame] = []
    metadata_rows: List[Dict[str, object]] = []

    for chunk in args.chunks:
        weekly_path = PROCESSED_DIR / f"weekly_features_{chunk}.csv"
        metadata_path = PROCESSED_DIR / f"metadata_{chunk}.json"
        if not weekly_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Artefatos ausentes para o chunk {chunk}")
        weekly_frames.append(pd.read_csv(weekly_path))
        metadata_rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))

    weekly = pd.concat(weekly_frames, ignore_index=True)
    weekly, merge_diagnostics = merge_duplicate_weeks(weekly)

    metadata = {
        "scope_slug": args.scope_slug,
        "scope_type": "full_brazil",
        "chunks": args.chunks,
        "years": list(range(int(weekly["ANO_EPI"].min()), int(weekly["ANO_EPI"].max()) + 1)),
        "total_records": int(sum(row["total_records"] for row in metadata_rows)),
        "valid_week_records": int(sum(row["valid_week_records"] for row in metadata_rows)),
        "invalid_week_records": int(sum(row["invalid_week_records"] for row in metadata_rows)),
        "epidemiological_weeks": int(len(weekly)),
        "total_records_by_year": {
            year: value
            for row in metadata_rows
            for year, value in row["total_records_by_year"].items()
        },
        "valid_week_records_by_year": {
            year: value
            for row in metadata_rows
            for year, value in row["valid_week_records_by_year"].items()
        },
        "invalid_week_records_by_year": {
            year: value
            for row in metadata_rows
            for year, value in row["invalid_week_records_by_year"].items()
        },
        "source_files": [item for row in metadata_rows for item in row["source_files"]],
        "merge_diagnostics": merge_diagnostics,
    }
    quality_summary = build_quality_summary(weekly, metadata)

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

    outputs = {
        "weekly_features": PROCESSED_DIR / f"weekly_features_{args.scope_slug}.csv",
        "metadata": PROCESSED_DIR / f"metadata_{args.scope_slug}.json",
        "quality_summary": PROCESSED_DIR / f"quality_summary_{args.scope_slug}.json",
        "cluster_assignments": PROCESSED_DIR / f"cluster_assignments_{args.scope_slug}.csv",
        "cluster_evaluation": PROCESSED_DIR / f"cluster_evaluation_{args.scope_slug}.csv",
        "feature_ranking": PROCESSED_DIR / f"feature_ranking_{args.scope_slug}.csv",
        "selected_features": PROCESSED_DIR / f"selected_features_{args.scope_slug}.json",
        "cluster_assignments_profile": PROCESSED_DIR / f"cluster_assignments_profile_{args.scope_slug}.csv",
        "cluster_evaluation_profile": PROCESSED_DIR / f"cluster_evaluation_profile_{args.scope_slug}.csv",
        "feature_ranking_profile": PROCESSED_DIR / f"feature_ranking_profile_{args.scope_slug}.csv",
        "selected_features_profile": PROCESSED_DIR / f"selected_features_profile_{args.scope_slug}.json",
        "report": PROCESSED_DIR / f"relatorio_sinan_{args.scope_slug}.md",
    }

    weekly.to_csv(outputs["weekly_features"], index=False)
    outputs["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["quality_summary"].write_text(json.dumps(quality_summary, indent=2, ensure_ascii=False), encoding="utf-8")
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
                "scope_slug": args.scope_slug,
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
                "scope_slug": args.scope_slug,
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

    print("[done] consolidado final gerado:", flush=True)
    for key, path in outputs.items():
        print(f"  - {key}: {path}", flush=True)


if __name__ == "__main__":
    main()

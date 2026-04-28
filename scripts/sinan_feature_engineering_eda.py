#!/usr/bin/env python3
"""
Fase 3 do TCC2: engenharia de atributos e analise exploratoria do SINAN.

O script consome os artefatos semanais gerados por `scripts/sinan_pipeline.py`
e produz uma camada explicita de features para modelagem, alem de relatorios
exploratorios prontos para incorporacao na monografia.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed" / "sinan"

DEFAULT_SCOPE_SLUG = "brasilia_2022_2024"
DEFAULT_TOP_CORRELATIONS = 25

SYMPTOM_COLUMNS = [
    "febre_flag",
    "mialgia_flag",
    "cefaleia_flag",
    "exantema_flag",
    "vomito_flag",
    "nausea_flag",
    "dor_costas_flag",
    "conjuntvit_flag",
    "artrite_flag",
    "artralgia_flag",
    "petequia_n_flag",
    "leucopenia_flag",
    "laco_flag",
    "dor_retro_flag",
]

COMORBIDITY_COLUMNS = [
    "diabetes_flag",
    "hematolog_flag",
    "hepatopat_flag",
    "renal_flag",
    "hipertensa_flag",
    "acido_pept_flag",
    "auto_imune_flag",
]

ALARM_COLUMNS = [
    "alrm_hipot_flag",
    "alrm_plaq_flag",
    "alrm_vom_flag",
    "alrm_sang_flag",
    "alrm_hemat_flag",
    "alrm_abdom_flag",
    "alrm_letar_flag",
    "alrm_hepat_flag",
    "alrm_liq_flag",
]

SEVERE_COLUMNS = [
    "grav_pulso_flag",
    "grav_conv_flag",
    "grav_ench_flag",
    "grav_insuf_flag",
    "grav_taqui_flag",
    "grav_extre_flag",
    "grav_hipot_flag",
    "grav_hemat_flag",
    "grav_melen_flag",
    "grav_metro_flag",
    "grav_sang_flag",
    "grav_ast_flag",
    "grav_mioc_flag",
    "grav_consc_flag",
    "grav_orgao_flag",
]

HEMORRHAGIC_COLUMNS = [
    "mani_hemor_flag",
    "epistaxe_flag",
    "gengivo_flag",
    "metro_flag",
    "petequias_flag",
    "hematura_flag",
    "sangram_flag",
    "laco_n_flag",
    "plasmatico_flag",
    "evidencia_flag",
    "plaq_menor_flag",
    "con_fhd_flag",
]

CORE_EDA_COLUMNS = [
    "NOTIFICACOES",
    "idade_anos",
    "atraso_notificacao_dias",
    "hospitalizado",
    "caso_confirmado_provavel",
    "caso_inconclusivo",
    "caso_dengue_alarme",
    "caso_dengue_grave",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engenharia de atributos e EDA do SINAN")
    parser.add_argument("--scope-slug", default=DEFAULT_SCOPE_SLUG, help="Slug dos artefatos do recorte")
    parser.add_argument(
        "--top-correlations",
        type=int,
        default=DEFAULT_TOP_CORRELATIONS,
        help="Quantidade de correlacoes a reportar",
    )
    return parser.parse_args()


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    return pd.read_csv(path)


def parse_ano_semana_to_date(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.extract(r"(?P<year>\d{4})(?P<week>\d{2})")
    iso = text["year"].fillna("") + text["week"].fillna("") + "1"
    return pd.to_datetime(iso, format="%G%V%u", errors="coerce")


def available_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [column for column in candidates if column in df.columns]


def safe_mean(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = available_columns(df, columns)
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].mean(axis=1, skipna=True)


def engineer_features(weekly: pd.DataFrame) -> pd.DataFrame:
    frame = weekly.copy().sort_values(["ANO_EPI", "SEMANA_EPI"]).reset_index(drop=True)
    frame["week_start"] = parse_ano_semana_to_date(frame["ANO_SEMANA"])
    frame["ano"] = frame["ANO_EPI"].astype("Int64")
    frame["semana"] = frame["SEMANA_EPI"].astype("Int64")
    frame["semana_sin"] = np.sin(2 * np.pi * frame["SEMANA_EPI"] / 52.0)
    frame["semana_cos"] = np.cos(2 * np.pi * frame["SEMANA_EPI"] / 52.0)

    notifications = frame["NOTIFICACOES"].astype(float)
    for lag in [1, 2, 3, 4, 8, 12]:
        frame[f"notificacoes_lag_{lag}"] = notifications.shift(lag)

    for window in [3, 4, 8, 12]:
        shifted = notifications.shift(1)
        frame[f"notificacoes_media_movel_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
        frame[f"notificacoes_desvio_movel_{window}"] = shifted.rolling(window=window, min_periods=2).std()
        frame[f"notificacoes_min_movel_{window}"] = shifted.rolling(window=window, min_periods=1).min()
        frame[f"notificacoes_max_movel_{window}"] = shifted.rolling(window=window, min_periods=1).max()

    frame["notificacoes_diff_1"] = notifications.diff(1)
    frame["notificacoes_diff_4"] = notifications.diff(4)
    frame["notificacoes_pct_change_1"] = notifications.pct_change(1).replace([np.inf, -np.inf], np.nan)
    frame["notificacoes_pct_change_4"] = notifications.pct_change(4).replace([np.inf, -np.inf], np.nan)
    frame["notificacoes_aceleracao_1"] = frame["notificacoes_diff_1"].diff(1)
    frame["notificacoes_razao_media_4"] = notifications / frame["notificacoes_media_movel_4"]
    frame["notificacoes_razao_media_8"] = notifications / frame["notificacoes_media_movel_8"]

    frame["indice_sintomas"] = safe_mean(frame, SYMPTOM_COLUMNS)
    frame["indice_comorbidades"] = safe_mean(frame, COMORBIDITY_COLUMNS)
    frame["indice_alarme"] = safe_mean(frame, ALARM_COLUMNS)
    frame["indice_gravidade"] = safe_mean(frame, SEVERE_COLUMNS)
    frame["indice_hemorragico"] = safe_mean(frame, HEMORRHAGIC_COLUMNS)

    frame["indice_carga_clinica"] = frame[
        ["indice_sintomas", "indice_comorbidades", "indice_alarme", "indice_gravidade", "indice_hemorragico"]
    ].mean(axis=1, skipna=True)
    frame["indice_desfecho_severo"] = frame[
        [
            "hospitalizado",
            "caso_dengue_alarme",
            "caso_dengue_grave",
            "obito_agravo",
        ]
    ].mean(axis=1, skipna=True)
    frame["indice_confirmacao"] = frame["caso_confirmado_provavel"] - frame["caso_inconclusivo"]

    q75 = float(frame["NOTIFICACOES"].quantile(0.75))
    q90 = float(frame["NOTIFICACOES"].quantile(0.90))
    frame["alerta_notificacoes_q75"] = (frame["NOTIFICACOES"] >= q75).astype(int)
    frame["alerta_notificacoes_q90"] = (frame["NOTIFICACOES"] >= q90).astype(int)
    frame["alerta_crescimento"] = (
        frame["notificacoes_pct_change_1"].fillna(0.0) >= frame["notificacoes_pct_change_1"].quantile(0.75)
    ).astype(int)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_feature_catalog(features: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = {
        "identificacao": ["ANO_SEMANA", "ANO_EPI", "SEMANA_EPI", "week_start", "ano", "semana"],
        "volume": [
            "NOTIFICACOES",
            "notificacoes_lag_1",
            "notificacoes_lag_2",
            "notificacoes_lag_3",
            "notificacoes_lag_4",
            "notificacoes_lag_8",
            "notificacoes_lag_12",
        ],
        "janela_movel": [column for column in features.columns if column.startswith("notificacoes_") and "movel" in column],
        "variacao": [
            "notificacoes_diff_1",
            "notificacoes_diff_4",
            "notificacoes_pct_change_1",
            "notificacoes_pct_change_4",
            "notificacoes_aceleracao_1",
            "notificacoes_razao_media_4",
            "notificacoes_razao_media_8",
        ],
        "sazonalidade": ["semana_sin", "semana_cos"],
        "perfil_clinico": [
            "indice_sintomas",
            "indice_comorbidades",
            "indice_alarme",
            "indice_gravidade",
            "indice_hemorragico",
            "indice_carga_clinica",
            "indice_desfecho_severo",
            "indice_confirmacao",
        ],
        "targets_auxiliares": ["alerta_notificacoes_q75", "alerta_notificacoes_q90", "alerta_crescimento"],
    }

    descriptions = {
        "NOTIFICACOES": "Contagem semanal de notificacoes no recorte do SINAN.",
        "notificacoes_lag_1": "Volume de notificacoes na semana epidemiologica imediatamente anterior.",
        "notificacoes_lag_2": "Volume de notificacoes observado duas semanas antes.",
        "notificacoes_lag_3": "Volume de notificacoes observado tres semanas antes.",
        "notificacoes_lag_4": "Volume de notificacoes observado quatro semanas antes.",
        "notificacoes_lag_8": "Volume de notificacoes observado oito semanas antes.",
        "notificacoes_lag_12": "Volume de notificacoes observado doze semanas antes.",
        "notificacoes_diff_1": "Variacao absoluta semanal das notificacoes.",
        "notificacoes_diff_4": "Variacao absoluta em relacao a quatro semanas antes.",
        "notificacoes_pct_change_1": "Variacao percentual semanal das notificacoes.",
        "notificacoes_pct_change_4": "Variacao percentual em relacao a quatro semanas antes.",
        "notificacoes_aceleracao_1": "Aceleracao semanal do crescimento das notificacoes.",
        "notificacoes_razao_media_4": "Razao entre a semana corrente e a media movel das quatro semanas anteriores.",
        "notificacoes_razao_media_8": "Razao entre a semana corrente e a media movel das oito semanas anteriores.",
        "semana_sin": "Componente senoidal para representar sazonalidade anual.",
        "semana_cos": "Componente cossenoidal para representar sazonalidade anual.",
        "indice_sintomas": "Media semanal das flags de sintomas classicos de dengue.",
        "indice_comorbidades": "Media semanal das comorbidades notificadas.",
        "indice_alarme": "Media semanal dos sinais de alarme.",
        "indice_gravidade": "Media semanal dos marcadores de dengue grave.",
        "indice_hemorragico": "Media semanal dos marcadores hemorragicos.",
        "indice_carga_clinica": "Indicador composto do perfil clinico semanal.",
        "indice_desfecho_severo": "Indicador composto de hospitalizacao, alarme, gravidade e obito por agravo.",
        "indice_confirmacao": "Diferenca entre proporcao de casos confirmados/provaveis e inconclusivos.",
        "alerta_notificacoes_q75": "Rotulo auxiliar para semanas no quartil superior de notificacoes.",
        "alerta_notificacoes_q90": "Rotulo auxiliar para semanas no decil superior de notificacoes.",
        "alerta_crescimento": "Rotulo auxiliar para semanas com crescimento semanal elevado.",
    }

    for group, columns in groups.items():
        for column in columns:
            if column not in features.columns:
                continue
            rows.append(
                {
                    "feature": column,
                    "grupo": group,
                    "dtype": str(features[column].dtype),
                    "missing_pct": round(float(features[column].isna().mean() * 100), 4),
                    "descricao": descriptions.get(column, "Atributo derivado para modelagem e analise temporal."),
                }
            )

    return pd.DataFrame(rows).sort_values(["grupo", "feature"]).reset_index(drop=True)


def build_descriptive_stats(features: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for column in available_columns(features, CORE_EDA_COLUMNS):
        series = pd.to_numeric(features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "missing_pct": round(float(series.isna().mean() * 100), 4),
                "mean": round(float(series.mean()), 6),
                "median": round(float(series.median()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "q25": round(float(series.quantile(0.25)), 6),
                "q75": round(float(series.quantile(0.75)), 6),
                "max": round(float(series.max()), 6),
            }
        )
    return pd.DataFrame(rows)


def build_yearly_summary(features: pd.DataFrame) -> pd.DataFrame:
    grouped = features.groupby("ano", dropna=False).agg(
        semanas=("ANO_SEMANA", "count"),
        notificacoes_total=("NOTIFICACOES", "sum"),
        notificacoes_media=("NOTIFICACOES", "mean"),
        notificacoes_mediana=("NOTIFICACOES", "median"),
        notificacoes_pico=("NOTIFICACOES", "max"),
        sintomas_medios=("indice_sintomas", "mean"),
        alarme_medio=("indice_alarme", "mean"),
        gravidade_media=("indice_gravidade", "mean"),
        atraso_medio=("atraso_notificacao_dias", "mean"),
    )
    return grouped.reset_index().rename(columns={"ano": "ANO"})


def build_peak_weeks(features: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    columns = [
        "ANO_SEMANA",
        "week_start",
        "NOTIFICACOES",
        "notificacoes_pct_change_1",
        "indice_sintomas",
        "indice_alarme",
        "indice_gravidade",
        "indice_desfecho_severo",
    ]
    selected = available_columns(features, columns)
    return features[selected].sort_values("NOTIFICACOES", ascending=False).head(top_n).reset_index(drop=True)


def build_correlation_table(features: pd.DataFrame, top_n: int) -> pd.DataFrame:
    numeric = features.select_dtypes(include=[np.number]).copy()
    corr = numeric.corr(numeric_only=True)["NOTIFICACOES"].drop(labels=["NOTIFICACOES"])
    table = (
        corr.reset_index()
        .rename(columns={"index": "feature", "NOTIFICACOES": "pearson_corr_notificacoes"})
        .assign(abs_corr=lambda df: df["pearson_corr_notificacoes"].abs())
        .sort_values(["abs_corr", "pearson_corr_notificacoes"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    return table


def build_cluster_summary(features: pd.DataFrame, intensity: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    merged = (
        features.merge(
            intensity[["ANO_SEMANA", "CLUSTER_LABEL"]].rename(columns={"CLUSTER_LABEL": "cluster_intensidade"}),
            on="ANO_SEMANA",
            how="left",
        ).merge(
            profile[["ANO_SEMANA", "CLUSTER_LABEL"]].rename(columns={"CLUSTER_LABEL": "cluster_perfil"}),
            on="ANO_SEMANA",
            how="left",
        )
    )

    summary = merged.groupby(["cluster_intensidade", "cluster_perfil"], dropna=False).agg(
        semanas=("ANO_SEMANA", "count"),
        notificacoes_medias=("NOTIFICACOES", "mean"),
        sintomas_medios=("indice_sintomas", "mean"),
        alarme_medio=("indice_alarme", "mean"),
        gravidade_media=("indice_gravidade", "mean"),
        atraso_medio=("atraso_notificacao_dias", "mean"),
    )
    return summary.reset_index().sort_values("notificacoes_medias", ascending=False).reset_index(drop=True)


def build_summary_payload(
    scope_slug: str,
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    peak_weeks: pd.DataFrame,
) -> Dict[str, object]:
    peak_row = peak_weeks.iloc[0]
    total_notifications = int(features["NOTIFICACOES"].sum())
    mean_notifications = float(features["NOTIFICACOES"].mean())
    std_notifications = float(features["NOTIFICACOES"].std())
    cv = std_notifications / mean_notifications if mean_notifications else None

    return {
        "scope_slug": scope_slug,
        "weeks": int(len(features)),
        "period": {
            "start_week": str(features["ANO_SEMANA"].min()),
            "end_week": str(features["ANO_SEMANA"].max()),
            "start_date": str(features["week_start"].min().date()) if features["week_start"].notna().any() else None,
            "end_date": str(features["week_start"].max().date()) if features["week_start"].notna().any() else None,
        },
        "notifications": {
            "total": total_notifications,
            "mean": round(mean_notifications, 4),
            "median": round(float(features["NOTIFICACOES"].median()), 4),
            "std": round(std_notifications, 4),
            "cv": round(float(cv), 4) if cv is not None else None,
            "q75_threshold": round(float(features["NOTIFICACOES"].quantile(0.75)), 4),
            "q90_threshold": round(float(features["NOTIFICACOES"].quantile(0.90)), 4),
        },
        "peak_week": {
            "ano_semana": str(peak_row["ANO_SEMANA"]),
            "week_start": str(peak_row["week_start"]) if pd.notna(peak_row["week_start"]) else None,
            "notificacoes": int(peak_row["NOTIFICACOES"]),
        },
        "top_correlations": correlations.to_dict(orient="records"),
        "yearly_summary": yearly_summary.to_dict(orient="records"),
    }


def write_markdown_report(
    output_path: Path,
    summary: Dict[str, object],
    feature_catalog: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    peak_weeks: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Engenharia de Atributos e Analise Exploratoria - SINAN",
        "",
        "## 1. Escopo",
        "",
        f"- Recorte processado: `{summary['scope_slug']}`",
        f"- Semanas epidemiologicas analisadas: `{summary['weeks']}`",
        f"- Cobertura temporal: `{summary['period']['start_week']}` ate `{summary['period']['end_week']}`",
        f"- Intervalo calendario aproximado: `{summary['period']['start_date']}` ate `{summary['period']['end_date']}`",
        "",
        "## 2. Objetivo da etapa 3",
        "",
        "- Transformar a agregacao semanal do SINAN em uma camada explicita de atributos para modelagem preditiva.",
        "- Caracterizar o comportamento temporal, clinico e epidemiologico do recorte antes do treinamento dos modelos.",
        "- Produzir artefatos reprodutiveis para justificar a escolha de atributos no TCC2.",
        "",
        "## 3. Engenharia de atributos",
        "",
        "- Blocos de atributos gerados: volume historico, janelas moveis, variacao temporal, sazonalidade, perfis clinicos compostos e rotulos auxiliares.",
        f"- Total de atributos catalogados na nova camada: `{len(feature_catalog)}`",
        "",
        "Principais grupos:",
        "",
    ]

    for group, count in feature_catalog.groupby("grupo")["feature"].count().sort_index().items():
        lines.append(f"- `{group}`: {int(count)} atributos")

    lines.extend(
        [
            "",
            "## 4. Comportamento das notificacoes",
            "",
            f"- Total de notificacoes no periodo: `{summary['notifications']['total']}`",
            f"- Media semanal: `{summary['notifications']['mean']}`",
            f"- Mediana semanal: `{summary['notifications']['median']}`",
            f"- Desvio padrao semanal: `{summary['notifications']['std']}`",
            f"- Coeficiente de variacao: `{summary['notifications']['cv']}`",
            f"- Limiar de semanas altas (Q75): `{summary['notifications']['q75_threshold']}`",
            f"- Limiar de semanas extremas (Q90): `{summary['notifications']['q90_threshold']}`",
            f"- Semana de pico: `{summary['peak_week']['ano_semana']}` com `{summary['peak_week']['notificacoes']}` notificacoes",
            "",
            "## 5. Resumo anual",
            "",
        ]
    )

    for _, row in yearly_summary.iterrows():
        lines.append(
            f"- {int(row['ANO'])}: {int(row['notificacoes_total'])} notificacoes, "
            f"media semanal {row['notificacoes_media']:.2f}, pico {row['notificacoes_pico']:.0f}"
        )

    lines.extend(
        [
            "",
            "## 6. Correlacoes com notificacoes",
            "",
            "Atributos com maior associacao linear absoluta com `NOTIFICACOES`:",
            "",
        ]
    )

    for _, row in correlations.iterrows():
        lines.append(f"- `{row['feature']}`: correlacao de Pearson `{row['pearson_corr_notificacoes']:.4f}`")

    lines.extend(
        [
            "",
            "## 7. Semanas de pico",
            "",
        ]
    )

    for _, row in peak_weeks.head(8).iterrows():
        pct_change = row["notificacoes_pct_change_1"]
        pct_text = "NA" if pd.isna(pct_change) else f"{pct_change:.4f}"
        lines.append(
            f"- `{row['ANO_SEMANA']}`: {int(row['NOTIFICACOES'])} notificacoes, "
            f"variacao semanal `{pct_text}`, indice de alarme `{row['indice_alarme']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## 8. Leitura dos clusters",
            "",
            "- O cruzamento entre cluster de intensidade e cluster clinico ajuda a diferenciar semanas apenas volumosas de semanas volumosas com mudanca no perfil clinico.",
            "",
        ]
    )

    for _, row in cluster_summary.head(10).iterrows():
        lines.append(
            f"- Intensidade `{row['cluster_intensidade']}` + perfil `{row['cluster_perfil']}`: "
            f"{int(row['semanas'])} semanas, media `{row['notificacoes_medias']:.1f}` notificacoes"
        )

    lines.extend(
        [
            "",
            "## 9. Uso na proxima fase",
            "",
            "- Os lags e janelas moveis permitem modelagem temporal autoregressiva.",
            "- Os indicadores compostos clinicos agregam dezenas de sinais em variaveis de interpretacao mais simples.",
            "- Os rotulos auxiliares podem apoiar tarefas de classificacao de alerta e comparacao com modelos de regressao.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    scope_slug = args.scope_slug

    weekly_path = PROCESSED_DIR / f"weekly_features_{scope_slug}.csv"
    intensity_path = PROCESSED_DIR / f"cluster_assignments_{scope_slug}.csv"
    profile_path = PROCESSED_DIR / f"cluster_assignments_profile_{scope_slug}.csv"

    weekly = load_required_csv(weekly_path)
    intensity = load_required_csv(intensity_path)
    profile = load_required_csv(profile_path)

    features = engineer_features(weekly)
    feature_catalog = build_feature_catalog(features)
    descriptive_stats = build_descriptive_stats(features)
    yearly_summary = build_yearly_summary(features)
    peak_weeks = build_peak_weeks(features)
    correlations = build_correlation_table(features, top_n=args.top_correlations)
    cluster_summary = build_cluster_summary(features, intensity, profile)
    summary = build_summary_payload(scope_slug, features, correlations, yearly_summary, peak_weeks)

    outputs = {
        "weekly_model_features": PROCESSED_DIR / f"weekly_model_features_{scope_slug}.csv",
        "feature_catalog": PROCESSED_DIR / f"feature_catalog_{scope_slug}.csv",
        "eda_descriptive_stats": PROCESSED_DIR / f"eda_descriptive_stats_{scope_slug}.csv",
        "eda_yearly_summary": PROCESSED_DIR / f"eda_yearly_summary_{scope_slug}.csv",
        "eda_peak_weeks": PROCESSED_DIR / f"eda_peak_weeks_{scope_slug}.csv",
        "eda_correlations": PROCESSED_DIR / f"eda_correlations_{scope_slug}.csv",
        "eda_cluster_summary": PROCESSED_DIR / f"eda_cluster_summary_{scope_slug}.csv",
        "eda_summary": PROCESSED_DIR / f"eda_summary_{scope_slug}.json",
        "eda_report": PROCESSED_DIR / f"relatorio_eda_{scope_slug}.md",
    }

    features.to_csv(outputs["weekly_model_features"], index=False)
    feature_catalog.to_csv(outputs["feature_catalog"], index=False)
    descriptive_stats.to_csv(outputs["eda_descriptive_stats"], index=False)
    yearly_summary.to_csv(outputs["eda_yearly_summary"], index=False)
    peak_weeks.to_csv(outputs["eda_peak_weeks"], index=False)
    correlations.to_csv(outputs["eda_correlations"], index=False)
    cluster_summary.to_csv(outputs["eda_cluster_summary"], index=False)
    outputs["eda_summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(
        output_path=outputs["eda_report"],
        summary=summary,
        feature_catalog=feature_catalog,
        yearly_summary=yearly_summary,
        correlations=correlations,
        peak_weeks=peak_weeks,
        cluster_summary=cluster_summary,
    )

    print("[done] artefatos gerados:", flush=True)
    for key, path in outputs.items():
        print(f"  - {key}: {path}", flush=True)


if __name__ == "__main__":
    main()

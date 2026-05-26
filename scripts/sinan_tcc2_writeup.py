#!/usr/bin/env python3
"""
Gera material documental do TCC2 a partir dos artefatos oficiais da trilha SINAN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "sinan"
DOCS_DIR = ROOT / "docs"
DEFAULT_VERSION = "sinan_tcc2_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerador de write-up da trilha SINAN TCC2")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def maybe_read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def top_missing_columns(path: Path, limit: int = 10) -> List[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "column" not in df.columns or "missing_pct" not in df.columns:
        return []
    df = df.sort_values(["missing_pct", "column"], ascending=[False, True]).head(limit)
    return [
        f"{row['column']} ({float(row['missing_pct']):.2f}%)"
        for row in df.to_dict(orient="records")
    ]


def build_paths(version: str) -> Dict[str, Path]:
    return {
        "bronze": DATA_ROOT / "bronze" / version,
        "silver": DATA_ROOT / "silver" / version,
        "gold": DATA_ROOT / "gold" / version,
        "governance": DATA_ROOT / "governance" / version,
        "serving": DATA_ROOT / "serving" / version,
    }


def summarize_selected_features(path: Path) -> List[str]:
    if not path.exists():
        return []
    payload = read_json(path)
    return payload.get("selected_features", [])


def format_int(value: object) -> str:
    return f"{int(value):,}".replace(",", ".")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def main() -> None:
    args = parse_args()
    paths = build_paths(args.version)

    silver_quality = read_json(paths["governance"] / "quality_summary_silver.json")
    gold_quality = read_json(paths["governance"] / "quality_summary_gold.json")
    run_manifest = read_json(paths["governance"] / "run_manifest.json")
    silver_schema = read_json(paths["governance"] / "schema_silver.json")
    gold_schema = read_json(paths["governance"] / "schema_gold.json")
    bronze_inventory = maybe_read_csv(paths["bronze"] / "inventory" / "sinan_bronze_inventory.csv")
    eda_summary = read_json(paths["gold"] / "analytics" / "eda_summary.json")
    yearly_summary = maybe_read_csv(paths["gold"] / "analytics" / "eda_yearly_summary.csv")
    peak_weeks = maybe_read_csv(paths["gold"] / "analytics" / "eda_peak_weeks.csv")
    intensity_summary = maybe_read_csv(paths["gold"] / "analytics" / "cluster_summary_intensity.csv")
    profile_summary = maybe_read_csv(paths["gold"] / "analytics" / "cluster_summary_profile.csv")
    intensity_features = summarize_selected_features(paths["gold"] / "analytics" / "selected_features_intensity.json")
    profile_features = summarize_selected_features(paths["gold"] / "analytics" / "selected_features_profile.json")

    methodology_lines = [
        "# Material TCC2 - Trilha SINAN",
        "",
        f"- Versao oficial: `{args.version}`",
        "",
        "## Metodologia executada",
        "",
        "A trilha SINAN do TCC2 foi consolidada como pipeline nacional em `municipio-semana`, com preservacao do bronze oficial anual, agregacao observada na silver e densificacao da serie na gold para modelagem e alertas.",
        "",
        "Fluxo executado:",
        "",
        "1. Inventario e checksum dos arquivos anuais oficiais do SINAN/Dengue.",
        "2. Parsing incremental dos JSON anuais, sem materializar toda a serie em memoria.",
        "3. Reconciliacao territorial dos codigos municipais com a tabela oficial do IBGE.",
        "4. Derivacao e validacao de semana epidemiologica por registro, priorizando a semana de notificacao.",
        "5. Remocao de duplicidades exatas por assinatura canonica do registro.",
        "6. Agregacao para `ibge_municipio + ano_semana` na camada silver.",
        "7. Densificacao temporal por municipio na camada gold, com `notificacoes = 0` em semanas sem casos.",
        "8. Engenharia de atributos temporais e clinico-epidemiologicos.",
        "9. Clusterizacao exploratoria em amostra ativa da gold e selecao de atributos.",
        "10. Geracao de schema, lineage, missingness, discard report e camada operacional.",
        "",
        "## Contratos principais",
        "",
        f"- Silver: `{silver_schema['granularity']}`",
        f"- Gold: `{gold_schema['granularity']}`",
        f"- Chave primaria oficial: `{', '.join(gold_schema['primary_key'])}`",
        "",
        "## Resultados consolidados",
        "",
        f"- Notificacoes aproveitadas na silver: `{format_int(silver_quality['record_counts']['total_notifications'])}`",
        f"- Linhas da silver observada: `{format_int(silver_quality['record_counts']['rows'])}`",
        f"- Municipios cobertos na silver: `{format_int(silver_quality['record_counts']['municipalities'])}`",
        f"- Linhas da gold densa: `{format_int(gold_quality['record_counts']['rows'])}`",
        f"- Janela final: `{silver_quality['coverage']['min_ano_semana']}` ate `{silver_quality['coverage']['max_ano_semana']}`",
        "",
    ]

    if peak_weeks is not None and not peak_weeks.empty:
        methodology_lines.extend(
            [
                "## Principais semanas nacionais",
                "",
            ]
        )
        for row in peak_weeks.head(10).to_dict(orient="records"):
            methodology_lines.append(
                f"- {row['ano_semana']}: {format_int(row['notificacoes_total'])} notificacoes"
            )

    methodology_path = DOCS_DIR / "sinan_tcc2_writeup.md"
    methodology_path.write_text("\n".join(methodology_lines), encoding="utf-8")

    results_lines = [
        "# Resultados SINAN TCC2",
        "",
        f"- Versao analisada: `{args.version}`",
        "",
        "## Bronze",
        "",
        f"- Arquivos anuais inventariados: `{len(bronze_inventory)}`",
        f"- Registros brutos lidos: `{format_int(bronze_inventory['total_records'].sum())}`",
        f"- Duplicatas exatas removidas: `{format_int(bronze_inventory['duplicate_records_removed'].sum())}`",
        "",
        "## Silver",
        "",
        f"- Linhas observadas: `{format_int(silver_quality['record_counts']['rows'])}`",
        f"- Municipios cobertos: `{format_int(silver_quality['record_counts']['municipalities'])}`",
        f"- Descartes por semana invalida: `{format_int(silver_quality['discard_summary']['invalid_week_records'])}`",
        f"- Descartes por municipio invalido: `{format_int(silver_quality['discard_summary']['invalid_municipality_records'])}`",
        "",
        "## Gold",
        "",
        f"- Linhas densas: `{format_int(gold_quality['record_counts']['rows'])}`",
        f"- Amostra ativa usada para analytics: `{format_int(gold_quality['sampling']['analytics_sample_rows'])}`",
        "",
    ]

    if yearly_summary is not None and not yearly_summary.empty:
        results_lines.extend(
            [
                "## Resumo anual",
                "",
            ]
        )
        for row in yearly_summary.sort_values("notificacoes_total", ascending=False).head(10).to_dict(orient="records"):
            results_lines.append(
                f"- {int(row['ano'])}: {format_int(row['notificacoes_total'])} notificacoes, pico municipal-semana {format_int(row['notificacoes_max'])}"
            )

    if intensity_summary is not None and not intensity_summary.empty:
        results_lines.extend(
            [
                "",
                "## Regimes de cluster - intensidade",
                "",
            ]
        )
        for row in intensity_summary.to_dict(orient="records"):
            results_lines.append(
                f"- {row['cluster_label']}: {format_int(row['rows'])} linhas, media {float(row['media_notificacoes']):.2f} notificacoes"
            )

    if profile_summary is not None and not profile_summary.empty:
        results_lines.extend(
            [
                "",
                "## Regimes de cluster - perfil",
                "",
            ]
        )
        for row in profile_summary.to_dict(orient="records"):
            results_lines.append(
                f"- {row['cluster_label']}: {format_int(row['rows'])} linhas, media {float(row['media_notificacoes']):.2f} notificacoes"
            )

    if intensity_features:
        results_lines.extend(
            [
                "",
                "## Principais atributos selecionados - intensidade",
                "",
            ]
        )
        for item in intensity_features[:15]:
            results_lines.append(f"- {item}")

    if profile_features:
        results_lines.extend(
            [
                "",
                "## Principais atributos selecionados - perfil",
                "",
            ]
        )
        for item in profile_features[:15]:
            results_lines.append(f"- {item}")

    results_path = DOCS_DIR / "sinan_tcc2_results.md"
    results_path.write_text("\n".join(results_lines), encoding="utf-8")

    top_missing_silver = top_missing_columns(paths["governance"] / "missingness_silver.csv", limit=8)
    top_missing_gold = top_missing_columns(paths["governance"] / "missingness_gold.csv", limit=8)
    top_year_row = yearly_summary.sort_values("notificacoes_total", ascending=False).head(1)
    peak_week_row = peak_weeks.sort_values("notificacoes_total", ascending=False).head(1)
    top_year_text = ""
    if not top_year_row.empty:
        row = top_year_row.iloc[0]
        top_year_text = (
            f"O ano com maior volume agregado de notificacoes na serie foi {int(row['ano'])}, "
            f"com {format_int(row['notificacoes_total'])} notificacoes consolidadas na camada gold."
        )
    top_peak_text = ""
    if not peak_week_row.empty:
        row = peak_week_row.iloc[0]
        top_peak_text = (
            f"A maior semana nacional observada foi {row['ano_semana']}, "
            f"com {format_int(row['notificacoes_total'])} notificacoes somadas entre municipios."
        )

    text_lines = [
        "# Texto-Base TCC2 - Trilha SINAN",
        "",
        f"- Versao oficial: `{args.version}`",
        "",
        "## Metodologia",
        "",
        "A trilha SINAN do TCC2 foi consolidada como uma pipeline nacional reproduzivel e auditavel em nivel municipio-semana. O processo partiu dos arquivos anuais oficiais de dengue do SINAN, preservados em Bronze, seguiu para uma Silver com reconciliacao territorial, normalizacao de tipos, remocao de duplicidades exatas e agregacao por `ibge_municipio + ano_semana`, e culminou em uma Gold densa com atributos temporais e clinico-epidemiologicos prontos para modelagem.",
        "",
        "A regra temporal oficial foi congelada sobre semana epidemiologica de notificacao. A derivacao priorizou `SEM_NOT`, depois `DT_NOTIFIC`, e somente usou `SEM_PRI` ou `DT_SIN_PRI` como fallback controlado, para evitar contaminar a serie oficial com semanas antigas, ambiguas ou semanticamente inadequadas para a contagem de notificacoes.",
        "",
        "## Execucao real",
        "",
        f"Foram inventariados `{len(bronze_inventory)}` arquivos anuais oficiais, cobrindo a janela operacional `2000-2026`. A camada Silver aproveitou `{format_int(silver_quality['record_counts']['total_notifications'])}` notificacoes validas, distribuidas em `{format_int(silver_quality['record_counts']['rows'])}` linhas observadas de municipio-semana e `{format_int(silver_quality['record_counts']['municipalities'])}` municipios unicos.",
        "",
        f"A camada Gold produziu `{format_int(gold_quality['record_counts']['rows'])}` linhas densas para consumo analitico e preditivo. A governanca final incluiu schema versionado, lineage, quality summaries, missingness, discard report, manifesto de execucao e camada operacional para armazenamento em objetos e carga em PostgreSQL/Supabase.",
        "",
        "## Resultados nacionais",
        "",
        top_year_text or "O resumo anual consolidado deve ser lido em `eda_yearly_summary.csv`.",
        top_peak_text or "As semanas de pico devem ser lidas em `eda_peak_weeks.csv`.",
        "",
        "A analise exploratoria foi complementada por clusterizacao de semanas e selecao de atributos, produzindo regimes exploratorios e rankings reproduziveis para apoiar a etapa de modelagem preditiva.",
        "",
        "## Limitacoes",
        "",
        "- a serie depende da qualidade e estabilidade dos microdados oficiais publicados pelo SINAN/OpenDataSUS",
        "- anos recentes podem ser atualizados no portal oficial apos a execucao desta versao",
        "- parte das variaveis clinicas apresenta ausencias estruturais ou preenchimento heterogeneo entre anos",
        "- a camada oficial nao incorpora, nesta etapa, variaveis climaticas ou socioambientais exogenas",
        "",
        "## Ameacas a validade",
        "",
        "- divergencias historicas de preenchimento entre anos podem afetar comparabilidade longitudinal",
        "- subnotificacao, atraso de digitacao e inconsistencias locais podem impactar o sinal epidemiologico observado",
        "- a densificacao da Gold preserva semanas sem notificacao como zero, o que e adequado para modelagem temporal, mas exige interpretacao cuidadosa em municipios com baixa cobertura historica",
        "",
        "## Contribuicoes da trilha SINAN",
        "",
        "- congelamento de um contrato oficial `municipio-semana` para a frente epidemiologica do TCC2",
        "- separacao profissional em Bronze, Silver, Gold, governanca e camada operacional",
        "- producao de features temporais, clinicas, labels auxiliares, clusterizacao e feature selection em uma unica trilha reproduzivel",
        "- preparacao da base para integracao futura com a trilha exogena e para exposicao por API",
        "",
        "## Relacao com a modelagem preditiva",
        "",
        "A camada Gold final foi desenhada para alimentar modelos de previsao de surtos de dengue em nivel municipal, com suporte a lags, medias moveis, aceleracao, sazonalidade e indicadores clinico-epidemiologicos agregados. Essa estrutura tambem favorece a construcao de labels auxiliares para semanas criticas e uma futura camada de alertas.",
        "",
        "## Divergencias que devem ser refletidas no texto final",
        "",
        "- os artefatos nacionais legados em `data/processed/sinan` nao sao a camada oficial final, porque foram consolidados em `Brasil-semana`",
        "- a escrita do TCC2 deve substituir referencias a nacionalizacao futura por descricao da pipeline nacional efetivamente executada",
        "- a descricao da fonte epidemiologica principal deve ser alinhada ao SINAN/OpenDataSUS, e nao ao uso anterior de InfoDengue como recorte comparativo do TCC1",
        "",
        "## Pontos de qualidade a citar",
        "",
        f"- duplicatas exatas removidas na Silver: `{format_int(silver_quality['discard_summary']['duplicate_records_removed'])}`",
        f"- descartes por semana invalida: `{format_int(silver_quality['discard_summary']['invalid_week_records'])}`",
        f"- descartes por municipio invalido: `{format_int(silver_quality['discard_summary']['invalid_municipality_records'])}`",
    ]

    if top_missing_silver:
        text_lines.extend(
            [
                "",
                "Campos com maior missingness na Silver:",
                "",
            ]
        )
        for item in top_missing_silver:
            text_lines.append(f"- {item}")
    if top_missing_gold:
        text_lines.extend(
            [
                "",
                "Campos com maior missingness na Gold:",
                "",
            ]
        )
        for item in top_missing_gold:
            text_lines.append(f"- {item}")

    text_base_path = DOCS_DIR / "sinan_tcc2_texto_base.md"
    text_base_path.write_text("\n".join(text_lines), encoding="utf-8")

    latex_top_year_text = "O resumo anual consolidado deve ser lido em eda_yearly_summary.csv."
    if not top_year_row.empty:
        row = top_year_row.iloc[0]
        latex_top_year_text = (
            f"O ano com maior volume agregado de notificacoes na serie foi {int(row['ano'])}, "
            f"com {format_int(row['notificacoes_total'])} notificacoes consolidadas na camada Gold."
        )

    latex_peak_text = "As semanas de pico devem ser lidas em eda_peak_weeks.csv."
    if not peak_week_row.empty:
        row = peak_week_row.iloc[0]
        latex_peak_text = (
            f"A maior semana nacional observada foi {row['ano_semana']}, "
            f"com {format_int(row['notificacoes_total'])} notificacoes somadas entre municipios."
        )

    latex_lines = [
        "% Trecho gerado automaticamente pela trilha SINAN TCC2.",
        f"% Versao oficial: {latex_escape(args.version)}",
        "",
        "\\subsection{Trilha epidemiologica nacional do TCC2}",
        "A frente epidemiologica do TCC2 foi consolidada como uma pipeline nacional reproduzivel e auditavel em nivel municipio-semana. O processamento partiu dos microdados anuais oficiais do SINAN/Dengue publicados no OpenDataSUS, preservados na camada Bronze, seguiu para uma Silver com reconciliacao territorial, normalizacao de tipos, remocao de duplicidades exatas e agregacao por \\texttt{ibge\\_municipio + ano\\_semana}, e culminou em uma Gold densa pronta para modelagem e alertas.",
        "",
        "\\subsection{Regra temporal e consolidacao territorial}",
        "A regra temporal oficial foi congelada sobre semana epidemiologica de notificacao. A derivacao priorizou \\texttt{SEM\\_NOT}, depois \\texttt{DT\\_NOTIFIC}, e utilizou \\texttt{SEM\\_PRI} ou \\texttt{DT\\_SIN\\_PRI} apenas como fallback controlado, para evitar contaminar a serie oficial com semanas antigas, ambiguas ou semanticamente inadequadas para a contagem de notificacoes. Como o microdado nao chega sempre com o codigo municipal final do IBGE em formato consistente, a trilha executou reconciliacao territorial antes da agregacao municipio-semana.",
        "",
        "\\subsection{Execucao real da trilha}",
        (
            f"Foram inventariados {len(bronze_inventory)} arquivos anuais oficiais, cobrindo a janela operacional de 2000 a 2026. "
            f"A camada Silver aproveitou {format_int(silver_quality['record_counts']['total_notifications'])} notificacoes validas, distribuidas em "
            f"{format_int(silver_quality['record_counts']['rows'])} linhas observadas de municipio-semana e {format_int(silver_quality['record_counts']['municipalities'])} municipios unicos. "
            f"A camada Gold produziu {format_int(gold_quality['record_counts']['rows'])} linhas densas para consumo analitico e preditivo."
        ),
        "",
        "\\subsection{Resultados consolidados}",
        latex_escape(latex_top_year_text),
        latex_escape(latex_peak_text),
        "A analise exploratoria foi complementada por clusterizacao de semanas e selecao de atributos, produzindo regimes exploratorios e rankings reproduziveis para apoiar a etapa de modelagem preditiva.",
        "",
        "\\subsection{Qualidade e governanca}",
        (
            f"A trilha oficial registrou {format_int(silver_quality['discard_summary']['duplicate_records_removed'])} duplicatas exatas removidas na Silver, "
            f"{format_int(silver_quality['discard_summary']['invalid_week_records'])} descartes por semana invalida e "
            f"{format_int(silver_quality['discard_summary']['invalid_municipality_records'])} descartes por municipio invalido. "
            "A governanca final incluiu schema versionado, lineage, quality summaries, missingness, discard report e manifesto de execucao."
        ),
        "",
        "\\subsection{Limitacoes e ameacas a validade}",
        "A serie depende da qualidade e estabilidade dos microdados oficiais publicados pelo SINAN/OpenDataSUS. Anos recentes podem ser atualizados no portal oficial apos a execucao desta versao, parte das variaveis clinicas apresenta ausencias estruturais ou preenchimento heterogeneo entre anos, e a densificacao da Gold preserva semanas sem notificacao como zero, o que e adequado para modelagem temporal, mas exige interpretacao cuidadosa em municipios com baixa cobertura historica.",
        "",
        "\\subsection{Contribuicao da frente SINAN}",
        "A principal contribuicao desta etapa foi congelar uma camada epidemiologica nacional oficial em \\texttt{municipio-semana}, separada em Bronze, Silver, Gold, governanca e camada operacional, pronta para alimentar a modelagem preditiva, uma futura camada de alertas e integracao posterior com a trilha exogena.",
    ]

    latex_path = DOCS_DIR / "sinan_tcc2_latex_snippets.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    delivery_lines = [
        "# Mapa de Entrega - SINAN TCC2",
        "",
        f"- Versao oficial: `{args.version}`",
        "",
        "## Arquivos-base para metodologia",
        "",
        f"- `{DOCS_DIR / 'sinan_national_pipeline_tcc2.md'}`",
        f"- `{DOCS_DIR / 'sinan_tcc2_latex_snippets.tex'}`",
        f"- `{paths['governance'] / 'run_manifest.json'}`",
        f"- `{paths['governance'] / 'schema_silver.json'}`",
        f"- `{paths['governance'] / 'schema_gold.json'}`",
        "",
        "## Arquivos-base para resultados",
        "",
        f"- `{paths['gold'] / 'analytics' / 'eda_summary.json'}`",
        f"- `{paths['gold'] / 'analytics' / 'eda_yearly_summary.csv'}`",
        f"- `{paths['gold'] / 'analytics' / 'eda_peak_weeks.csv'}`",
        f"- `{paths['gold'] / 'analytics' / 'cluster_summary_intensity.csv'}`",
        f"- `{paths['gold'] / 'analytics' / 'cluster_summary_profile.csv'}`",
        "",
        "## Arquivos-base para defesa tecnica",
        "",
        f"- `{paths['bronze'] / 'inventory' / 'sinan_bronze_inventory.csv'}`",
        f"- `{paths['governance'] / 'quality_summary_silver.json'}`",
        f"- `{paths['governance'] / 'quality_summary_gold.json'}`",
        f"- `{paths['governance'] / 'missingness_silver.csv'}`",
        f"- `{paths['governance'] / 'missingness_gold.csv'}`",
        f"- `{paths['governance'] / 'coverage_temporal.csv'}`",
        f"- `{paths['governance'] / 'coverage_municipal.csv'}`",
        f"- `{paths['governance'] / 'coverage_summary.json'}`",
        f"- `{paths['governance'] / 'discard_report_silver.csv'}`",
        "",
        "## Camada operacional",
        "",
        f"- `{paths['serving'] / 'sql' / 'sinan_gold_schema.sql'}`",
        f"- `{paths['serving'] / 'api' / 'sinan_api.py'}`",
        f"- `{paths['serving'] / 'docs' / 'sinan_operational_publish.md'}`",
    ]
    delivery_path = DOCS_DIR / "sinan_tcc2_delivery_map.md"
    delivery_path.write_text("\n".join(delivery_lines), encoding="utf-8")

    print(methodology_path)
    print(results_path)
    print(text_base_path)
    print(latex_path)
    print(delivery_path)
    print(run_manifest.get("version"))


if __name__ == "__main__":
    main()

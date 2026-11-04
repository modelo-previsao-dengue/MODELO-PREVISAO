#!/usr/bin/env python3
"""v4-53: Gerar tabelas LaTeX e figuras finais para o TCC/Overleaf.

RF-16: Tabela comparativa A × B × C (métricas + IC 95%), gráfico de cobertura INMET,
e resumo em Markdown.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"
OVERLEAF_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB"
FIG_DIR = OVERLEAF_DIR / "figuras" / "resultados_v4"
TABLE_DIR = OVERLEAF_DIR / "tabelas"


def generate_main_table(report, out_path):
    models = report["modelos"]
    rows = []
    for var in ["A", "B", "C"]:
        for sub in ["sinan_only", "inmet_bruto", "inmet_enriquecido"]:
            key = f"{var}_{sub}"
            m = models[key]
            sub_label = {"sinan_only": "SINAN", "inmet_bruto": "INMET bruto",
                         "inmet_enriquecido": "INMET enriq."}[sub]
            ci = f"[{m['R2_log_CI95_lo']:.4f}, {m['R2_log_CI95_hi']:.4f}]"
            rows.append(f"    {var} & {sub_label} & {m['n_features']} & {m['R2_log']:.4f} & {ci} & {m['R2']:.4f} & {m['RMSE']:.2f} & {m['MAE']:.4f} \\\\")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparação de variantes e sub-modelos --- Regressão v4 mesorregional}
\label{tab:v4-regression-comparison}
\small
\begin{tabular}{llccccccc}
\toprule
Var. & Sub-modelo & \# feat & $R^2_{\log}$ & IC 95\% & $R^2$ & RMSE & MAE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def generate_ttest_table(report, out_path):
    ttests = report["testes_t"]
    boot = report["bootstrap_diffs"]

    rows = []
    for name in ["A_vs_B", "A_vs_C", "B_vs_C"]:
        t = ttests[name]
        b = boot[name]
        p = t["p_value"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ci_str = f"[{b['CI95_lo']:.4f}, {b['CI95_hi']:.4f}]"
        name_fmt = name.replace("_", " ")
        rows.append(f"    {name_fmt} & {t['t_stat']:.2f} & {p:.2e} & {sig} & {b['mean_diff']:+.4f} & {ci_str} \\\\")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Testes estatísticos --- comparação entre variantes (INMET enriquecido)}
\label{tab:v4-statistical-tests}
\begin{tabular}{lccccc}
\toprule
Comparação & $t$ & $p$ & Sig. & $\Delta R^2_{\log}$ & IC 95\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def generate_walkforward_table(wf_report, out_path):
    rows = []
    for var in ["A", "B", "C"]:
        for sub in ["sinan_only", "inmet_bruto", "inmet_enriquecido"]:
            key = f"{var}_{sub}"
            m = wf_report["modelos"][key]
            sub_label = {"sinan_only": "SINAN", "inmet_bruto": "INMET bruto",
                         "inmet_enriquecido": "INMET enriq."}[sub]
            fold_r2s = " & ".join(f"{f['R2_log']:.4f}" for f in m["folds"])
            rows.append(f"    {var} & {sub_label} & {fold_r2s} & {m['mean_R2_log']:.4f} \\\\")

    latex = r"""\begin{table}[htbp]
\centering
\caption{Walk-forward v4 --- $R^2_{\log}$ por fold e modelo}
\label{tab:v4-walkforward}
\small
\begin{tabular}{llccccc}
\toprule
Var. & Sub-modelo & F1 (2023) & F2 (2024) & F3 (2025) & F4 (2026) & Média \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def plot_inmet_coverage(integration_report, out_path):
    cov = integration_report["cobertura_por_ano"]
    years = sorted(int(y) for y in cov.keys())
    pcts = [cov[str(y)] for y in years]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#4CAF50" if p >= 80 else "#FF9800" if p >= 40 else "#E53935" for p in pcts]
    ax.bar(years, pcts, color=colors, width=0.8)
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80%")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Cobertura INMET (%)")
    ax.set_title("Cobertura INMET por Ano — Dataset Integrado v4", fontsize=12)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def plot_delta_mesorregiao(meso_csv_path, out_path):
    meso = pd.read_csv(meso_csv_path)
    a = meso[meso["modelo"].str.contains("Var A") & meso["modelo"].str.contains("enriquecido")]
    b = meso[meso["modelo"].str.contains("Var B") & meso["modelo"].str.contains("enriquecido")]
    if a.empty or b.empty:
        print("  AVISO: sem dados para delta mesorregião")
        return

    a = a.set_index("cod_mesorregiao")["R2_log"]
    b = b.set_index("cod_mesorregiao")["R2_log"]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return

    deltas = [(cod, b.loc[cod] - a.loc[cod]) for cod in common]
    deltas.sort(key=lambda x: x[1])

    n_show = 20
    if len(deltas) > 2 * n_show:
        deltas_show = deltas[:n_show] + deltas[-n_show:]
    else:
        deltas_show = deltas

    cods = [str(d[0]) for d in deltas_show]
    vals = [d[1] for d in deltas_show]
    colors = ["#4CAF50" if v > 0 else "#E53935" for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(8, len(cods) * 0.25)))
    ax.barh(range(len(cods)), vals, color=colors)
    ax.set_yticks(range(len(cods)))
    ax.set_yticklabels(cods, fontsize=7)
    ax.set_xlabel("ΔR²_log (Var B − Var A)", fontsize=10)
    ax.set_title(f"Impacto da Série Completa por Mesorregião\n(top/bottom {n_show} de {len(deltas)})", fontsize=12)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def generate_year_table(reg_report, out_path):
    year_mets = reg_report["metricas_por_ano"]
    enriq = [m for m in year_mets if "enriquecido" in m["modelo"]]

    rows = []
    for year in [2024, 2025, 2026]:
        cells = []
        for var in ["A", "B", "C"]:
            label = f"Var {var}: INMET enriquecido"
            m = next((x for x in enriq if x["ano"] == year and x["modelo"] == label), None)
            if m:
                cells.append(f"{m['R2_log']:.4f}")
            else:
                cells.append("---")
        rows.append(f"    {year} & {' & '.join(cells)} \\\\")

    latex = r"""\begin{table}[htbp]
\centering
\caption{$R^2_{\log}$ por ano de teste --- INMET enriquecido}
\label{tab:v4-r2-by-year}
\begin{tabular}{lccc}
\toprule
Ano & Var A & Var B & Var C \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def generate_flag_table(reg_report, out_path):
    flag_mets = reg_report["metricas_por_flag_inmet"]
    enriq = [m for m in flag_mets if "enriquecido" in m["modelo"]]

    rows = []
    for var in ["A", "B", "C"]:
        label = f"Var {var}: INMET enriquecido"
        for flag in [0, 1]:
            m = next((x for x in enriq if x["modelo"] == label and x["flag_fonte_inmet"] == flag), None)
            if m:
                flag_label = "Sem INMET" if flag == 0 else "Com INMET"
                rows.append(f"    {var} & {flag_label} & {m['R2_log']:.4f} & {m['RMSE']:.2f} & {m['n']:,} \\\\")

    latex = r"""\begin{table}[htbp]
\centering
\caption{$R^2_{\log}$ por cobertura INMET --- INMET enriquecido}
\label{tab:v4-r2-by-flag}
\begin{tabular}{llccc}
\toprule
Var. & Cobertura & $R^2_{\log}$ & RMSE & $n$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def generate_classification_table(cls_report, out_path):
    models = cls_report.get("modelos", {})
    rows = []
    for var in ["A", "B", "C"]:
        for sub in ["sinan_only", "inmet_bruto", "inmet_enriquecido"]:
            key = f"{var}_{sub}"
            if key not in models:
                continue
            m = models[key]
            sub_label = {"sinan_only": "SINAN", "inmet_bruto": "INMET bruto",
                         "inmet_enriquecido": "INMET enriq."}[sub]
            f1_surto = m.get("f1_per_class", {}).get("surto", 0)
            rows.append(
                f"    {var} & {sub_label} & {m['f1_macro']:.4f} & {m['auc_macro_ovr']:.4f} "
                f"& {m['accuracy']:.4f} & {f1_surto:.4f} \\\\"
            )

    latex = r"""\begin{table}[htbp]
\centering
\caption{Classificação de risco v4 --- 3 variantes $\times$ 3 sub-modelos}
\label{tab:v4-classification}
\small
\begin{tabular}{llcccc}
\toprule
Var. & Sub-modelo & F1\_macro & AUC\_macro & Acc & F1\_surto \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    out_path.write_text(latex)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v4-53: Gerar Outputs para TCC/Overleaf")
    print("=" * 70)

    with open(DATA_DIR / "46_regression_v4_report.json") as f:
        reg = json.load(f)

    with open(DATA_DIR / "47_walkforward_v4_report.json") as f:
        wf = json.load(f)

    with open(DATA_DIR / "42_integration_report.json") as f:
        integ = json.load(f)

    cls_path = DATA_DIR / "54_classification_v4_report.json"
    cls = None
    if cls_path.exists():
        with open(cls_path) as f:
            cls = json.load(f)

    print("\nGerando tabelas LaTeX...")
    generate_main_table(reg, TABLE_DIR / "tab_v4_regression_comparison.tex")
    generate_ttest_table(reg, TABLE_DIR / "tab_v4_statistical_tests.tex")
    generate_walkforward_table(wf, TABLE_DIR / "tab_v4_walkforward.tex")
    generate_year_table(reg, TABLE_DIR / "tab_v4_r2_by_year.tex")
    generate_flag_table(reg, TABLE_DIR / "tab_v4_r2_by_flag.tex")

    if cls:
        generate_classification_table(cls, TABLE_DIR / "tab_v4_classification.tex")

    print("\nGerando figuras...")
    plot_inmet_coverage(integ, FIG_DIR / "fig_v4_inmet_coverage.png")
    plot_delta_mesorregiao(MODEL_DIR / "metrics_por_mesorregiao_v4.csv",
                           FIG_DIR / "fig_v4_delta_mesorregiao_B_vs_A.png")

    summary_path = DATA_DIR / "53_tcc_outputs_summary.md"
    enriq_a = reg["modelos"]["A_inmet_enriquecido"]
    enriq_b = reg["modelos"]["B_inmet_enriquecido"]
    enriq_c = reg["modelos"]["C_inmet_enriquecido"]

    summary = f"""# Resumo dos Resultados v4

## Resultados Principais (INMET enriquecido, 266 features)

| Variante | R²_log | IC 95% | RMSE | MAE |
|----------|--------|--------|------|-----|
| A (baseline 4yr) | {enriq_a['R2_log']:.4f} | [{enriq_a['R2_log_CI95_lo']:.4f}, {enriq_a['R2_log_CI95_hi']:.4f}] | {enriq_a['RMSE']:.2f} | {enriq_a['MAE']:.4f} |
| B (série 22yr) | {enriq_b['R2_log']:.4f} | [{enriq_b['R2_log_CI95_lo']:.4f}, {enriq_b['R2_log_CI95_hi']:.4f}] | {enriq_b['RMSE']:.2f} | {enriq_b['MAE']:.4f} |
| C (série + peso) | {enriq_c['R2_log']:.4f} | [{enriq_c['R2_log_CI95_lo']:.4f}, {enriq_c['R2_log_CI95_hi']:.4f}] | {enriq_c['RMSE']:.2f} | {enriq_c['MAE']:.4f} |

## Diferenças Estatísticas

| Comparação | ΔR²_log | IC 95% da diferença | p-value | Significância |
|------------|---------|---------------------|---------|---------------|
| B vs A | {reg['deltas_enriquecido']['B_vs_A']:+.4f} | [{reg['bootstrap_diffs']['A_vs_B']['CI95_lo']:.4f}, {reg['bootstrap_diffs']['A_vs_B']['CI95_hi']:.4f}] | {reg['testes_t']['A_vs_B']['p_value']:.2e} | *** |
| C vs A | {reg['deltas_enriquecido']['C_vs_A']:+.4f} | [{reg['bootstrap_diffs']['A_vs_C']['CI95_lo']:.4f}, {reg['bootstrap_diffs']['A_vs_C']['CI95_hi']:.4f}] | {reg['testes_t']['A_vs_C']['p_value']:.2e} | *** |
| C vs B | {reg['deltas_enriquecido']['C_vs_B']:+.4f} | [{reg['bootstrap_diffs']['B_vs_C']['CI95_lo']:.4f}, {reg['bootstrap_diffs']['B_vs_C']['CI95_hi']:.4f}] | {reg['testes_t']['B_vs_C']['p_value']:.2e} | ns |

## Walk-Forward (INMET enriquecido)

| Variante | R²_log médio |
|----------|-------------|
| A | {wf['resumo_enriquecido']['mean_R2_log_A']:.4f} |
| B | {wf['resumo_enriquecido']['mean_R2_log_B']:.4f} |
| C | {wf['resumo_enriquecido']['mean_R2_log_C']:.4f} |

## Conclusões

1. **A série temporal completa (22 anos) melhora significativamente** a predição de dengue (ΔR²_log = {reg['deltas_enriquecido']['B_vs_A']:+.4f}, p < 0.001).
2. **O decaimento exponencial (variante C) não melhora** em relação a B (ΔR²_log = {reg['deltas_enriquecido']['C_vs_B']:+.4f}, ns).
3. **Melhor variante: {reg['conclusao']['melhor_variante']}** (série completa sem peso).
4. O walk-forward confirma a superioridade de B sobre A (Δ = {wf['resumo_enriquecido']['delta_B_vs_A']:+.4f}).

## Limitações

1. INMET Silver ausente para 2009, 2020 e 2022 reduz a cobertura climática.
2. A variante A tem cobertura INMET de ~48% (vs ~90% original) devido à falta de 2020/2022.
3. 2026 é ano parcial (17 semanas no momento do corte).
"""
    summary_path.write_text(summary)
    print(f"  Salvo: {summary_path.name}")

    print(f"\n{'=' * 70}")
    print(f"  OUTPUTS GERADOS:")
    print(f"  Tabelas LaTeX: 3 arquivos em {TABLE_DIR.name}/")
    print(f"  Figuras: fig_v4_inmet_coverage.png")
    print(f"  Resumo: 53_tcc_outputs_summary.md")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

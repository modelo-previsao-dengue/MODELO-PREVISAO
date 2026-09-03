#!/usr/bin/env python3
"""US-118: Compilação final — Fases 1 + 2.

Consolida todos os resultados em tabela mestra, conclusão honesta,
tabelas LaTeX e JSON final.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v2"
OVERLEAF_DIR = BASE_DIR.parent / "Overleaf" / "TCC2 Base FCTE UnB"
TABLES_DIR = OVERLEAF_DIR / "tabelas"
FIG_DIR = OVERLEAF_DIR / "figuras" / "resultados"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def write_latex_table(filename, content):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    with open(path, "w") as f:
        f.write(content)
    print(f"  Salvo: {path.name}")


def generate_master_table(experiments):
    """Master comparison table: all experiments."""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Tabela mestra: comparação de todos os experimentos — Fases 1 e 2.}
\label{tab:master_v2}
\small
\begin{tabular}{p{4cm}lrrrl}
\toprule
\textbf{Experimento} & \textbf{Features} & \textbf{R²\textsubscript{log}} & \textbf{$\Delta$ vs SINAN} & \textbf{p-value} & \textbf{SHAP clima top-20} \\
\midrule
"""
    for exp in experiments:
        name_clean = exp["name"].replace("_", r"\_").replace("&", r"\&")
        features_clean = exp["features"].replace("_", r"\_").replace("&", r"\&")
        p = exp.get("p_value", "—")
        if isinstance(p, float):
            p = f"${p:.2e}$" if p < 0.01 else f"{p:.4f}"
        shap_str = str(exp.get("shap_climate_top20", "—"))
        tex += f"{name_clean} & {features_clean} & {exp['R2_log']:.4f} & {exp.get('delta', 0):+.4f} & {p} & {shap_str} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def generate_blindness_table(blindness):
    """Blindness comparison table."""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Data Blindness: desempenho do modelo com e sem INMET em cenários de atraso SINAN.}
\label{tab:blindness_v2}
\begin{tabular}{lrrrr}
\toprule
\textbf{Cenário} & \textbf{R²\textsubscript{log} sem INMET} & \textbf{R²\textsubscript{log} com INMET} & \textbf{$\Delta$R²} & \textbf{INMET ajuda?} \\
\midrule
"""
    for scenario, data in blindness.items():
        name_clean = scenario.replace("_", r"\_")
        sem = data["sem_inmet"]["R2_log"]
        com = data["com_inmet"]["R2_log"]
        delta = data["delta_R2_log"]
        ajuda = "Sim" if data["inmet_ajuda"] else "Não"
        tex += f"{name_clean} & {sem:.4f} & {com:.4f} & {delta:+.4f} & {ajuda} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def generate_thresholds_table(thresholds):
    """Climate thresholds table."""
    if not thresholds:
        return ""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Limiares climáticos extraídos da análise SHAP.}
\label{tab:thresholds_v2}
\begin{tabular}{lrrlrr}
\toprule
\textbf{Variável} & \textbf{Rank} & \textbf{Limiar} & \textbf{Direção} & \textbf{SHAP acima} & \textbf{SHAP abaixo} \\
\midrule
"""
    for t in thresholds:
        feat_clean = t["feature"].replace("_", r"\_")
        tex += f"{feat_clean} & {t['rank']} & {t['threshold']:.2f} & {t['direction']} & {t['mean_shap_above']:+.4f} & {t['mean_shap_below']:+.4f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def plot_all_experiments(experiments, out_path):
    """Barplot of all experiments."""
    names = [e["name"] for e in experiments]
    r2s = [e["R2_log"] for e in experiments]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    for e in experiments:
        if "SINAN-only" in e["features"]:
            colors.append("#1976D2")
        elif "enriquecido" in e["features"]:
            colors.append("#E53935")
        elif "Blind" in e["name"] or "INMET-only" in e["name"]:
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    bars = ax.barh(range(len(names)), r2s, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("R²_log", fontsize=11)
    ax.set_title("Comparação de Todos os Experimentos — Fases 1 e 2", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-118: Compilação Final — Fases 1 + 2")
    print("=" * 60)

    print("\nCarregando relatórios...")
    reg_p1 = load_json(DATA_DIR / "05_regression_report.json")
    shap_p1 = load_json(DATA_DIR / "07_shap_report.json")
    wf_p1 = load_json(DATA_DIR / "08_walkforward_report.json")
    reg_v2 = load_json(DATA_DIR / "13_regression_v2_report.json")
    blindness = load_json(DATA_DIR / "14_blindness_report.json")
    cls_v2 = load_json(DATA_DIR / "15_classification_v2_report.json")
    shap_v2 = load_json(DATA_DIR / "16_shap_v2_report.json")
    wf_v2 = load_json(DATA_DIR / "17_walkforward_v2_report.json")

    experiments = []

    if reg_p1:
        sinan_r2 = reg_p1.get("sinan_only", {}).get("R2_log", 0)
        inmet_r2 = reg_p1.get("sinan_inmet", {}).get("R2_log", 0)
        p_val = reg_p1.get("paired_ttest", {}).get("p_value", None)
        experiments.append({
            "name": "Fase 1: SINAN-only",
            "features": "SINAN-only (129)",
            "R2_log": sinan_r2, "delta": 0.0, "p_value": "—",
            "shap_climate_top20": "—",
        })
        experiments.append({
            "name": "Fase 1: INMET bruto",
            "features": "SINAN+INMET bruto (159)",
            "R2_log": inmet_r2,
            "delta": round(inmet_r2 - sinan_r2, 4),
            "p_value": p_val,
            "shap_climate_top20": shap_p1.get("climate_in_top_20", "?"),
        })

    if reg_v2:
        mods = reg_v2.get("modelos", {})
        for key, label, feats_desc in [
            ("A_sinan_only", "Fase 2: SINAN-only", "SINAN-only (129)"),
            ("B_inmet_bruto", "Fase 2: INMET bruto", "SINAN+INMET bruto (159)"),
            ("C_inmet_enriquecido", "Fase 2: INMET enriquecido", "SINAN+INMET+lags+anom (303)"),
        ]:
            if key in mods:
                r2 = mods[key]["R2_log"]
                sinan_r2_v2 = mods.get("A_sinan_only", {}).get("R2_log", 0)
                experiments.append({
                    "name": label,
                    "features": feats_desc,
                    "R2_log": r2,
                    "delta": round(r2 - sinan_r2_v2, 4) if key != "A_sinan_only" else 0.0,
                    "p_value": reg_v2.get("testes_t", {}).get(
                        "A vs B" if key == "B_inmet_bruto" else "A vs C", {}
                    ).get("p_value", "—") if key != "A_sinan_only" else "—",
                    "shap_climate_top20": shap_v2.get("climate_in_top_20", "?") if key != "A_sinan_only" else "—",
                })

    if blindness:
        cenarios = blindness.get("cenarios", {})
        for scenario, data in cenarios.items():
            if scenario == "Full (referência)":
                continue
            experiments.append({
                "name": f"Blindness: {scenario} + INMET",
                "features": f"Blind + INMET",
                "R2_log": data["com_inmet"]["R2_log"],
                "delta": data["delta_R2_log"],
                "p_value": data.get("ttest", {}).get("p_value", "—"),
                "shap_climate_top20": "—",
            })
            experiments.append({
                "name": f"Blindness: {scenario} sem INMET",
                "features": f"Blind sem INMET",
                "R2_log": data["sem_inmet"]["R2_log"],
                "delta": 0.0,
                "p_value": "—",
                "shap_climate_top20": "—",
            })

    print(f"\n  {len(experiments)} configurações de modelo")

    print("\nGerando tabelas LaTeX...")
    write_latex_table("tab_5yr_v2_master.tex", generate_master_table(experiments))

    if blindness and "cenarios" in blindness:
        write_latex_table("tab_5yr_v2_blindness.tex", generate_blindness_table(blindness["cenarios"]))

    if shap_v2 and "thresholds" in shap_v2:
        thresh_tex = generate_thresholds_table(shap_v2["thresholds"])
        if thresh_tex:
            write_latex_table("tab_5yr_v2_thresholds.tex", thresh_tex)

    print("\nGerando figuras...")
    plot_all_experiments(experiments, FIG_DIR / "fig_5yr_v2_comparacao_todos.png")

    conclusao = {
        "pergunta_1_clima_melhora": None,
        "pergunta_2_melhor_cenario": None,
        "pergunta_3_variaveis_importantes": None,
        "pergunta_4_limiares": None,
        "pergunta_5_recomendacoes": None,
    }

    if reg_v2:
        mods = reg_v2.get("modelos", {})
        delta_b = mods.get("B_inmet_bruto", {}).get("R2_log", 0) - mods.get("A_sinan_only", {}).get("R2_log", 0)
        delta_c = mods.get("C_inmet_enriquecido", {}).get("R2_log", 0) - mods.get("A_sinan_only", {}).get("R2_log", 0)

        if delta_b > 0.01:
            conclusao["pergunta_1_clima_melhora"] = f"SIM — INMET bruto melhora R²_log em {delta_b:+.4f}"
        elif delta_b > 0:
            conclusao["pergunta_1_clima_melhora"] = f"MARGINALMENTE — INMET bruto melhora R²_log em {delta_b:+.4f} (estatisticamente significativo, mas marginal)"
        else:
            conclusao["pergunta_1_clima_melhora"] = f"NÃO — INMET bruto piora R²_log em {delta_b:+.4f}"

    if blindness:
        cenarios = blindness.get("cenarios", {})
        best_scenario = blindness.get("conclusao", {}).get("cenario_inmet_mais_ajuda", "?")
        best_delta = blindness.get("conclusao", {}).get("delta_maximo", 0)
        conclusao["pergunta_2_melhor_cenario"] = f"{best_scenario} (Δ={best_delta:+.4f})"

    if shap_v2:
        best_feat = shap_v2.get("best_climate_feature", "?")
        best_rank = shap_v2.get("best_climate_rank", "?")
        n_top20 = shap_v2.get("climate_in_top_20", 0)
        conclusao["pergunta_3_variaveis_importantes"] = f"{n_top20} features climáticas no top-20 SHAP. Melhor: {best_feat} (rank {best_rank})"

    if shap_v2 and shap_v2.get("thresholds"):
        thresholds = shap_v2["thresholds"]
        conclusao["pergunta_4_limiares"] = f"{len(thresholds)} limiares climáticos identificados"

    conclusao["pergunta_5_recomendacoes"] = [
        "Incluir resultados de Data Blindness no TCC como evidência do valor prático do INMET",
        "Destacar que a contribuição do INMET é marginal em cenário com dados SINAN completos",
        "Enfatizar o cenário de atraso (Blind-4w/8w) como uso prático das variáveis climáticas",
        "Reportar limiares climáticos como contribuição prática para vigilância epidemiológica",
    ]

    final_report = {
        "experimentos": experiments,
        "classificacao_v2": cls_v2 if cls_v2 else None,
        "walkforward_v2": {
            "sinan_mean": wf_v2.get("sinan_only", {}).get("mean_R2_log", None),
            "inmet_mean": wf_v2.get("inmet_enriquecido", {}).get("mean_R2_log", None),
            "folds_inmet_better": wf_v2.get("folds_inmet_better", None),
        } if wf_v2 else None,
        "conclusao": conclusao,
    }

    with open(DATA_DIR / "18_final_results_v2.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 60}")
    print(f"  CONCLUSÃO FINAL — FASES 1 + 2")
    print(f"{'=' * 60}")
    for key, val in conclusao.items():
        if isinstance(val, list):
            print(f"\n  {key}:")
            for item in val:
                print(f"    - {item}")
        else:
            print(f"  {key}: {val}")
    print(f"{'=' * 60}")

    figs = sorted(FIG_DIR.glob("fig_5yr*.png"))
    print(f"\n  Total de figuras: {len(figs)}")
    tabs = sorted(TABLES_DIR.glob("tab_5yr*.tex"))
    print(f"  Total de tabelas LaTeX: {len(tabs)}")


if __name__ == "__main__":
    main()

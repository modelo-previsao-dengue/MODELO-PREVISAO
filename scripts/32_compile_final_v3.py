#!/usr/bin/env python3
"""US-212: Compilação final v3 (mesorregional) — consolida todos os resultados.

Carrega todos os relatórios `2N_*_report.json` de `data/model_ready_v3/`,
compara com os resultados finais da v2 (`18_final_results_v2.json`), produz
`32_final_results_v3.json`, uma figura comparativa v2 vs v3 e tabelas LaTeX
para uso direto no Overleaf.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
DATA_DIR_V2 = BASE_DIR / "data" / "model_ready_v2"
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


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def generate_master_table_v3(experiments):
    """Master comparison table: regression + blindness v3 experiments."""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Tabela mestra v3 (mesorregional): comparação de todos os experimentos de regressão.}
\label{tab:master_v3}
\small
\begin{tabular}{p{5cm}lrrl}
\toprule
\textbf{Experimento} & \textbf{Features} & \textbf{R²\textsubscript{log}} & \textbf{$\Delta$ vs A} & \textbf{p-value} \\
\midrule
"""
    for exp in experiments:
        p = exp.get("p_value", "—")
        if isinstance(p, float):
            p = f"${p:.2e}$" if p < 0.01 else f"{p:.4f}"
        tex += f"{esc(exp['name'])} & {esc(exp['features'])} & {exp['R2_log']:.4f} & {exp.get('delta', 0):+.4f} & {p} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def generate_classification_table_v3(cls_report):
    """Classification metrics table (F1_macro, AUC, per-class F1)."""
    modelos = cls_report.get("modelos", {})
    if not modelos:
        return ""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Classificação v3 (mesorregional): F1, AUC e desempenho por classe de risco.}
\label{tab:classification_v3}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Modelo} & \textbf{F1\textsubscript{macro}} & \textbf{AUC\textsubscript{macro}} & \textbf{F1\textsubscript{baixo}} & \textbf{F1\textsubscript{moderado}} & \textbf{F1\textsubscript{alto}} & \textbf{F1\textsubscript{muito\_alto}} \\
\midrule
"""
    for key, m in modelos.items():
        f1c = m.get("f1_per_class", {})
        tex += (
            f"{esc(m['label'])} & {m['f1_macro']:.4f} & {m['auc_macro_ovr']:.4f} & "
            f"{f1c.get('baixo', 0):.4f} & {f1c.get('moderado', 0):.4f} & "
            f"{f1c.get('alto', 0):.4f} & {f1c.get('muito_alto', 0):.4f} \\\\\n"
        )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def generate_walkforward_table_v3(wf_report):
    """Walk-forward per-fold R²_log table."""
    modelos = wf_report.get("modelos", {})
    if not modelos:
        return ""
    a_folds = modelos.get("A_sinan_only", {}).get("folds", [])
    b_folds = modelos.get("B_inmet_bruto", {}).get("folds", [])
    c_folds = modelos.get("C_inmet_enriquecido", {}).get("folds", [])
    n = len(a_folds)
    tex = r"""\begin{table}[htbp]
\centering
\caption{Validação walk-forward v3 (mesorregional): R²\textsubscript{log} por fold.}
\label{tab:walkforward_v3}
\begin{tabular}{lrrrr}
\toprule
\textbf{Fold} & \textbf{Teste} & \textbf{A: SINAN} & \textbf{B: INMET bruto} & \textbf{C: INMET enriq.} \\
\midrule
"""
    for i in range(n):
        test_years = "/".join(str(y) for y in a_folds[i]["test_years"])
        r2_a = a_folds[i]["R2_log"]
        r2_b = b_folds[i]["R2_log"] if i < len(b_folds) else float("nan")
        r2_c = c_folds[i]["R2_log"] if i < len(c_folds) else float("nan")
        tex += f"{i + 1} & {test_years} & {r2_a:.4f} & {r2_b:.4f} & {r2_c:.4f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def generate_thresholds_table_v3(thresholds):
    """Climate thresholds table."""
    if not thresholds:
        return ""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Limiares climáticos extraídos da análise SHAP (v3, mesorregional).}
\label{tab:thresholds_v3}
\begin{tabular}{lrrlrr}
\toprule
\textbf{Variável} & \textbf{Rank} & \textbf{Limiar} & \textbf{Direção} & \textbf{SHAP acima} & \textbf{SHAP abaixo} \\
\midrule
"""
    for t in thresholds:
        tex += f"{esc(t['feature'])} & {t['rank']} & {t['threshold']:.2f} & {t['direction']} & {t['mean_shap_above']:+.4f} & {t['mean_shap_below']:+.4f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


def plot_v2_vs_v3(reg_v3, final_v2, out_path):
    """Grouped barplot: R²_log per model type, v2 (Fase 2) vs v3 (mesorregional)."""
    mods_v3 = reg_v3.get("modelos", {})
    keys = [
        ("A_sinan_only", "SINAN-only"),
        ("B_inmet_bruto", "INMET bruto"),
        ("C_inmet_enriquecido", "INMET enriquecido"),
    ]

    v2_by_name = {e["name"]: e["R2_log"] for e in final_v2.get("experimentos", [])}
    v2_labels = {
        "A_sinan_only": "Fase 2: SINAN-only",
        "B_inmet_bruto": "Fase 2: INMET bruto",
        "C_inmet_enriquecido": "Fase 2: INMET enriquecido",
    }

    labels = [name for _, name in keys]
    v3_vals = [mods_v3.get(k, {}).get("R2_log") for k, _ in keys]
    v2_vals = [v2_by_name.get(v2_labels[k]) for k, _ in keys]

    valid = [i for i in range(len(keys)) if v3_vals[i] is not None and v2_vals[i] is not None]
    if not valid:
        print("  AVISO: sem dados suficientes para comparação v2 vs v3, figura não gerada.")
        return

    x = np.arange(len(valid))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, [v2_vals[i] for i in valid], width, label="v2 (municipal)", color="#1976D2")
    ax.bar(x + width / 2, [v3_vals[i] for i in valid], width, label="v3 (mesorregional)", color="#E53935")

    for i, xi in enumerate(x):
        ax.text(xi - width / 2, v2_vals[valid[i]], f"{v2_vals[valid[i]]:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, v3_vals[valid[i]], f"{v3_vals[valid[i]]:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in valid], fontsize=10)
    ax.set_ylabel("R² (log)", fontsize=11)
    ax.set_title("Comparação v2 (municipal) vs v3 (mesorregional)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-212: Compilação Final v3 (Mesorregional)")
    print("=" * 60)

    print("\nCarregando relatórios v3...")
    reports_v3 = {}
    for path in sorted(DATA_DIR.glob("*_report.json")):
        key = path.stem
        reports_v3[key] = load_json(path)
        print(f"  {path.name}")

    mapping_r = reports_v3.get("20_mapping_report", {})
    sinan_meso_r = reports_v3.get("21_sinan_meso_report", {})
    inmet_meso_r = reports_v3.get("22_inmet_meso_report", {})
    integration_r = reports_v3.get("23_integration_report", {})
    lags_r = reports_v3.get("24_lags_report", {})
    anomalias_r = reports_v3.get("25_anomalias_report", {})
    splits_r = reports_v3.get("26_splits_v3_report", {})
    reg_v3 = reports_v3.get("27_regression_v3_report", {})
    blindness_v3 = reports_v3.get("28_blindness_v3_report", {})
    cls_v3 = reports_v3.get("29_classification_v3_report", {})
    shap_v3 = reports_v3.get("30_shap_v3_report", {})
    wf_v3 = reports_v3.get("31_walkforward_v3_report", {})

    print("\nCarregando resultados finais v2...")
    final_v2 = load_json(DATA_DIR_V2 / "18_final_results_v2.json")

    experiments = []
    if reg_v3:
        mods = reg_v3.get("modelos", {})
        a_r2 = mods.get("A_sinan_only", {}).get("R2_log", 0)
        for key, label, feats_desc in [
            ("A_sinan_only", "v3: SINAN-only", "SINAN-only (110)"),
            ("B_inmet_bruto", "v3: INMET bruto", "SINAN+INMET bruto (122)"),
            ("C_inmet_enriquecido", "v3: INMET enriquecido", "SINAN+INMET+lags+anom (266)"),
        ]:
            if key in mods:
                r2 = mods[key]["R2_log"]
                p_key = "A vs B" if key == "B_inmet_bruto" else "A vs C" if key == "C_inmet_enriquecido" else None
                experiments.append({
                    "name": label,
                    "features": feats_desc,
                    "R2_log": r2,
                    "delta": round(r2 - a_r2, 4) if key != "A_sinan_only" else 0.0,
                    "p_value": reg_v3.get("testes_t", {}).get(p_key, {}).get("p_value", "—") if p_key else "—",
                })

    if blindness_v3:
        for scenario, data in blindness_v3.get("cenarios", {}).items():
            if scenario == "Full (referência)":
                continue
            experiments.append({
                "name": f"v3 Blindness: {scenario} + INMET",
                "features": "Blind + INMET",
                "R2_log": data["com_inmet"]["R2_log"],
                "delta": data["delta_R2_log"],
                "p_value": data.get("ttest", {}).get("p_value", "—"),
            })
            experiments.append({
                "name": f"v3 Blindness: {scenario} sem INMET",
                "features": "Blind sem INMET",
                "R2_log": data["sem_inmet"]["R2_log"],
                "delta": 0.0,
                "p_value": "—",
            })

    print(f"\n  {len(experiments)} configurações de modelo (v3)")

    print("\nGerando tabelas LaTeX...")
    write_latex_table("tab_v3_master.tex", generate_master_table_v3(experiments))
    if cls_v3:
        tex = generate_classification_table_v3(cls_v3)
        if tex:
            write_latex_table("tab_v3_classification.tex", tex)
    if wf_v3:
        tex = generate_walkforward_table_v3(wf_v3)
        if tex:
            write_latex_table("tab_v3_walkforward.tex", tex)
    if shap_v3 and shap_v3.get("thresholds"):
        tex = generate_thresholds_table_v3(shap_v3["thresholds"])
        if tex:
            write_latex_table("tab_v3_thresholds.tex", tex)

    print("\nGerando figura de comparação v2 vs v3...")
    plot_v2_vs_v3(reg_v3, final_v2, FIG_DIR / "fig_v3_comparison_v2_v3.png")

    conclusao = {
        "pergunta_1_clima_melhora": None,
        "pergunta_2_melhor_cenario_blindness": None,
        "pergunta_3_variaveis_importantes": None,
        "pergunta_4_limiares": None,
        "pergunta_5_classificacao": None,
        "pergunta_6_estabilidade_temporal": None,
        "pergunta_7_v2_vs_v3": None,
        "pergunta_8_recomendacoes": None,
    }

    if reg_v3:
        mods = reg_v3.get("modelos", {})
        delta_b = mods.get("B_inmet_bruto", {}).get("R2_log", 0) - mods.get("A_sinan_only", {}).get("R2_log", 0)
        delta_c = mods.get("C_inmet_enriquecido", {}).get("R2_log", 0) - mods.get("A_sinan_only", {}).get("R2_log", 0)
        if delta_b > 0.01:
            conclusao["pergunta_1_clima_melhora"] = f"SIM — INMET bruto melhora R²_log em {delta_b:+.4f} (mesorregional)"
        elif delta_b > 0:
            conclusao["pergunta_1_clima_melhora"] = f"MARGINALMENTE — INMET bruto melhora R²_log em {delta_b:+.4f}"
        else:
            conclusao["pergunta_1_clima_melhora"] = f"NÃO — INMET bruto piora R²_log em {delta_b:+.4f}"

    if blindness_v3:
        best_scenario = blindness_v3.get("conclusao", {}).get("cenario_inmet_mais_ajuda", "?")
        best_delta = blindness_v3.get("conclusao", {}).get("delta_maximo", 0)
        conclusao["pergunta_2_melhor_cenario_blindness"] = f"{best_scenario} (Δ={best_delta:+.4f})"

    if shap_v3:
        best_feat = shap_v3.get("best_climate_feature", "?")
        best_rank = shap_v3.get("best_climate_rank", "?")
        n_top20 = shap_v3.get("climate_in_top_20", 0)
        conclusao["pergunta_3_variaveis_importantes"] = f"{n_top20} features climáticas no top-20 SHAP. Melhor: {best_feat} (rank {best_rank})"
        if shap_v3.get("thresholds"):
            conclusao["pergunta_4_limiares"] = f"{len(shap_v3['thresholds'])} limiares climáticos identificados"

    if cls_v3:
        conclusao["pergunta_5_classificacao"] = (
            f"Melhor modelo de classificação: {cls_v3.get('melhor_modelo', '?')} "
            f"(F1_macro={cls_v3.get('modelos', {}).get('C_inmet_enriquecido', {}).get('f1_macro', '?')})"
        )

    if wf_v3:
        mean_a = wf_v3.get("modelos", {}).get("A_sinan_only", {}).get("mean_R2_log")
        mean_c = wf_v3.get("modelos", {}).get("C_inmet_enriquecido", {}).get("mean_R2_log")
        folds_c_better = wf_v3.get("folds_c_better_than_a")
        n_folds = wf_v3.get("n_folds")
        conclusao["pergunta_6_estabilidade_temporal"] = (
            f"R²_log médio walk-forward: A={mean_a}, C={mean_c}. "
            f"C supera A em {folds_c_better}/{n_folds} folds."
        )

    if reg_v3 and final_v2:
        v2_exps = {e["name"]: e["R2_log"] for e in final_v2.get("experimentos", [])}
        v3_c = reg_v3.get("modelos", {}).get("C_inmet_enriquecido", {}).get("R2_log")
        v2_c = v2_exps.get("Fase 2: INMET enriquecido")
        if v3_c is not None and v2_c is not None:
            delta_v2v3 = v3_c - v2_c
            conclusao["pergunta_7_v2_vs_v3"] = (
                f"R²_log modelo C: v2 (municipal)={v2_c:.4f} vs v3 (mesorregional)={v3_c:.4f} "
                f"(Δ={delta_v2v3:+.4f}, {'v3 melhor' if delta_v2v3 > 0 else 'v2 melhor'})"
            )

    conclusao["pergunta_8_recomendacoes"] = [
        "Agregação mesorregional reduz esparsidade dos dados INMET e melhora cobertura climática",
        "Reportar limiares climáticos v3 como contribuição prática para vigilância epidemiológica regional",
        "Validar estabilidade temporal via walk-forward antes de recomendar uso operacional",
        "Comparar explicitamente granularidade municipal (v2) vs mesorregional (v3) na discussão do TCC",
    ]

    final_report = {
        "pipeline_v3": {
            "mapping": mapping_r,
            "sinan_mesorregiao": sinan_meso_r,
            "inmet_mesorregiao": inmet_meso_r,
            "integration": integration_r,
            "lags": lags_r,
            "anomalias": anomalias_r,
            "splits": splits_r,
        },
        "experimentos": experiments,
        "regressao_v3": reg_v3 if reg_v3 else None,
        "blindness_v3": blindness_v3 if blindness_v3 else None,
        "classificacao_v3": cls_v3 if cls_v3 else None,
        "shap_v3": shap_v3 if shap_v3 else None,
        "walkforward_v3": wf_v3 if wf_v3 else None,
        "comparacao_v2_v3": {
            "v2_final_results": final_v2 if final_v2 else None,
        },
        "conclusao": conclusao,
    }

    with open(DATA_DIR / "32_final_results_v3.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 60}")
    print(f"  CONCLUSÃO FINAL — v3 (MESORREGIONAL)")
    print(f"{'=' * 60}")
    for key, val in conclusao.items():
        if isinstance(val, list):
            print(f"\n  {key}:")
            for item in val:
                print(f"    - {item}")
        else:
            print(f"  {key}: {val}")
    print(f"{'=' * 60}")

    figs = sorted(FIG_DIR.glob("fig_v3*.png"))
    print(f"\n  Total de figuras v3: {len(figs)}")
    tabs = sorted(TABLES_DIR.glob("tab_v3*.tex"))
    print(f"  Total de tabelas LaTeX v3: {len(tabs)}")

    print("\n" + "=" * 60)
    print("  SNIPPETS LATEX (copiar para Overleaf)")
    print("=" * 60)
    print(generate_master_table_v3(experiments))
    if cls_v3:
        print(generate_classification_table_v3(cls_v3))
    if wf_v3:
        print(generate_walkforward_table_v3(wf_v3))
    if shap_v3 and shap_v3.get("thresholds"):
        print(generate_thresholds_table_v3(shap_v3["thresholds"]))


if __name__ == "__main__":
    main()

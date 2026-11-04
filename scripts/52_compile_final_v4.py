#!/usr/bin/env python3
"""v4-52: Compilação final de todos os resultados v4.

Reúne todas as métricas, relatórios e conclusões em um único JSON.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v4"
MODEL_DIR = BASE_DIR / "models" / "regression_v4"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("  v4-52: Compilação Final de Resultados")
    print("=" * 70)

    reports = {
        "40_sinan_aggregation": load_json(DATA_DIR / "40_sinan_meso_report.json"),
        "41_inmet_aggregation": load_json(DATA_DIR / "41_inmet_meso_report.json"),
        "42_integration": load_json(DATA_DIR / "42_integration_report.json"),
        "43_lags": load_json(DATA_DIR / "43_lags_report.json"),
        "44_anomalias": load_json(DATA_DIR / "44_anomalias_report.json"),
        "45_splits": load_json(DATA_DIR / "45_splits_report.json"),
        "46_regression": load_json(DATA_DIR / "46_regression_v4_report.json"),
        "47_walkforward": load_json(DATA_DIR / "47_walkforward_v4_report.json"),
        "48_shap": load_json(DATA_DIR / "48_shap_report.json"),
        "49_optuna": load_json(DATA_DIR / "49_optuna_report.json"),
        "50_drift": load_json(DATA_DIR / "50_drift_report.json"),
        "51_learning_curve": load_json(DATA_DIR / "51_learning_curve_report.json"),
        "54_classification": load_json(DATA_DIR / "54_classification_v4_report.json"),
    }

    loaded = {k: v for k, v in reports.items() if v is not None}
    missing = [k for k, v in reports.items() if v is None]
    print(f"  Reports carregados: {len(loaded)}/{len(reports)}")
    if missing:
        print(f"  Faltando: {missing}")

    reg = reports["46_regression"]
    wf = reports["47_walkforward"]

    enriq_a = reg["modelos"]["A_inmet_enriquecido"] if reg else {}
    enriq_b = reg["modelos"]["B_inmet_enriquecido"] if reg else {}
    enriq_c = reg["modelos"]["C_inmet_enriquecido"] if reg else {}

    optuna = reports.get("49_optuna")
    cls = reports.get("54_classification")

    final = {
        "pipeline_v4": {
            "dataset": {
                "rows": reports["42_integration"]["linhas"] if reports["42_integration"] else None,
                "cols": reports["44_anomalias"]["dataset_cols"] if reports["44_anomalias"] else None,
                "years": reports["42_integration"]["anos"] if reports["42_integration"] else None,
                "mesorregioes": reports["42_integration"]["mesorregioes"] if reports["42_integration"] else None,
                "inmet_coverage_pct": reports["42_integration"]["cobertura_inmet_pct"] if reports["42_integration"] else None,
                "inmet_missing_years": reports["41_inmet_aggregation"]["anos_sem_silver"] if reports["41_inmet_aggregation"] else None,
            },
            "features": {
                "total": reports["45_splits"]["n_features"] if reports["45_splits"] else None,
                "sinan": reports["45_splits"]["n_sinan_features"] if reports["45_splits"] else None,
                "inmet_total": reports["45_splits"]["n_inmet_total"] if reports["45_splits"] else None,
            },
        },
        "regression_results": {
            "variante_a": enriq_a,
            "variante_b": enriq_b,
            "variante_c": enriq_c,
            "deltas": reg["deltas_enriquecido"] if reg else {},
            "testes_t": reg["testes_t"] if reg else {},
            "bootstrap_diffs": reg["bootstrap_diffs"] if reg else {},
        },
        "optuna_results": {
            "n_trials": optuna["n_trials"] if optuna else None,
            "baseline_val_R2_log": optuna["baseline_val_R2_log"] if optuna else None,
            "best_val_R2_log": optuna["best_val_R2_log"] if optuna else None,
            "improvement": optuna["improvement_val"] if optuna else None,
            "adopted": optuna["adopted"] if optuna else None,
        } if optuna else None,
        "classification_results": {
            "melhor_modelo": cls["melhor_modelo_f1_macro"] if cls else None,
            "f1_macro_B_enriquecido": cls["conclusao"]["f1_macro_B_enriquecido"] if cls else None,
            "serie_completa_melhora": cls["conclusao"]["serie_completa_melhora_classificacao"] if cls else None,
        } if cls else None,
        "walkforward_results": {
            "n_models": wf["n_models_total"] if wf else None,
            "resumo_enriquecido": wf["resumo_enriquecido"] if wf else {},
        },
        "conclusoes": {
            "serie_completa_ajuda": reg["conclusao"]["serie_completa_ajuda"] if reg else None,
            "peso_recencia_ajuda": reg["conclusao"]["peso_recencia_ajuda"] if reg else None,
            "melhor_variante": reg["conclusao"]["melhor_variante"] if reg else None,
            "delta_B_vs_A_R2_log": reg["deltas_enriquecido"]["B_vs_A"] if reg else None,
            "delta_B_vs_A_significativo": (reg["testes_t"]["A_vs_B"]["p_value"] < 0.05) if reg else None,
        },
        "reports_individuais": reports,
    }

    out_path = DATA_DIR / "52_final_results_v4.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Salvo: {out_path.name}")

    print(f"\n{'=' * 70}")
    print(f"  CONCLUSÕES FINAIS (INMET enriquecido):")
    print(f"  Var A (baseline 4yr):     R²_log = {enriq_a.get('R2_log', '?')}")
    print(f"  Var B (série 22yr):       R²_log = {enriq_b.get('R2_log', '?')}")
    print(f"  Var C (série + peso):     R²_log = {enriq_c.get('R2_log', '?')}")
    if reg:
        print(f"  Δ B vs A: {reg['deltas_enriquecido']['B_vs_A']:+.4f} (p={reg['testes_t']['A_vs_B']['p_value']:.2e})")
        print(f"  Melhor variante: {reg['conclusao']['melhor_variante']}")
    if wf:
        print(f"  Walk-forward Δ B vs A: {wf['resumo_enriquecido']['delta_B_vs_A']:+.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

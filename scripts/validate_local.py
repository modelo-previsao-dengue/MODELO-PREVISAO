"""Valida integridade dos artefatos locais antes de publicar."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(path, label=""):
    global ok
    p = ROOT / path
    if not p.exists():
        print(f"  FALTA: {path}")
        ok = False
    elif p.stat().st_size == 0:
        print(f"  VAZIO: {path}")
        ok = False


# Splits v4
for var in ["a", "b", "c"]:
    for split in ["train", "val", "test"]:
        check(f"data/model_ready_v4/splits_{var}/{split}.parquet")

# Modelos
for v in ["A", "B", "C"]:
    for arm in ["sinan_only", "inmet_bruto", "inmet_enriquecido"]:
        check(f"models/regression_v4/model_{v}_{arm}.ubj")
        check(f"models/classification_v4/model_{v}_{arm}_cls.ubj")

# Reports
for i in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54]:
    matches = list((ROOT / "data/model_ready_v4").glob(f"{i}_*.json"))
    if not matches:
        print(f"  FALTA: report {i}_*.json")
        ok = False

# Schema
check("data/model_ready_v4/feature_schema_v4.csv")

# Dashboard predictions
preds = list((ROOT / "dashboard/data").glob("predictions_*.parquet"))
if len(preds) != 9:
    print(f"  FALTA: predictions (encontrados {len(preds)}/9)")
    ok = False

# GeoJSON + config
check("data/geojson/mesorregioes.geojson")
check("config/recorte.json")

if ok:
    print("  Validação OK: todos os artefatos presentes")
else:
    print("\n  ERRO: artefatos faltantes")
    sys.exit(1)

#!/usr/bin/env python3
"""Carrega o recorte espaco-temporal de config/recorte.json (FR-1)."""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "recorte.json"


def load(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["anos_recorte"] = list(range(cfg["ano_inicio"], cfg["ano_fim"] + 1))
    cfg["anos_pipeline"] = list(range(cfg["warmup_ano"], cfg["cooldown_ano"] + 1))
    return cfg


def load_v3(path=None):
    """Split da pipeline v3 mesorregional, com a lista de anos ja expandida."""
    v3 = dict(load(path)["split_v3"])
    v3["anos_pipeline"] = list(range(v3["warmup_ano"], v3["ano_fim"] + 1))
    return v3


def describe(cfg):
    return (
        f"UFs={','.join(cfg['ufs'])} | recorte={cfg['ano_inicio']}-{cfg['ano_fim']} "
        f"| pipeline={cfg['warmup_ano']}-{cfg['cooldown_ano']} "
        f"| cobertura minima={cfg['cobertura_municipal_minima']:.0%}"
    )


if __name__ == "__main__":
    c = load()
    print(describe(c))
    print(json.dumps(c, indent=2, ensure_ascii=False))

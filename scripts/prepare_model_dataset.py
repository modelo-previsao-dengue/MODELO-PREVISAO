#!/usr/bin/env python3
"""US-002/US-005/US-006: recorte, split temporal e rotulo de risco sem vazamento.

Mudancas em relacao a versao anterior:

- US-005: o recorte espaco-temporal (UFs, anos, cobertura minima) vem de
  config/recorte.json. O filtro municipal de cobertura climatica e
  recalculado aqui, depois da correcao dos zeros falsos da US-001.
- US-006: o split deixa de ser fixo em train <= 2019. Com a janela nova
  isso deixaria o treino com um ano so.
- US-002: os limiares de risco eram percentis calculados sobre o dataframe
  inteiro, antes do split, e vazavam val e test para dentro do rotulo.
  Passam a sair so do treino, com fallback por UF para municipios que o
  treino nao viu, e sao persistidos para auditoria.

Modos:
    e1  ablaçao limpa, recorte de 6 UFs em 2019-2023, com e sem clima
    e2  referencia de producao, SINAN-only sobre o historico completo

Uso:
    python3 scripts/prepare_model_dataset.py --modo e1
    python3 scripts/prepare_model_dataset.py --modo e2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
INTEGRATED_DIR = BASE_DIR / "data" / "integrated"
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
DOCS_DIR = BASE_DIR / "docs"

TARGET = "notificacoes_t4"
CLASS_TARGET = "risco_surto_t4"
CLASS_TARGET_LEAKY = "risco_surto_t4_com_vazamento"
ID_COLS = ["ibge_municipio", "ano", "semana_epidemiologica"]

FHD_COLS_PATTERN = "con_fhd"
EXCLUDE_COLS = [
    "ano_semana", "week_start", "municipio", "uf", "regiao",
    "source_year", "municipio_resolution", "municipio_source_field",
]

QUANTIS = {"p50": 0.50, "p75": 0.75, "p90": 0.90}


def add_target(df):
    """Target t+4 por municipio, calculado antes de recortar os anos.

    O cool-down existe justamente para isto: as ultimas semanas do ultimo ano
    do recorte buscam o alvo no ano seguinte.
    """
    df = df.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"])
    df[TARGET] = df.groupby("ibge_municipio")["notificacoes"].shift(-4)
    return df


def municipal_coverage(df, cfg, anos):
    """Fracao de semanas do recorte com as tres variaveis climaticas presentes."""
    vars_cov = [v for v in cfg["variaveis_cobertura"] if v in df.columns]
    janela = df[df["ano"].isin(anos)].copy()
    janela["_completo"] = janela[vars_cov].notna().all(axis=1)
    cov = janela.groupby("ibge_municipio")["_completo"].agg(
        semanas="size", semanas_completas="sum"
    ).reset_index()
    cov["cobertura"] = cov["semanas_completas"] / cov["semanas"]
    return cov


def compute_risk_thresholds(train, cfg):
    """Percentis de notificacoes por municipio, calculados so no treino.

    Municipios que o treino nao viu recebem os percentis da propria UF, e a
    origem de cada limiar fica registrada em threshold_source (US-002).
    """
    por_municipio = train.groupby("ibge_municipio")["notificacoes"].agg(
        **{k: (lambda x, q=q: x.quantile(q)) for k, q in QUANTIS.items()}
    ).reset_index()
    por_municipio["threshold_source"] = "municipio"

    por_uf = train.groupby("uf")["notificacoes"].agg(
        **{k: (lambda x, q=q: x.quantile(q)) for k, q in QUANTIS.items()}
    ).reset_index()

    nacional = {k: float(train["notificacoes"].quantile(q)) for k, q in QUANTIS.items()}
    return por_municipio, por_uf, nacional


def apply_risk_class(df, por_municipio, por_uf, nacional, col_out,
                     col_source="threshold_source"):
    """Classifica notificacoes_t4 contra os limiares, com fallback por UF."""
    pm = por_municipio.rename(columns={"threshold_source": col_source})
    out = df.merge(pm, on="ibge_municipio", how="left")
    out = out.merge(por_uf, on="uf", how="left", suffixes=("", "_uf"))

    for k in QUANTIS:
        fonte_uf = out[k].isna() & out[f"{k}_uf"].notna()
        out.loc[fonte_uf, k] = out.loc[fonte_uf, f"{k}_uf"]
        out.loc[out[k].isna(), k] = nacional[k]

    faltava = out[col_source].isna()
    tem_uf = faltava & out["p50_uf"].notna()
    out.loc[tem_uf, col_source] = "uf_fallback"
    out.loc[out[col_source].isna(), col_source] = "nacional_fallback"

    t = out[TARGET]
    out[col_out] = np.select(
        [t <= out["p50"],
         (t > out["p50"]) & (t <= out["p75"]),
         (t > out["p75"]) & (t <= out["p90"]),
         t > out["p90"]],
        [0, 1, 2, 3],
        default=0,
    )
    return out.drop(columns=[c for c in out.columns
                             if c in list(QUANTIS) + [f"{k}_uf" for k in QUANTIS]])


def select_feature_cols(train, cfg):
    """Colunas descartadas por serem IDs, FHD ou quase vazias no treino."""
    fhd = [c for c in train.columns if FHD_COLS_PATTERN in c.lower()]
    vazias = [c for c in train.columns if train[c].isna().mean() > 0.99]
    drop = set(fhd + vazias + EXCLUDE_COLS)
    drop |= {TARGET, CLASS_TARGET, CLASS_TARGET_LEAKY, "threshold_source"}
    if not cfg.get("incluir_notificacoes_atual", False):
        drop.add("notificacoes")
    drop |= set(ID_COLS)
    return [c for c in train.columns if c not in drop], sorted(drop & set(train.columns))


def split_e1(df, cfg):
    s = cfg["split"]
    return (df[df["ano"].isin(s["train"])],
            df[df["ano"].isin(s["val"])],
            df[df["ano"].isin(s["test"])])


def split_e2(df, cfg):
    s = cfg["split_e2"]
    return (df[df["ano"] <= s["train_ate"]],
            df[df["ano"].isin(s["val"])],
            df[df["ano"].isin(s["test"])])


def load_e1(cfg):
    path = INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet"
    print(f"Carregando {path.name}...")
    df = pd.read_parquet(path)
    print(f"  {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def load_e2(cfg):
    anos = list(range(cfg["split_e2"]["ano_inicio"], cfg["cooldown_ano"] + 1))
    print(f"Carregando SINAN Gold, anos {anos[0]}-{anos[-1]}...")
    frames = []
    for ano in anos:
        part = SINAN_GOLD / f"year={ano}"
        if part.exists():
            frames.append(pq.read_table(part).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def recorte_report(df, cfg, cov, mantidos, out_path):
    """Relatorio do recorte por UF, exigido pela US-005."""
    vars_cov = [v for v in cfg["variaveis_cobertura"] if v in df.columns]
    completo = df[vars_cov].notna().all(axis=1)
    rows = []
    for uf, g in df.groupby("uf"):
        rows.append({
            "uf": uf,
            "linhas": len(g),
            "municipios": int(g["ibge_municipio"].nunique()),
            "notificacoes": int(g["notificacoes"].sum()),
            "pct_cobertura_completa": round(float(completo.loc[g.index].mean() * 100), 2),
        })
    tabela = pd.DataFrame(rows).sort_values("linhas", ascending=False)
    total = pd.DataFrame([{
        "uf": "TOTAL", "linhas": len(df),
        "municipios": int(df["ibge_municipio"].nunique()),
        "notificacoes": int(df["notificacoes"].sum()),
        "pct_cobertura_completa": round(float(completo.mean() * 100), 2),
    }])
    tabela = pd.concat([tabela, total], ignore_index=True)
    tabela.to_csv(out_path, index=False)
    return tabela


def main():
    cfg = recorte_config.load()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modo", choices=["e1", "e2"], default="e1")
    args = ap.parse_args()

    out_dir = MODEL_READY_DIR if args.modo == "e1" else MODEL_READY_DIR / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modo: {args.modo.upper()} | {recorte_config.describe(cfg)}\n")

    df = load_e1(cfg) if args.modo == "e1" else load_e2(cfg)

    if args.modo == "e1":
        antes = len(df)
        df = df[df["uf"].isin(cfg["ufs"])].copy()
        print(f"\nFiltro de UF {cfg['ufs']}: {antes:,} -> {len(df):,} linhas")

    print("Calculando target t+4 (antes do recorte de anos, usando o cool-down)...")
    df = add_target(df)

    anos_entrega = (cfg["anos_recorte"] if args.modo == "e1"
                    else list(range(cfg["split_e2"]["ano_inicio"], cfg["ano_fim"] + 1)))

    if args.modo == "e1":
        print(f"Filtro municipal de cobertura >= "
              f"{cfg['cobertura_municipal_minima']:.0%} das semanas...")
        cov = municipal_coverage(df, cfg, anos_entrega)
        mantidos = cov.loc[cov["cobertura"] >= cfg["cobertura_municipal_minima"],
                           "ibge_municipio"]
        print(f"  {len(mantidos):,} de {len(cov):,} municipios mantidos")
        df = df[df["ibge_municipio"].isin(mantidos)].copy()
        cov.to_csv(out_dir / "cobertura_municipal.csv", index=False)
    else:
        cov, mantidos = None, None

    antes = len(df)
    df = df[df["ano"].isin(anos_entrega)].copy()
    print(f"Recorte de anos {anos_entrega[0]}-{anos_entrega[-1]}: "
          f"{antes:,} -> {len(df):,} linhas")

    antes = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Linhas sem target removidas: {antes - len(df):,}")

    if args.modo == "e1":
        tabela = recorte_report(df, cfg, cov, mantidos,
                                DOCS_DIR / "relatorio_recorte_2019_2023.csv")
        print("\nRecorte por UF:")
        print(tabela.to_string(index=False))

    print("\nSplit temporal:")
    train, val, test = (split_e1(df, cfg) if args.modo == "e1"
                        else split_e2(df, cfg))
    for nome, parte in [("train", train), ("val", val), ("test", test)]:
        anos = sorted(parte["ano"].unique().tolist())
        print(f"  {nome:5s} {anos}: {len(parte):,} linhas "
              f"({len(parte)/len(df)*100:.1f}%)")

    print("\nLimiares de risco a partir do treino (US-002)...")
    por_municipio, por_uf, nacional = compute_risk_thresholds(train, cfg)
    print(f"  {len(por_municipio):,} municipios no treino, "
          f"{len(por_uf)} UFs para fallback")

    # Limiar com vazamento, so para dimensionar a inflacao na US-008.
    leaky_mun, leaky_uf, leaky_nac = compute_risk_thresholds(df, cfg)

    partes = {}
    for nome, parte in [("train", train), ("val", val), ("test", test)]:
        p = apply_risk_class(parte, por_municipio, por_uf, nacional, CLASS_TARGET)
        p = apply_risk_class(p, leaky_mun, leaky_uf, leaky_nac, CLASS_TARGET_LEAKY,
                             col_source="_source_leaky")
        partes[nome] = p.drop(columns=["_source_leaky"])

    fontes = pd.concat([p["threshold_source"] for p in partes.values()])
    print(f"  Origem dos limiares: {fontes.value_counts().to_dict()}")
    mudou = sum(int((p[CLASS_TARGET] != p[CLASS_TARGET_LEAKY]).sum())
                for p in partes.values())
    print(f"  Rotulos que mudam sem o vazamento: {mudou:,} de {len(df):,} "
          f"({mudou/len(df)*100:.2f}%)")

    limiares = por_municipio.copy()
    limiares["escopo"] = "municipio"
    uf_rows = por_uf.copy()
    uf_rows["escopo"] = "uf_fallback"
    uf_rows = uf_rows.rename(columns={"uf": "ibge_municipio"})
    nac_row = pd.DataFrame([{**nacional, "ibge_municipio": "NACIONAL",
                             "escopo": "nacional_fallback"}])
    pd.concat([limiares, uf_rows, nac_row], ignore_index=True).to_csv(
        out_dir / "risk_thresholds.csv", index=False)

    feature_cols, dropadas = select_feature_cols(partes["train"], cfg)
    print(f"\nFeatures: {len(feature_cols)} | colunas descartadas: {len(dropadas)}")

    keep = feature_cols + [TARGET, CLASS_TARGET, CLASS_TARGET_LEAKY,
                           "threshold_source"] + ID_COLS
    for nome, parte in partes.items():
        cols = [c for c in dict.fromkeys(keep) if c in parte.columns]
        parte[cols].to_parquet(out_dir / f"{nome}.parquet", index=False)

    schema = pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(partes["train"][c].dtype) for c in feature_cols],
        "pct_missing_train": [round(float(partes["train"][c].isna().mean() * 100), 2)
                              for c in feature_cols],
        "origem": ["inmet" if c in _inmet_cols(partes["train"]) else "sinan"
                   for c in feature_cols],
    })
    schema.to_csv(out_dir / "feature_schema.csv", index=False)

    resumo = {
        "modo": args.modo,
        "config": {k: cfg[k] for k in
                   ["ufs", "ano_inicio", "ano_fim", "cobertura_municipal_minima",
                    "incluir_notificacoes_atual"]},
        "linhas": {n: int(len(p)) for n, p in partes.items()},
        "municipios": {n: int(p["ibge_municipio"].nunique()) for n, p in partes.items()},
        "n_features": len(feature_cols),
        "n_features_inmet": int(schema["origem"].eq("inmet").sum()),
        "rotulos_alterados_sem_vazamento": mudou,
    }
    with open(out_dir / "resumo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"\nSalvo em {out_dir}")
    print(json.dumps(resumo["linhas"], indent=2))


INMET_PREFIXES = (
    "rain_", "temp_", "humidity_", "dewpoint_", "pressure_", "wind_", "radiation_",
)


def _inmet_cols(df):
    return {c for c in df.columns if c.startswith(INMET_PREFIXES)}


if __name__ == "__main__":
    main()

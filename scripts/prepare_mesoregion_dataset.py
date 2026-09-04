#!/usr/bin/env python3
"""US-009: mesmo recorte, agregado por mesorregiao em vez de municipio.

A hipotese e que o municipio-semana e granular demais para o sinal existir:
mais de um terco das linhas tem zero caso, e uma estacao meteorologica a
80 km representa mal o microclima de um municipio pequeno. Agregar para a
mesorregiao aumenta a razao sinal-ruido e aproxima a escala espacial dos
casos da escala em que o clima foi de fato medido.

Regras de agregacao, por tipo de coluna:

    contagens (notificacoes, qt_*)   soma
    proporcoes (prop_*) e medias     media ponderada por notificacoes, que e
                                     o denominador de que elas sairam
    clima                            media entre as estacoes DISTINTAS da
                                     mesorregiao, nao entre municipios: uma
                                     estacao que atende oito municipios
                                     pesaria oito vezes
    derivadas de notificacoes        recalculadas sobre a serie agregada

As derivadas nao podem ser somadas. Lag e media movel sao lineares e
sobreviveriam, mas minimo, maximo, razao e variacao percentual nao: o maximo
de uma soma nao e a soma dos maximos. Recalcular tudo sobre a serie ja
agregada e exato e usa as mesmas definicoes verificadas no dado municipal.

A ponderacao populacional, levantada como questao aberta no PRD, nao foi
usada: o dataset nao carrega populacao e traze-la do IBGE ficaria fora do
recorte deste trabalho. A media por estacao distinta resolve o pior do
problema, que era o peso arbitrario do numero de municipios por estacao.

Uso:
    python3 scripts/prepare_mesoregion_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_model_dataset as pmd
import recorte_config

BASE_DIR = Path(__file__).resolve().parent.parent
IBGE_JSON = BASE_DIR / "data" / "reference" / "ibge_municipios_api.json"
MAPPING = BASE_DIR / "data" / "inmet" / "bronze" / "municipio_estacao_mapping.csv"
OUT_DIR = BASE_DIR / "data" / "model_ready" / "mesorregiao"

CHAVE = ["mesorregiao", "ano", "semana_epidemiologica"]
LAGS = [1, 2, 3, 4, 8, 12]
JANELAS = [3, 4, 8, 12]


def carregar_mesorregioes():
    """Codigo IBGE de 7 digitos -> codigo e nome da mesorregiao."""
    dados = json.loads(IBGE_JSON.read_text(encoding="utf-8"))
    linhas = []
    for m in dados:
        # Municipio criado depois da reforma do IBGE nao tem microrregiao, so
        # a regiao intermediaria, que ocupa o mesmo nivel hierarquico. Hoje e
        # um so no pais e nenhum nas UFs do recorte, mas o cadastro muda.
        micro = m.get("microrregiao")
        meso = (micro["mesorregiao"] if micro
                else m["regiao-imediata"]["regiao-intermediaria"])
        linhas.append({
            "ibge_municipio": str(m["id"]),
            "mesorregiao": str(meso["id"]),
            "mesorregiao_nome": meso["nome"],
            "uf": meso["UF"]["sigla"],
        })
    return pd.DataFrame(linhas)


def classificar_colunas(df):
    """Separa as colunas pelo tipo de agregacao que cada uma admite."""
    clima = [c for c in df.columns if c.startswith(pmd.INMET_PREFIXES)]
    derivadas = [c for c in df.columns
                 if c.startswith("notificacoes_") or c == "is_zero_notification_week"]
    soma = [c for c in df.columns
            if (c.startswith("qt_") or c == "notificacoes") and c not in derivadas]
    ponderada = [c for c in df.columns
                 if c.startswith("prop_")
                 or c in ("idade_media_anos", "atraso_notificacao_medio_dias")]
    return clima, derivadas, soma, ponderada


def agregar(df, clima, soma, ponderada):
    """Uma linha por mesorregiao-semana."""
    print("  somando contagens...")
    agg = df.groupby(CHAVE, sort=False)[soma].sum(min_count=1)

    print("  media ponderada das proporcoes...")
    peso = df["notificacoes"].fillna(0).to_numpy()
    ponderado = {}
    for col in ponderada:
        v = df[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        tmp = pd.DataFrame({
            "num": np.where(ok, v * peso, 0.0),
            "den": np.where(ok, peso, 0.0),
            "simples": np.where(ok, v, np.nan),
        })
        for k in CHAVE:
            tmp[k] = df[k].to_numpy()
        g = tmp.groupby(CHAVE, sort=False).agg(
            num=("num", "sum"), den=("den", "sum"), simples=("simples", "mean"))
        # Semana sem nenhuma notificacao na mesorregiao nao tem denominador;
        # ai a media simples e a unica leitura possivel.
        ponderado[col] = np.where(g["den"] > 0, g["num"] / g["den"].replace(0, np.nan),
                                  g["simples"])
        indice = g.index
    agg = agg.join(pd.DataFrame(ponderado, index=indice))

    print("  media climatica entre estacoes distintas...")
    est = df.drop_duplicates(subset=["codigo_wmo", "ano", "semana_epidemiologica"])
    n_est = est.groupby(CHAVE, sort=False)["codigo_wmo"].nunique().rename("n_estacoes")
    agg = agg.join(est.groupby(CHAVE, sort=False)[clima].mean()).join(n_est)

    return agg.reset_index()


def recalcular_derivadas(df):
    """Refaz as features de notificacoes sobre a serie ja agregada.

    Definicoes conferidas contra o dado municipal: a media movel de w olha as
    w semanas ANTERIORES, sem incluir a atual, e as razoes e variacoes
    percentuais valem zero quando o denominador e zero.
    """
    df = df.sort_values(CHAVE)
    g = df.groupby("mesorregiao", sort=False)["notificacoes"]
    novas = {}

    for k in LAGS:
        novas[f"notificacoes_lag_{k}"] = g.shift(k)

    anterior = g.shift(1)
    for w in JANELAS:
        r = anterior.groupby(df["mesorregiao"], sort=False).rolling(w, min_periods=1)
        novas[f"notificacoes_media_movel_{w}"] = r.mean().reset_index(level=0, drop=True)
        novas[f"notificacoes_min_movel_{w}"] = r.min().reset_index(level=0, drop=True)
        novas[f"notificacoes_max_movel_{w}"] = r.max().reset_index(level=0, drop=True)

    x = df["notificacoes"]
    for k in (1, 4):
        lag = novas[f"notificacoes_lag_{k}"]
        novas[f"notificacoes_diff_{k}"] = x - lag
        novas[f"notificacoes_pct_change_{k}"] = np.where(
            lag.to_numpy() > 0, (x - lag) / lag.replace(0, np.nan), 0.0)

    d1 = pd.Series(novas["notificacoes_diff_1"], index=df.index)
    novas["notificacoes_aceleracao_1"] = d1 - d1.groupby(df["mesorregiao"]).shift(1)

    for w in (4, 8):
        mm = pd.Series(novas[f"notificacoes_media_movel_{w}"], index=df.index)
        novas[f"notificacoes_razao_media_{w}"] = np.where(
            mm.to_numpy() > 0, x / mm.replace(0, np.nan), 0.0)

    novas["is_zero_notification_week"] = (x == 0).astype(int)

    saida = pd.concat([df, pd.DataFrame(novas, index=df.index)], axis=1)
    return saida.fillna({c: 0.0 for c in novas if c.startswith("notificacoes_")})


def main():
    cfg = recorte_config.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pmd.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando o integrado e aplicando o mesmo recorte do E1...")
    df = pmd.load_e1(cfg)
    df = df[df["uf"].isin(cfg["ufs"])].copy()
    df, _ = pmd.collapse_split_weeks(df, "mesorregiao")

    # Mesmos municipios do E1: a comparacao entre granularidades so vale se as
    # linhas de origem forem as mesmas.
    cov = pmd.municipal_coverage(df, cfg, cfg["anos_recorte"])
    mantidos = cov.loc[cov["cobertura"] >= cfg["cobertura_municipal_minima"],
                       "ibge_municipio"]
    df = df[df["ibge_municipio"].isin(mantidos)].copy()
    print(f"  {len(df):,} linhas, {df['ibge_municipio'].nunique():,} municipios")

    meso = carregar_mesorregioes()
    est = pd.read_csv(MAPPING, dtype={"ibge_municipio": str, "codigo_wmo": str})
    df = df.merge(meso[["ibge_municipio", "mesorregiao", "mesorregiao_nome"]],
                  on="ibge_municipio", how="left")
    df = df.merge(est[["ibge_municipio", "codigo_wmo"]].drop_duplicates(),
                  on="ibge_municipio", how="left")
    sem_meso = int(df["mesorregiao"].isna().sum())
    if sem_meso:
        print(f"  aviso: {sem_meso:,} linhas sem mesorregiao no cadastro IBGE")
        df = df[df["mesorregiao"].notna()]

    clima, derivadas, soma, ponderada = classificar_colunas(df)
    print(f"  colunas: {len(soma)} somadas, {len(ponderada)} ponderadas, "
          f"{len(clima)} climaticas, {len(derivadas)} recalculadas")

    print("\nAgregando...")
    agg = agregar(df, clima, soma, ponderada)
    nomes = df[["mesorregiao", "mesorregiao_nome", "uf"]].drop_duplicates("mesorregiao")
    agg = agg.merge(nomes, on="mesorregiao", how="left")
    print(f"  {len(agg):,} linhas, {agg['mesorregiao'].nunique()} mesorregioes")

    print("Recalculando as derivadas sobre a serie agregada...")
    agg = recalcular_derivadas(agg)

    # A partir daqui e o mesmo caminho do E1, so que a chave de municipio
    # passa a ser a mesorregiao, para reaproveitar alvo, split e limiares.
    agg = agg.rename(columns={"ibge_municipio": "_descartado"})
    agg["ibge_municipio"] = agg["mesorregiao"]

    agg = pmd.add_target(agg)
    agg = agg[agg["ano"].isin(cfg["anos_recorte"])].dropna(subset=[pmd.TARGET])
    print(f"  {len(agg):,} linhas apos recorte de anos e alvo t+4")

    train, val, test = pmd.split_e1(agg, cfg)
    print("\nSplit temporal:")
    for nome, parte in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nome:5s} {sorted(parte['ano'].unique().tolist())}: "
              f"{len(parte):,} linhas")

    por_meso, por_uf, nacional = pmd.compute_risk_thresholds(train, cfg)
    leaky = pmd.compute_risk_thresholds(agg, cfg)
    partes = {}
    for nome, parte in [("train", train), ("val", val), ("test", test)]:
        p = pmd.apply_risk_class(parte, por_meso, por_uf, nacional, pmd.CLASS_TARGET)
        p = pmd.apply_risk_class(p, *leaky, pmd.CLASS_TARGET_LEAKY,
                                 col_source="_source_leaky")
        partes[nome] = p.drop(columns=["_source_leaky"])

    limiares = por_meso.copy()
    limiares["escopo"] = "municipio"   # mesmo nome de escopo que o E1, para
    uf_rows = por_uf.copy()            # os baselines lerem sem ramificacao
    uf_rows["escopo"] = "uf_fallback"
    uf_rows = uf_rows.rename(columns={"uf": "ibge_municipio"})
    pd.concat([limiares, uf_rows], ignore_index=True).to_csv(
        OUT_DIR / "risk_thresholds.csv", index=False)

    feature_cols, _ = pmd.select_feature_cols(partes["train"], cfg)
    feature_cols = [c for c in feature_cols
                    if c not in ("mesorregiao", "mesorregiao_nome", "_descartado",
                                 "codigo_wmo", "n_estacoes")]
    keep = feature_cols + [pmd.TARGET, pmd.CLASS_TARGET, pmd.CLASS_TARGET_LEAKY,
                           "threshold_source"] + pmd.ID_COLS
    for nome, parte in partes.items():
        cols = [c for c in dict.fromkeys(keep) if c in parte.columns]
        parte[cols].to_parquet(OUT_DIR / f"{nome}.parquet", index=False)

    pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(partes["train"][c].dtype) for c in feature_cols],
        "pct_missing_train": [round(float(partes["train"][c].isna().mean() * 100), 2)
                              for c in feature_cols],
        "origem": ["inmet" if c.startswith(pmd.INMET_PREFIXES) else "sinan"
                   for c in feature_cols],
    }).to_csv(OUT_DIR / "feature_schema.csv", index=False)

    resumo = {
        "granularidade": "mesorregiao",
        "linhas": {n: int(len(p)) for n, p in partes.items()},
        "mesorregioes": int(agg["mesorregiao"].nunique()),
        "municipios_de_origem": int(df["ibge_municipio"].nunique()),
        "n_features": len(feature_cols),
        "n_features_inmet": sum(1 for c in feature_cols
                                if c.startswith(pmd.INMET_PREFIXES)),
        "estacoes_por_mesorregiao_mediana": float(
            agg.groupby("mesorregiao")["n_estacoes"].max().median()),
    }
    with open(OUT_DIR / "resumo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em {OUT_DIR}")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
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
    """Dataset integrado, lido em Arrow e ja estreitado.

    Lido de forma ingenua, o parquet integrado abre com 7,4 GB residentes: sao
    2 milhoes de linhas por 190 colunas, com os float64 do Parquet e sete
    colunas de texto viradas objeto. O primeiro .copy() do filtro de UF dobra
    isso e a VM do WSL nao aguenta. Descartar na leitura as colunas que nunca
    virariam feature e converter para float32, precisao com que o XGBoost
    trabalha de qualquer forma, resolve sem mudar nenhum resultado.
    """
    path = INTEGRATED_DIR / "sinan_inmet_municipal_weekly.parquet"
    print(f"Carregando {path.name}...")
    esquema = pq.read_schema(path)
    cols = [f.name for f in esquema if f.name not in E1_DROP_COLS]
    alvo = pa.schema([f.with_type(pa.float32()) if f.type == pa.float64() else f
                      for f in esquema if f.name in cols])

    # Ler o arquivo inteiro e so entao converter faria o float64 completo
    # existir em memoria antes do float32. Em lotes, o float64 nunca passa do
    # tamanho de um lote.
    lotes = [pa.RecordBatch.from_struct_array(b.to_struct_array()).cast(alvo)
             for b in pq.ParquetFile(path).iter_batches(
                 batch_size=200_000, columns=cols)]
    tabela = pa.Table.from_batches(lotes, schema=alvo)
    del lotes
    df = tabela.to_pandas()
    del tabela
    gc.collect()
    df["ibge_municipio"] = df["ibge_municipio"].astype(str)
    print(f"  {df.shape[0]:,} linhas, {df.shape[1]} colunas, "
          f"{df.memory_usage(deep=True).sum() / 1e9:.2f} GB em memoria")
    return df


# Colunas do SINAN Gold que nao viram feature nem chave.
E2_DROP_COLS = ["ano_semana", "week_start", "municipio", "regiao",
                "source_year", "municipio_resolution", "municipio_source_field"]

# No E1 a UF e necessaria para o fallback dos limiares e para o relatorio de
# recorte, entao so ela sobrevive entre as colunas de texto.
E1_DROP_COLS = [c for c in E2_DROP_COLS if c != "uf"]

# Colunas suficientes para calcular alvo, split e limiares de risco.
E2_LIGHT_COLS = ["ibge_municipio", "ano", "semana_epidemiologica", "uf",
                 "notificacoes"]


def _e2_part(ano):
    return SINAN_GOLD / f"year={ano}"


def _e2_anos(cfg):
    """Anos carregados (com cool-down) e anos entregues nos splits."""
    ini = cfg["split_e2"]["ano_inicio"]
    return (list(range(ini, cfg["cooldown_ano"] + 1)),
            list(range(ini, cfg["ano_fim"] + 1)))


def _e2_load_light(anos):
    """Passada 1: so as colunas de chave, UF e notificacoes.

    O historico completo do SINAN sao ~7,3 milhoes de linhas por 141 colunas.
    Em float64, com as sete colunas de texto, isso passa de 8 GB antes de
    qualquer operacao, e o sort_values do alvo copia o frame inteiro. Nesta
    maquina o pico derrubava a VM do WSL antes de escrever qualquer parquet.
    Cinco colunas cabem com folga e bastam para tudo que depende da serie
    temporal inteira.
    """
    frames = []
    for ano in anos:
        part = _e2_part(ano)
        if part.exists():
            frames.append(pq.ParquetDataset(part).read(
                columns=E2_LIGHT_COLS).to_pandas())
    light = pd.concat(frames, ignore_index=True)
    light["ibge_municipio"] = light["ibge_municipio"].astype(str)
    return light


def _e2_null_fraction(anos):
    """Passada 2: fracao de nulos por coluna lida dos metadados do Parquet.

    As estatisticas de cada column chunk ja trazem null_count, entao o corte
    de colunas quase vazias sai sem ler uma linha de dados. A fracao e sobre
    as particoes inteiras dos anos de treino, nao sobre as linhas que
    sobrevivem ao dropna do alvo; para um corte em 99% a diferenca e inocua.
    """
    nulos, linhas = {}, 0
    for ano in anos:
        part = _e2_part(ano)
        if not part.exists():
            continue
        for arquivo in sorted(part.glob("*.parquet")):
            md = pq.read_metadata(arquivo)
            linhas += md.num_rows
            for rg in range(md.num_row_groups):
                for c in range(md.num_columns):
                    col = md.row_group(rg).column(c)
                    nome = col.path_in_schema
                    st = col.statistics
                    if st is not None:
                        nulos[nome] = nulos.get(nome, 0) + st.null_count
    return {k: v / linhas for k, v in nulos.items()}, linhas


def _e2_stream_split(nome, anos_split, rotulos, cols_saida, out_dir):
    """Passada 3: escreve o split ano a ano, sem juntar tudo em memoria."""
    writer, esquema, total = None, None, 0
    for ano in anos_split:
        part = _e2_part(ano)
        if not part.exists():
            continue
        disponiveis = [f.name for f in pq.ParquetDataset(part).schema
                       if f.name not in E2_DROP_COLS]
        df = pq.ParquetDataset(part).read(columns=disponiveis).to_pandas()
        df["ibge_municipio"] = df["ibge_municipio"].astype(str)
        # As duplicatas de virada de ano vivem dentro da propria particao do
        # ano, entao a consolidacao vale aqui igual ao frame leve. Sem ela o
        # merge com os rotulos multiplicaria as chaves.
        df, _ = collapse_split_weeks(df, f"{nome}:{ano}")
        df = df.drop(columns=[c for c in ("uf", "notificacoes")
                              if c in df.columns and c in rotulos.columns])
        df = df.merge(rotulos, on=ID_COLS, how="inner")
        if df.empty:
            continue
        df = df.reindex(columns=cols_saida)
        for c in df.columns:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
        tabela = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            esquema = tabela.schema
            writer = pq.ParquetWriter(out_dir / f"{nome}.parquet", esquema)
        else:
            tabela = tabela.cast(esquema)
        writer.write_table(tabela)
        total += len(df)
        del df, tabela
    if writer is not None:
        writer.close()
    return total


def collapse_split_weeks(df, escopo, out_path=None):
    """Consolida semanas de virada de ano duplicadas no SINAN Gold.

    Mesma classe de defeito da US-001, agora do lado epidemiologico: a semana
    que cruza 31/12 aparece duas vezes, uma vinda do arquivo anual anterior e
    outra do seguinte, e cada copia traz so a parte das notificacoes que caiu
    no seu arquivo. Nenhuma das duas e o total da semana.

    Deixar as duas linhas multiplicaria as chaves em qualquer merge; escolher
    uma repetiria o erro do drop_duplicates que a US-004 removeu, que ficava
    com um fragmento parcial. Aqui as notificacoes sao somadas, que e o valor
    real da semana e o que o alvo t+4 precisa, e as demais colunas vem do
    fragmento maior, o mais bem coberto dos dois. Sao 0,02% das linhas do
    historico completo.
    """
    dup = df.duplicated(subset=ID_COLS, keep=False)
    n_linhas = int(dup.sum())
    if not n_linhas:
        return df, pd.DataFrame()

    parte = df[dup].sort_values(ID_COLS + ["notificacoes"])
    somas = parte.groupby(ID_COLS, sort=False)["notificacoes"].sum()
    maior = parte.drop_duplicates(subset=ID_COLS, keep="last").set_index(ID_COLS)

    relatorio = pd.DataFrame({
        "escopo": escopo,
        "notificacoes_fragmento_maior": maior["notificacoes"],
        "notificacoes_consolidadas": somas,
    }).reset_index()

    maior["notificacoes"] = somas
    consolidado = pd.concat([df[~dup], maior.reset_index()], ignore_index=True)
    consolidado = consolidado[df.columns]

    n_chaves = len(relatorio)
    perdido = float(relatorio["notificacoes_consolidadas"].sum()
                    - relatorio["notificacoes_fragmento_maior"].sum())
    print(f"  Semanas partidas consolidadas: {n_chaves:,} chaves "
          f"({n_linhas:,} linhas), {perdido:,.0f} notificacoes que a escolha "
          f"de um fragmento so descartaria")
    if out_path is not None:
        relatorio.to_csv(out_path, index=False)
    return consolidado, relatorio


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



def run_e2(cfg, out_dir):
    """E2: SINAN-only sobre o historico completo, em tres passadas.

    Nenhuma passada segura mais que um ano de dados largos, o que mantem o
    pico de memoria em pouco mais de um GB.
    """
    anos_carga, anos_entrega = _e2_anos(cfg)
    s = cfg["split_e2"]

    print(f"Passada 1/3: chaves e notificacoes, {anos_carga[0]}-{anos_carga[-1]}...")
    light = _e2_load_light(anos_carga)
    print(f"  {len(light):,} linhas, "
          f"{light.memory_usage(deep=True).sum() / 1e9:.2f} GB em memoria")

    light, _ = collapse_split_weeks(
        light, "e2", DOCS_DIR / "semanas_partidas_sinan_e2.csv")
    light = add_target(light)
    antes = len(light)
    light = light[light["ano"].isin(anos_entrega)]
    light = light.dropna(subset=[TARGET])
    print(f"  {antes:,} -> {len(light):,} linhas apos recorte de anos e alvo t+4")

    print("\nSplit temporal:")
    train_l, val_l, test_l = split_e2(light, cfg)
    for nome, parte in [("train", train_l), ("val", val_l), ("test", test_l)]:
        anos = sorted(parte["ano"].unique().tolist())
        faixa = f"{anos[0]}-{anos[-1]}" if len(anos) > 1 else str(anos[0])
        print(f"  {nome:5s} {faixa}: {len(parte):,} linhas "
              f"({len(parte) / len(light) * 100:.1f}%)")

    print("\nLimiares de risco a partir do treino (US-002)...")
    por_municipio, por_uf, nacional = compute_risk_thresholds(train_l, cfg)
    print(f"  {len(por_municipio):,} municipios no treino, "
          f"{len(por_uf)} UFs para fallback")
    leaky_mun, leaky_uf, leaky_nac = compute_risk_thresholds(light, cfg)

    partes = {}
    for nome, parte in [("train", train_l), ("val", val_l), ("test", test_l)]:
        r = apply_risk_class(parte, por_municipio, por_uf, nacional, CLASS_TARGET)
        r = apply_risk_class(r, leaky_mun, leaky_uf, leaky_nac, CLASS_TARGET_LEAKY,
                             col_source="_source_leaky")
        partes[nome] = r.drop(columns=["_source_leaky"])

    fontes = pd.concat([r["threshold_source"] for r in partes.values()])
    print(f"  Origem dos limiares: {fontes.value_counts().to_dict()}")
    mudou = sum(int((r[CLASS_TARGET] != r[CLASS_TARGET_LEAKY]).sum())
                for r in partes.values())
    print(f"  Rotulos que mudam sem o vazamento: {mudou:,} de {len(light):,} "
          f"({mudou / len(light) * 100:.2f}%)")

    limiares = por_municipio.copy()
    limiares["escopo"] = "municipio"
    uf_rows = por_uf.copy()
    uf_rows["escopo"] = "uf_fallback"
    uf_rows = uf_rows.rename(columns={"uf": "ibge_municipio"})
    nac_row = pd.DataFrame([{**nacional, "ibge_municipio": "NACIONAL",
                             "escopo": "nacional_fallback"}])
    pd.concat([limiares, uf_rows, nac_row], ignore_index=True).to_csv(
        out_dir / "risk_thresholds.csv", index=False)

    anos_train = [a for a in anos_entrega if a <= s["train_ate"]]
    print(f"\nPassada 2/3: nulos por coluna nos metadados de "
          f"{anos_train[0]}-{anos_train[-1]}...")
    frac_nulos, linhas_train = _e2_null_fraction(anos_train)

    schema_ref = pq.ParquetDataset(_e2_part(anos_entrega[-1])).schema
    dtypes = {f.name: str(f.type) for f in schema_ref}
    candidatas = [f.name for f in schema_ref if f.name not in E2_DROP_COLS]

    drop = set(EXCLUDE_COLS) | set(ID_COLS)
    drop |= {TARGET, CLASS_TARGET, CLASS_TARGET_LEAKY, "threshold_source"}
    drop |= {c for c in candidatas if FHD_COLS_PATTERN in c.lower()}
    vazias = sorted(c for c in candidatas if frac_nulos.get(c, 0.0) > 0.99)
    drop |= set(vazias)
    if not cfg.get("incluir_notificacoes_atual", False):
        drop.add("notificacoes")
    feature_cols = [c for c in candidatas if c not in drop]
    print(f"  {linhas_train:,} linhas de treino inspecionadas | "
          f"{len(vazias)} colunas quase vazias descartadas")
    print(f"  Features: {len(feature_cols)}")

    rotulo_cols = ID_COLS + [TARGET, CLASS_TARGET, CLASS_TARGET_LEAKY,
                             "threshold_source"]
    cols_saida = list(dict.fromkeys(feature_cols + rotulo_cols))

    print("\nPassada 3/3: escrevendo os splits ano a ano...")
    anos_split = {
        "train": anos_train,
        "val": [a for a in s["val"] if a in anos_entrega],
        "test": [a for a in s["test"] if a in anos_entrega],
    }
    linhas = {}
    for nome in ("train", "val", "test"):
        linhas[nome] = _e2_stream_split(
            nome, anos_split[nome], partes[nome][rotulo_cols], cols_saida, out_dir)
        print(f"  {nome:5s} -> {linhas[nome]:,} linhas")

    pd.DataFrame({
        "feature": feature_cols,
        "dtype": [dtypes.get(c, "") for c in feature_cols],
        "pct_missing_train": [round(frac_nulos.get(c, 0.0) * 100, 2)
                              for c in feature_cols],
        "origem": ["inmet" if c.startswith(INMET_PREFIXES) else "sinan"
                   for c in feature_cols],
    }).to_csv(out_dir / "feature_schema.csv", index=False)

    resumo = {
        "modo": "e2",
        "config": {
            "descricao": s["descricao"],
            "ano_inicio": s["ano_inicio"], "train_ate": s["train_ate"],
            "val": s["val"], "test": s["test"],
            "incluir_notificacoes_atual": cfg["incluir_notificacoes_atual"],
        },
        "linhas": {k: int(v) for k, v in linhas.items()},
        "municipios": {n: int(r["ibge_municipio"].nunique())
                       for n, r in partes.items()},
        "n_features": len(feature_cols),
        "n_features_inmet": 0,
        "rotulos_alterados_sem_vazamento": mudou,
    }
    with open(out_dir / "resumo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"\nSalvo em {out_dir}")
    print(json.dumps(resumo["linhas"], indent=2))


def main():
    cfg = recorte_config.load()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modo", choices=["e1", "e2"], default="e1")
    args = ap.parse_args()

    out_dir = MODEL_READY_DIR if args.modo == "e1" else MODEL_READY_DIR / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Modo: {args.modo.upper()} | {recorte_config.describe(cfg)}\n")

    if args.modo == "e2":
        run_e2(cfg, out_dir)
        return

    df = load_e1(cfg)

    if args.modo == "e1":
        antes = len(df)
        df = df[df["uf"].isin(cfg["ufs"])].copy()
        print(f"\nFiltro de UF {cfg['ufs']}: {antes:,} -> {len(df):,} linhas")

    df, _ = collapse_split_weeks(df, "e1", DOCS_DIR / "semanas_partidas_sinan_e1.csv")

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

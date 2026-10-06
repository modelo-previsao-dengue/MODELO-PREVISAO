#!/usr/bin/env python3
"""US-010: canal endemico e alvo binario de surto por mesorregiao.

O classificador de 4 classes perde para o baseline de persistencia em todos
os bracos da ablacao (F1_macro 0,399 contra 0,484), mesmo com AUC de 0,786 —
ele ordena bem e decide mal. Duas causas: nao ha calibracao de limiar em
lugar nenhum do repositorio, e o proprio rotulo esta confundido com
sazonalidade. As classes de risco atuais (US-002) sao percentis p50/p75/p90
calculados sobre TODAS as semanas do municipio juntas; como dengue e
fortemente sazonal, "acima do p90" significa na pratica "e verao", e o
modelo aprende isso so com week_of_year_sin/cos.

Este script substitui o percentil agregado por um canal endemico: o limiar
passa a ser calculado por semana epidemiologica, comparando cada semana com
a mesma semana em anos anteriores — o metodo padrao de vigilancia (OPAS) e
o que usa o trabalho mais proximo deste TCC (Kirstein et al. 2025, deteccao
de surto em municipios brasileiros via LSTM), que reporta duas variantes:

    (a) surto = notificacoes > media_historica(mesorregiao, semana_epi)
    (b) surto = notificacoes > media_historica + 2 * desvio_padrao

Reportar as duas nao e indecisao: Brady et al. (2015) testaram 102
definicoes de surto sobre dados brasileiros e mostraram que a proporcao de
casos classificados como surto varia ate 65% conforme a definicao escolhida.
Publicar uma definicao unica, sem mostrar sensibilidade a ela, nao se
sustenta.

Janela da climatologia: 2010-2018, estritamente anterior a todos os splits
(train 2019-2021, val 2022, test 2023). Nao e 2014-2018 (5 anos): com n
anos, o maior z-score possivel numa amostra e (n-1)/sqrt(n). Para n=5 isso
da 1,789 < 2 — nenhuma observacao historica poderia exceder o proprio
media+2dp, e a taxa de excedencia in-sample da variante (b) em 2014-2018 e
exatamente 0,000. Com n=9, 8/sqrt(9) = 2,67 > 2, e o limiar volta a ser
atingivel. A janela de 9 anos tambem fica no limite superior do intervalo
de 5 a 9 anos avaliado por Munoz et al. (2026) para parametrizacao de canal
endemico, o que e o lado conservador para a variante estrita.

2015 e 2016 foram anos epidemicos grandes nas 6 UFs do recorte (1,06M e
1,07M notificacoes contra 198k em 2017). A pratica operacional as vezes
exclui anos epidemicos do baseline, mas a literatura adverte contra
decisoes arbitrarias sobre quais anos remover. Decisao: manter os 9 anos
sem exclusao, e reportar uma checagem de sensibilidade excluindo 2015-2016
no resumo_dataset.json.

O alinhamento do rotulo e o ponto mais facil de errar: o alvo compara
notificacoes_t4 contra a climatologia da semana DO ALVO (t+4), nao da
linha. Aqui isso sai de graca porque as flags de surto sao calculadas
semana a semana e so DEPOIS deslocadas com o mesmo shift(-4) por
mesorregiao que pmd.add_target() usa para o alvo de regressao — usar a
mesma mecanica torna as duas series comparaveis por construcao.

Entrada: SINAN gold (2010-2024) + data/model_ready/mesorregiao/*.
Saida:   data/model_ready/mesorregiao_surto/*.

Uso:
    python3 scripts/canal_endemico.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_mesoregion_dataset as pmeso  # noqa: E402
import prepare_model_dataset as pmd  # noqa: E402
import recorte_config  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
MESO_DIR = BASE_DIR / "data" / "model_ready" / "mesorregiao"
OUT_DIR = BASE_DIR / "data" / "model_ready" / "mesorregiao_surto"
DOCS_DIR = BASE_DIR / "docs"

CLIM_INICIO = 2010
CLIM_FIM = 2018  # estritamente anterior ao treino (2019+)
ANOS_EPIDEMICOS_SENSIBILIDADE = [2015, 2016]

VARIANTES = {
    "media": "limiar_media",
    "media_2dp": "limiar_media_2dp",
}
ROTULO_COLS = [
    "surto_media_t", "surto_media_2dp_t",
    "surto_media_t4", "surto_media_2dp_t4",
    "semana_alvo", "ano_alvo",
]


def carregar_series_meso(municipios, cfg):
    """SINAN gold, so os 1.369 municipios do recorte, agregado por mesorregiao.

    Carrega 2010-2024: 2010-2018 vira a climatologia, 2019-2024 vira a serie
    de rotulos (2024 e o cool-down do shift(-4) das ultimas semanas de 2023).
    """
    anos = list(range(CLIM_INICIO, cfg["cooldown_ano"] + 1))
    print(f"Carregando SINAN gold leve, anos {anos[0]}-{anos[-1]}...")
    df = pmd._e2_load_light(anos)
    df = df[df["ibge_municipio"].isin(municipios)].copy()
    print(f"  {len(df):,} linhas, {df['ibge_municipio'].nunique():,} municipios")

    df, relatorio = pmd.collapse_split_weeks(
        df, "canal_endemico", DOCS_DIR / "canal_endemico_semanas_partidas.csv")

    meso_map = pmeso.carregar_mesorregioes()
    df = df.merge(meso_map[["ibge_municipio", "mesorregiao"]],
                  on="ibge_municipio", how="left")
    sem_meso = int(df["mesorregiao"].isna().sum())
    if sem_meso:
        print(f"  aviso: {sem_meso:,} linhas sem mesorregiao no cadastro IBGE")
        df = df[df["mesorregiao"].notna()]

    print("  agregando por mesorregiao-semana...")
    agg = df.groupby(["mesorregiao", "ano", "semana_epidemiologica"],
                     as_index=False)["notificacoes"].sum()
    # Reaproveita pmd.add_target: ele agrupa por ibge_municipio, entao a
    # mesorregiao assume temporariamente esse papel — mesma convencao que
    # prepare_mesoregion_dataset.py ja usa para reaproveitar alvo/split.
    agg["ibge_municipio"] = agg["mesorregiao"].astype(str)
    agg = pmd.add_target(agg)
    print(f"  {len(agg):,} linhas, {agg['mesorregiao'].nunique()} mesorregioes")
    return agg


def construir_canal(serie, ano_ini, ano_fim):
    """Media, desvio-padrao e limiares por (mesorregiao, semana_epidemiologica).

    Duas etapas de agregacao: primeiro colapsa SE 52 e 53 do MESMO ano numa
    unica observacao por media (nao soma) — sem isso, o unico ano de 53
    semanas na janela (2014) deixaria a celula se_clim=53 com n=1, cujo
    z-score maximo possivel e (1-1)/sqrt(1) = 0, tornando a variante (b)
    inatingivel so ali. Mapear se_clim = min(SE, 52) evita a celula de n=1
    por construcao, tanto aqui quanto na hora de aplicar o limiar.
    """
    base = serie[serie["ano"].between(ano_ini, ano_fim)].copy()
    base["se_clim"] = base["semana_epidemiologica"].clip(upper=52)

    valor_ano = base.groupby(["mesorregiao", "ano", "se_clim"],
                             as_index=False)["notificacoes"].mean()
    canal = valor_ano.groupby(["mesorregiao", "se_clim"], as_index=False).agg(
        n_anos=("notificacoes", "count"),
        media=("notificacoes", "mean"),
        dp=("notificacoes", lambda s: s.std(ddof=1)),
    )
    canal["limiar_media"] = canal["media"]
    canal["limiar_media_2dp"] = canal["media"] + 2 * canal["dp"]

    n_meso = serie["mesorregiao"].nunique()
    esperado = n_meso * 52
    if len(canal) != esperado:
        raise RuntimeError(
            f"canal endemico com {len(canal)} celulas, esperado {esperado} "
            f"({n_meso} mesorregioes x 52 semanas)")
    if (canal["n_anos"] < 3).any():
        piores = canal.loc[canal["n_anos"] < 3, ["mesorregiao", "se_clim", "n_anos"]]
        raise RuntimeError(f"celulas com menos de 3 anos de historico:\n{piores}")
    if canal["dp"].isna().any():
        piores = canal.loc[canal["dp"].isna(), ["mesorregiao", "se_clim", "n_anos"]]
        raise RuntimeError(f"desvio-padrao NaN em celulas do canal:\n{piores}")
    if not (canal["limiar_media_2dp"] >= canal["limiar_media"]).all():
        raise RuntimeError("limiar_media_2dp menor que limiar_media em alguma celula")

    return canal


def aplicar_rotulos(serie, canal):
    """Marca surto por semana e desloca -4 semanas por mesorregiao.

    A mesma mecanica de pmd.add_target(): a flag e calculada na semana em
    que o caso ocorre (t) e so depois deslocada para virar o alvo (t+4). O
    surto de uma linha de teste de 2023 compara notificacoes_t4 contra a
    climatologia da semana do ALVO — que e exatamente o que o shift(-4)
    aplicado a flag-ja-calculada garante, sem precisar of um segundo lookup.
    """
    s = serie.copy()
    s["se_clim"] = s["semana_epidemiologica"].clip(upper=52)
    s = s.merge(
        canal[["mesorregiao", "se_clim", "limiar_media", "limiar_media_2dp"]],
        on=["mesorregiao", "se_clim"], how="left")
    faltando = int(s["limiar_media"].isna().sum())
    if faltando:
        raise RuntimeError(
            f"{faltando:,} linhas sem climatologia correspondente (mesorregiao, se_clim)")

    s["surto_media_t"] = s["notificacoes"] > s["limiar_media"]
    s["surto_media_2dp_t"] = s["notificacoes"] > s["limiar_media_2dp"]

    s = s.sort_values(["ibge_municipio", "ano", "semana_epidemiologica"]).reset_index(drop=True)
    g = s.groupby("ibge_municipio", sort=False)
    s["surto_media_t4"] = g["surto_media_t"].shift(-4)
    s["surto_media_2dp_t4"] = g["surto_media_2dp_t"].shift(-4)
    s["semana_alvo"] = g["semana_epidemiologica"].shift(-4)
    s["ano_alvo"] = g["ano"].shift(-4)
    return s


def taxas_positivos(df):
    return {
        "media": round(float(df["surto_media_t4"].mean()), 4),
        "media_2dp": round(float(df["surto_media_2dp_t4"].mean()), 4),
    }


def checagem_sensibilidade(serie_completa, cfg, anos_excluir):
    """Reconstroi o canal excluindo anos epidemicos e compara as taxas.

    Nao bloqueia a execucao: e um diagnostico de robustez para o texto do
    TCC, nao uma validacao de corretude.
    """
    anos_baseline = [a for a in range(CLIM_INICIO, CLIM_FIM + 1) if a not in anos_excluir]
    base = serie_completa[serie_completa["ano"].isin(anos_baseline)]
    canal_alt = construir_canal(base, min(anos_baseline), max(anos_baseline))
    rotulado_alt = aplicar_rotulos(serie_completa, canal_alt)
    anos_recorte = cfg["anos_recorte"]
    final_alt = rotulado_alt[rotulado_alt["ano"].isin(anos_recorte)]
    return {
        "anos_excluidos": anos_excluir,
        "n_anos_baseline": len(anos_baseline),
        "taxas_positivos": taxas_positivos(final_alt),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = recorte_config.load()

    print("Municipios do recorte E1 (1.369 esperados)...")
    municipios = pd.read_parquet(
        BASE_DIR / "data" / "model_ready" / "train.parquet",
        columns=["ibge_municipio"])["ibge_municipio"].unique().tolist()
    print(f"  {len(municipios):,} municipios")

    serie = carregar_series_meso(municipios, cfg)

    print(f"\nConstruindo o canal endemico ({CLIM_INICIO}-{CLIM_FIM})...")
    canal = construir_canal(serie, CLIM_INICIO, CLIM_FIM)
    print(f"  {len(canal):,} celulas (mesorregiao x semana_epi), "
          f"n_anos={canal['n_anos'].iloc[0]} em todas")
    canal.to_csv(OUT_DIR / "canal_endemico.csv", index=False)

    print("\nAplicando rotulos e deslocando -4 semanas...")
    rotulado = aplicar_rotulos(serie, canal)

    anos_recorte = cfg["anos_recorte"]
    final = rotulado[rotulado["ano"].isin(anos_recorte)].copy()
    n_sem_alvo = int(final[["surto_media_t4", "surto_media_2dp_t4"]].isna().any(axis=1).sum())
    if n_sem_alvo:
        raise RuntimeError(
            f"{n_sem_alvo:,} linhas de {anos_recorte[0]}-{anos_recorte[-1]} sem alvo t+4 "
            "— cool-down insuficiente no carregamento do gold")
    n_se53 = int((final["semana_alvo"] == 53).sum())
    print(f"  {len(final):,} linhas no recorte {anos_recorte[0]}-{anos_recorte[-1]}, "
          f"{n_se53} com semana_alvo=53")
    print(f"  taxas de positivos: {taxas_positivos(final)}")

    novas_cols = ["ibge_municipio", "ano", "semana_epidemiologica"] + ROTULO_COLS
    novas_cols += ["notificacoes_t4"]  # para a asserção V1, comparado e descartado
    a_juntar = final[novas_cols].rename(
        columns={"notificacoes_t4": "notificacoes_t4_reconstruido"})

    print("\nJuntando aos splits de data/model_ready/mesorregiao/...")
    resumo_splits = {}
    for nome in ("train", "val", "test"):
        origem = pd.read_parquet(MESO_DIR / f"{nome}.parquet")
        n_origem = len(origem)
        m = origem.merge(a_juntar, on=["ibge_municipio", "ano", "semana_epidemiologica"],
                          how="left")
        if len(m) != n_origem:
            raise RuntimeError(
                f"{nome}: merge produziu {len(m)} linhas, origem tinha {n_origem} "
                "(fan-out inesperado)")
        faltando = m[ROTULO_COLS].isna().any(axis=1).sum()
        if faltando:
            raise RuntimeError(f"{nome}: {faltando:,} linhas sem rotulo apos o merge")

        # V1 — a asserção mestra: notificacoes_t4 reconstruído do gold bate
        # exatamente com o que já estava no parquet do E1 mesorregional.
        diff = (m["notificacoes_t4"].astype("float64")
                - m["notificacoes_t4_reconstruido"].astype("float64")).abs()
        diff_max = float(diff.max())
        if diff_max > 1e-3:
            piores = m.loc[diff > 1e-3,
                           ["ibge_municipio", "ano", "semana_epidemiologica",
                            "notificacoes_t4", "notificacoes_t4_reconstruido"]]
            raise RuntimeError(
                f"{nome}: notificacoes_t4 reconstruido diverge do parquet "
                f"(diff_max={diff_max}). Amostra:\n{piores.head(10)}")
        print(f"  {nome}: {len(m):,} linhas, reconstrucao V1 ok (diff_max={diff_max:.6f})")

        m = m.drop(columns=["notificacoes_t4_reconstruido"])
        for c in ("surto_media_t", "surto_media_2dp_t",
                  "surto_media_t4", "surto_media_2dp_t4"):
            m[c] = m[c].astype(bool)
        m.to_parquet(OUT_DIR / f"{nome}.parquet", index=False)
        resumo_splits[nome] = {
            "n_linhas": len(m),
            "taxas_positivos": taxas_positivos(m),
        }

    for arq in ("feature_schema.csv", "risk_thresholds.csv"):
        (OUT_DIR / arq).write_bytes((MESO_DIR / arq).read_bytes())
    print("\nfeature_schema.csv e risk_thresholds.csv copiados sem alteracao")

    print("\nChecagem de sensibilidade: excluindo anos epidemicos do baseline...")
    sensibilidade = checagem_sensibilidade(serie, cfg, ANOS_EPIDEMICOS_SENSIBILIDADE)
    print(f"  {ANOS_EPIDEMICOS_SENSIBILIDADE} excluidos -> "
          f"taxas {sensibilidade['taxas_positivos']}")

    n_features = pd.read_csv(OUT_DIR / "feature_schema.csv").shape[0]
    resumo = {
        # Mesmo contrato de data/model_ready/mesorregiao/resumo_dataset.json,
        # para que scripts/validar_model_ready.py:checar() leia sem alteracao.
        "granularidade": "mesorregiao_surto",
        "linhas": {nome: r["n_linhas"] for nome, r in resumo_splits.items()},
        "unidades": int(serie["mesorregiao"].nunique()),
        "municipios_de_origem": len(municipios),
        "n_features": int(n_features),
        # Campos proprios deste dataset.
        "climatologia": {
            "janela": [CLIM_INICIO, CLIM_FIM], "n_anos": CLIM_FIM - CLIM_INICIO + 1,
            "n_mesorregioes": int(serie["mesorregiao"].nunique()),
        },
        "variantes": list(VARIANTES.keys()),
        "taxas_positivos_por_split": {nome: r["taxas_positivos"]
                                      for nome, r in resumo_splits.items()},
        "checagem_sensibilidade_anos_epidemicos": sensibilidade,
    }
    with open(OUT_DIR / "resumo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"\nresumo_dataset.json gravado em {OUT_DIR}")
    print("\nUS-010 (canal endemico) completo")


if __name__ == "__main__":
    main()

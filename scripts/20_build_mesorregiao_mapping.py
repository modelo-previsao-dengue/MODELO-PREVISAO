#!/usr/bin/env python3
"""US-200: Mapeamento geográfico município→mesorregião e estação→mesorregião."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "model_ready_v3"
REF_DIR = BASE_DIR / "data" / "reference"
BRONZE_DIR = BASE_DIR / "data" / "inmet" / "bronze"


def haversine_matrix(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2[:, None] - lat1[None, :]
    dlon = lon2[:, None] - lon1[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[None, :]) * np.cos(lat2[:, None]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_municipio_mesorregiao():
    print("Carregando IBGE API JSON...")
    with open(REF_DIR / "ibge_municipios_api.json") as f:
        data = json.load(f)

    rows = []
    skipped = 0
    for item in data:
        micro = item.get("microrregiao")
        if not micro:
            skipped += 1
            continue
        meso = micro.get("mesorregiao")
        if not meso:
            skipped += 1
            continue
        uf = meso.get("UF", {})
        regiao = uf.get("regiao", {})
        rows.append({
            "ibge_municipio": str(item["id"])[:7],
            "nome_municipio": item["nome"],
            "cod_microrregiao": int(micro["id"]),
            "nome_microrregiao": micro["nome"],
            "cod_mesorregiao": int(meso["id"]),
            "nome_mesorregiao": meso["nome"],
            "uf_id": int(uf.get("id", 0)),
            "uf_sigla": uf.get("sigla", "??"),
            "uf_nome": uf.get("nome", "??"),
            "cod_grande_regiao": int(regiao.get("id", 0)),
            "nome_grande_regiao": regiao.get("nome", "??"),
        })

    if skipped:
        print(f"  AVISO: {skipped} itens sem microrregiao/mesorregiao")

    df = pd.DataFrame(rows)
    print(f"  {len(df)} municípios mapeados")
    print(f"  {df['cod_mesorregiao'].nunique()} mesorregiões")
    print(f"  {df['uf_sigla'].nunique()} UFs")
    print(f"  {df['cod_grande_regiao'].nunique()} grandes regiões")

    # NOTA: o JSON fonte tem 5571 municípios, mas "Boa Esperança do Norte" (MT,
    # id 5101837) tem microrregiao=null na API do IBGE (só carrega
    # regiao-imediata/regiao-intermediaria, não o par microrregiao/mesorregiao
    # legado). Esse item é descartado pela regra do passo 3 (nunca inventar
    # valor ausente), resultando em 5570 municípios válidos.
    assert len(df) == 5570, f"Esperado 5570 municípios válidos, obteve {len(df)}"
    assert df["cod_mesorregiao"].nunique() == 137, f"Esperado 137 meso, obteve {df['cod_mesorregiao'].nunique()}"
    assert df["uf_sigla"].nunique() == 27
    assert df["cod_grande_regiao"].nunique() == 5
    assert df["ibge_municipio"].str.len().eq(7).all()
    assert df[["ibge_municipio", "cod_mesorregiao", "uf_sigla"]].notna().all().all()

    df.to_csv(DATA_DIR / "municipio_mesorregiao.csv", index=False)
    print(f"  Salvo: municipio_mesorregiao.csv")
    return df


def build_estacao_mesorregiao(mun_meso):
    print("\nMapeando estações INMET → mesorregião...")
    stations = pd.read_csv(BRONZE_DIR / "estacoes_inmet.csv", dtype={"codigo_wmo": str})
    stations = stations.dropna(subset=["latitude", "longitude"])
    print(f"  {len(stations)} estações com coordenadas")

    mun_coords = pd.read_csv(BRONZE_DIR / "municipios_coords.csv", dtype={"ibge_municipio": str})
    mun_coords = mun_coords.dropna(subset=["latitude", "longitude"])

    st_lat = stations["latitude"].values.astype(float)
    st_lon = stations["longitude"].values.astype(float)
    mn_lat = mun_coords["latitude"].values.astype(float)
    mn_lon = mun_coords["longitude"].values.astype(float)

    print("  Calculando matriz haversine...")
    dist = haversine_matrix(st_lat, st_lon, mn_lat, mn_lon)

    nearest_mun_idx = np.argmin(dist, axis=0)
    nearest_dist = dist[nearest_mun_idx, np.arange(len(stations))]

    station_mun = mun_coords["ibge_municipio"].iloc[nearest_mun_idx].values

    result = pd.DataFrame({
        "codigo_wmo": stations["codigo_wmo"].values,
        "latitude": stations["latitude"].values,
        "longitude": stations["longitude"].values,
        "ibge_municipio": station_mun,
        "distancia_km": np.round(nearest_dist, 2),
    })

    meso_lookup = mun_meso[["ibge_municipio", "cod_mesorregiao", "nome_mesorregiao", "uf_sigla"]].drop_duplicates()
    result = result.merge(meso_lookup, on="ibge_municipio", how="left")

    n_unmapped = result["cod_mesorregiao"].isna().sum()
    if n_unmapped > 0:
        print(f"  AVISO: {n_unmapped} estações sem mesorregião mapeada")
    result = result.dropna(subset=["cod_mesorregiao"])
    result["cod_mesorregiao"] = result["cod_mesorregiao"].astype(int)

    result.to_csv(DATA_DIR / "estacao_mesorregiao.csv", index=False)
    print(f"  Salvo: estacao_mesorregiao.csv ({len(result)} estações)")

    meso_counts = result["cod_mesorregiao"].value_counts()
    n_meso_with = meso_counts.index.nunique()
    print(f"  Mesorregiões com estação: {n_meso_with}/137")
    print(f"  Estações/meso: min={meso_counts.min()}, max={meso_counts.max()}, "
          f"média={meso_counts.mean():.1f}, mediana={meso_counts.median():.1f}")

    return result, meso_counts


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  US-200: Mapeamento Geográfico Mesorregional")
    print("=" * 60)

    mun_meso = build_municipio_mesorregiao()
    station_meso, meso_counts = build_estacao_mesorregiao(mun_meso)

    report = {
        "municipios_mapeados": len(mun_meso),
        "mesorregioes": int(mun_meso["cod_mesorregiao"].nunique()),
        "ufs": int(mun_meso["uf_sigla"].nunique()),
        "grandes_regioes": int(mun_meso["cod_grande_regiao"].nunique()),
        "estacoes_mapeadas": len(station_meso),
        "mesorregioes_com_estacao": int(meso_counts.index.nunique()),
        "estacoes_por_mesorregiao": {
            "min": int(meso_counts.min()),
            "max": int(meso_counts.max()),
            "media": round(float(meso_counts.mean()), 1),
            "mediana": round(float(meso_counts.median()), 1),
        },
    }
    with open(DATA_DIR / "20_mapping_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RESUMO US-200: {len(mun_meso)} municípios → 137 mesorregiões")
    print(f"  {len(station_meso)} estações → {meso_counts.index.nunique()} mesorregiões")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

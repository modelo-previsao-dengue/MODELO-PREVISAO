#!/usr/bin/env python3
"""US-003: Mapeamento estação INMET → município IBGE via haversine."""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / "data" / "inmet" / "bronze"
SINAN_GOLD = BASE_DIR / "data" / "sinan" / "gold" / "sinan_tcc2_v2" / "official_dense"

MUNICIPIOS_URL = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
MAX_DIST_KM = 100


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance matrix (returns km). lat/lon in degrees."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2[:, None] - lat1[None, :]
    dlon = lon2[:, None] - lon1[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[None, :]) * np.cos(lat2[:, None]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_municipality_coords():
    coords_path = BRONZE_DIR / "municipios_coords.csv"
    if coords_path.exists():
        return pd.read_csv(coords_path, dtype={"ibge_municipio": str})

    print("  Baixando coordenadas de municípios do GitHub...")
    try:
        req = urllib.request.Request(MUNICIPIOS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        from io import StringIO
        mun = pd.read_csv(StringIO(raw))
    except Exception as e:
        print(f"  Erro ao baixar: {e}")
        print("  Extraindo municípios do SINAN Gold...")
        df = pq.read_table(SINAN_GOLD, columns=["ibge_municipio"], filters=[("year", "=", 2024)]).to_pandas()
        mun_codes = df["ibge_municipio"].unique()
        print(f"  {len(mun_codes)} municípios encontrados — sem coordenadas, abortando")
        raise

    mun = mun.rename(columns={
        "codigo_ibge": "ibge_municipio",
        "nome": "municipio_nome",
    })
    mun["ibge_municipio"] = mun["ibge_municipio"].astype(str).str[:7]
    mun = mun[["ibge_municipio", "municipio_nome", "latitude", "longitude"]].drop_duplicates(subset=["ibge_municipio"])
    mun.to_csv(coords_path, index=False)
    print(f"  → {len(mun)} municípios salvos em {coords_path}")
    return mun


def main():
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(BRONZE_DIR / "estacoes_inmet.csv")
    stations = stations.dropna(subset=["latitude", "longitude"])
    print(f"Estações INMET com coordenadas: {len(stations)}")

    mun = load_municipality_coords()
    mun = mun.dropna(subset=["latitude", "longitude"])
    print(f"Municípios com coordenadas: {len(mun)}")

    st_lat = stations["latitude"].values.astype(float)
    st_lon = stations["longitude"].values.astype(float)
    mn_lat = mun["latitude"].values.astype(float)
    mn_lon = mun["longitude"].values.astype(float)

    print("Calculando matriz de distâncias haversine...")
    dist = haversine_matrix(st_lat, st_lon, mn_lat, mn_lon)
    # dist shape: (n_municipios, n_stations)

    nearest_idx = np.argmin(dist, axis=1)
    nearest_dist = dist[np.arange(len(mun)), nearest_idx]

    n_50km = (dist <= 50).sum(axis=1)
    n_100km = (dist <= 100).sum(axis=1)

    mapping = pd.DataFrame({
        "ibge_municipio": mun["ibge_municipio"].values,
        "codigo_wmo": stations["codigo_wmo"].iloc[nearest_idx].values,
        "distancia_km": np.round(nearest_dist, 2),
        "n_estacoes_50km": n_50km,
        "n_estacoes_100km": n_100km,
    })

    mapping.loc[mapping["distancia_km"] > MAX_DIST_KM, "sem_cobertura_inmet"] = True
    mapping["sem_cobertura_inmet"] = mapping["sem_cobertura_inmet"].fillna(False).astype(bool)

    out_path = BRONZE_DIR / "municipio_estacao_mapping.csv"
    mapping.to_csv(out_path, index=False)

    total = len(mapping)
    lte50 = (mapping["distancia_km"] <= 50).sum()
    lte100 = (mapping["distancia_km"] <= 100).sum()
    gt100 = (mapping["distancia_km"] > 100).sum()

    print(f"\nMapeamento salvo: {out_path}")
    print(f"Total municípios: {total}")
    print(f"  <= 50km:  {lte50} ({lte50/total*100:.1f}%)")
    print(f"  <= 100km: {lte100} ({lte100/total*100:.1f}%)")
    print(f"  > 100km:  {gt100} ({gt100/total*100:.1f}%)")
    print(f"\nDistância média: {mapping['distancia_km'].mean():.1f} km")
    print(f"Distância mediana: {mapping['distancia_km'].median():.1f} km")
    print(f"Distância máxima: {mapping['distancia_km'].max():.1f} km")


if __name__ == "__main__":
    main()

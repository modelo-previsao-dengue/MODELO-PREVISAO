# Publicacao Operacional - SINAN TCC2

## Object storage

- publicar o diretório `official_dense/` em bucket compatível com S3
- preservar particionamento `year=YYYY`
- publicar junto `feature_catalog.csv`, `schema_gold.json` e `run_manifest.json`

## Banco operacional

1. aplicar `sql/sinan_gold_schema.sql` no PostgreSQL/Supabase
2. carregar a camada gold a partir dos parquet por ano
3. carregar `feature_catalog.csv` na tabela `sinan_feature_catalog`
4. registrar `run_manifest.json` em `sinan_run_manifest`
5. opcionalmente usar os scripts de automacao em `MODELO-PREVISAO/scripts/`

## Variaveis esperadas

- `SUPABASE_DB_*`
- `R2_*`

## Scripts recomendados

- `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2 --dry-run`
- `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2`
- `python3 scripts/sinan_tcc2_publish_postgres.py --version sinan_tcc2_v2`
- `python3 scripts/sinan_tcc2_publish_postgres.py --version sinan_tcc2_v2 --truncate`

## Contrato API sugerido

- `GET /health`
- `GET /v1/series/{ibge_municipio}?start=YYYYWW&end=YYYYWW`
- `GET /v1/features/catalog`
- `GET /v1/top-weeks?year=YYYY&limit=N`

## Execucao local da API

Subida com Docker:

```bash
cd data/sinan/serving/sinan_tcc2_v2
docker compose up --build
```

Validacao minima:

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/v1/top-weeks?year=2024&limit=5"
curl "http://localhost:8000/v1/series/5300108?start=202401&end=202420"
```

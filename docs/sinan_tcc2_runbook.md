# Runbook Operacional - SINAN TCC2

## Versao oficial

- pipeline oficial: `scripts/sinan_tcc2_pipeline.py`
- versao canonica: `sinan_tcc2_v2`

## Reexecucao completa

```bash
cd MODELO-PREVISAO
python3 scripts/sinan_tcc2_pipeline.py --start-year 2000 --end-year 2026 --version sinan_tcc2_v2
```

## Reexecucao parcial

Reprocessar Bronze e Silver, sem Gold:

```bash
python3 scripts/sinan_tcc2_pipeline.py --start-year 2000 --end-year 2026 --version sinan_tcc2_v2 --skip-gold --skip-analytics --skip-serving
```

Reprocessar Gold e analytics a partir de Silver existente:

```bash
python3 scripts/sinan_tcc2_pipeline.py --start-year 2000 --end-year 2026 --version sinan_tcc2_v2 --skip-bronze --skip-silver
```

Gerar apenas o material documental:

```bash
python3 scripts/sinan_tcc2_writeup.py --version sinan_tcc2_v2
```

## Validacao minima dos outputs

Verificar manifesto final:

```bash
cat data/sinan/governance/sinan_tcc2_v2/run_manifest.json
```

Verificar relatorio final:

```bash
cat data/sinan/governance/sinan_tcc2_v2/final_run_report.md
```

Verificar schemas:

```bash
cat data/sinan/governance/sinan_tcc2_v2/schema_silver.json
cat data/sinan/governance/sinan_tcc2_v2/schema_gold.json
```

Verificar qualidade:

```bash
cat data/sinan/governance/sinan_tcc2_v2/quality_summary_silver.json
cat data/sinan/governance/sinan_tcc2_v2/quality_summary_gold.json
```

Atualizar relatórios adicionais de cobertura:

```bash
python3 scripts/sinan_tcc2_governance_refresh.py --version sinan_tcc2_v2
```

## Estrutura esperada

- `data/sinan/bronze/sinan_tcc2_v2/inventory`
- `data/sinan/silver/sinan_tcc2_v2/official_observed/year=YYYY`
- `data/sinan/gold/sinan_tcc2_v2/official_dense/year=YYYY`
- `data/sinan/gold/sinan_tcc2_v2/analytics`
- `data/sinan/governance/sinan_tcc2_v2`
- `data/sinan/serving/sinan_tcc2_v2`

## Publicacao em banco

Arquivos relevantes:

- `data/sinan/serving/sinan_tcc2_v2/sql/sinan_gold_schema.sql`
- `data/sinan/serving/sinan_tcc2_v2/api/load_gold_to_postgres.py`
- `data/sinan/serving/sinan_tcc2_v2/.env.example`

Fluxo:

1. aplicar `sql/sinan_gold_schema.sql`
2. configurar variaveis `SUPABASE_DB_*`
3. exportar `SINAN_GOLD_ROOT`
4. executar `python3 scripts/sinan_tcc2_publish_postgres.py --version sinan_tcc2_v2`

## Publicacao em object storage

Arquivos relevantes:

- `data/sinan/gold/sinan_tcc2_v2/official_dense`
- `data/sinan/gold/sinan_tcc2_v2/feature_catalog.csv`
- `data/sinan/governance/sinan_tcc2_v2/run_manifest.json`
- `data/sinan/governance/sinan_tcc2_v2/schema_gold.json`

Fluxo:

1. configurar variaveis `R2_*`
2. validar com `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2 --dry-run`
3. publicar com `python3 scripts/sinan_tcc2_publish_r2.py --version sinan_tcc2_v2`
4. publicar `official_dense/` preservando `year=YYYY`
5. publicar catalogo, schema e manifesto

## API-ready layer

Arquivos gerados:

- `data/sinan/serving/sinan_tcc2_v2/api/sinan_api.py`
- `data/sinan/serving/sinan_tcc2_v2/api/Dockerfile`
- `data/sinan/serving/sinan_tcc2_v2/api/requirements.txt`
- `data/sinan/serving/sinan_tcc2_v2/docs/sinan_operational_publish.md`
- `data/sinan/serving/sinan_tcc2_v2/docs/api_contract.json`
- `data/sinan/serving/sinan_tcc2_v2/docker-compose.yml`

## Automacao CI

Workflow disponivel:

- `.github/workflows/sinan_tcc2_pipeline.yml`

Endpoints sugeridos:

- `GET /health`
- `GET /v1/series/{ibge_municipio}?start=YYYYWW&end=YYYYWW`
- `GET /v1/features/catalog`
- `GET /v1/top-weeks?year=YYYY&limit=N`

Subida local com Docker:

```bash
cd data/sinan/serving/sinan_tcc2_v2
docker compose up --build
```

Pre-requisito:

- daemon Docker local em execucao

Validacao minima da API:

```bash
curl http://localhost:8000/health
curl "http://localhost:8000/v1/top-weeks?year=2024&limit=5"
curl "http://localhost:8000/v1/series/5300108?start=202401&end=202420"
```

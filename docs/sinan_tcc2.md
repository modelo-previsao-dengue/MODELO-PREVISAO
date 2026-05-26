# Pipeline SINAN - TCC2

Este documento preserva o contexto historico da frente SINAN. A trilha oficial consolidada do TCC2 passou a ser a pipeline nacional em `municipio-semana`, implementada em `scripts/sinan_tcc2_pipeline.py` e formalizada em `docs/sinan_national_pipeline_tcc2.md`.

## Estado oficial atual

- pipeline oficial: `scripts/sinan_tcc2_pipeline.py`
- versao canonica: `sinan_tcc2_v2`
- Bronze/Silver/Gold oficiais: `data/sinan/*/sinan_tcc2_v2`
- pipeline legado preservado: `scripts/sinan_pipeline.py`
- artefatos legados preservados: `data/processed/sinan/*`

## Papel deste documento

- registrar a transicao do TCC1 para o TCC2
- preservar o racional do recorte Brasilia/DF
- explicar o papel do pipeline legado

## Decisao historica de recorte

O TCC1 documenta o problema como "Distrito Federal", mas o codigo efetivamente executado usou:

- `geocode 5300108`
- municipio de `Brasilia`
- periodo `2022-2024`
- serie agregada da API `InfoDengue`

Por isso, a frente SINAN do TCC2 adotou inicialmente duas decisoes:

1. Fazer o download dos arquivos anuais completos do SINAN a partir da fonte oficial.
2. Extrair `Brasilia/DF` em `2022-2024` como recorte analitico comparavel ao TCC1.

## Saidas do pipeline legado

Ao executar `scripts/sinan_pipeline.py`, o pipeline legado gera artefatos em `data/processed/sinan/`, incluindo:

- `tcc1_scope_summary.json`
- `recorte_brasilia_2022_2024.parquet`
- `normalized_recorte_brasilia_2022_2024.parquet`
- `weekly_features_brasilia_2022_2024.csv`
- artefatos de clusterizacao e selecao de atributos do recorte local
- relatorios tecnicos do recorte local e da serie nacional legada em `Brasil-semana`

Esses artefatos continuam validos como evidencia historica, mas nao constituem a camada oficial final do TCC2.

## Limitacao do legado

O principal problema do legado nacional encontrado no workspace foi a granularidade:

- os artefatos nacionais existentes estavam em `Brasil-semana`
- o contrato oficial do TCC2 exige `ibge_municipio + ano_semana`

Por isso, a trilha oficial precisou ser reconstruida em `data/sinan/bronze`, `data/sinan/silver`, `data/sinan/gold`, `data/sinan/governance` e `data/sinan/serving`.

## Execucao oficial do TCC2

```bash
cd MODELO-PREVISAO
python3 scripts/sinan_tcc2_pipeline.py --start-year 2000 --end-year 2026 --version sinan_tcc2_v2
```

Para gerar os documentos finais a partir da versao oficial:

```bash
python3 scripts/sinan_tcc2_writeup.py --version sinan_tcc2_v2
```

## Referencias internas principais

- `docs/sinan_tcc2_diagnostico_inicial.md`
- `docs/sinan_national_pipeline_tcc2.md`
- `docs/sinan_tcc2_latex_alignment.md`
- `data/sinan/governance/sinan_tcc2_v2/final_run_report.md`

# Diagnostico Inicial da Trilha SINAN TCC2

## Escopo do diagnostico

Inspecao executada sobre o workspace real em:

- `MODELO-PREVISAO/scripts`
- `MODELO-PREVISAO/data/raw/sinan`
- `MODELO-PREVISAO/data/processed/sinan`
- `MODELO-PREVISAO/docs`
- `TCC2-DOCS/docs`
- `Overleaf/TCC2 Base FCTE UnB/editaveis`

Critério adotado:

- codigo e artefatos reais como evidencia principal
- documentacao como evidencia complementar
- internet apenas para validacao oficial do IBGE e referencia institucional do SINAN/OpenDataSUS

## Resumo executivo

O repositório já continha uma frente SINAN relevante, mas heterogênea:

1. Um recorte `Brasilia/DF 2022-2024` tecnicamente consistente e reaproveitável.
2. Uma trilha nacional legada com boa cobertura analítica, porém consolidada em `Brasil-semana`, incompatível com o contrato oficial `municipio-semana` exigido para o TCC2.
3. Documentação histórica útil, mas ainda ancorada em narrativa parcial, local ou futura.
4. Texto LaTeX do TCC2 desalinhado com a execução real, ainda citando `InfoDengue`, foco em DF e nacionalização como etapa futura.

Decisão adotada:

- preservar o legado como evidencia histórica
- manter o recorte Brasilia como subproduto confiável
- substituir a trilha nacional legada por uma pipeline oficial nova, governada por Bronze/Silver/Gold em `data/sinan`

## Inventario e classificacao

### Scripts e pipelines

`MODELO-PREVISAO/scripts/sinan_pipeline.py`
- classificacao: `concluido mas inconsistente`
- motivo: materializa resultados úteis, porém com saida nacional em `Brasil-semana`, nao em `municipio-semana`

`MODELO-PREVISAO/scripts/sinan_tcc2_pipeline.py`
- classificacao: `faltando` no estado inicial
- acao: criado como pipeline oficial consolidada do TCC2

`MODELO-PREVISAO/scripts/sinan_tcc2_writeup.py`
- classificacao: `faltando` no estado inicial
- acao: criado para gerar material de metodologia, resultados e entrega

### Dados brutos oficiais

`MODELO-PREVISAO/data/raw/sinan/DENGBR00.json.zip` ate `DENGBR26.json.zip`
- classificacao: `concluido e confiavel`
- motivo: acervo bruto anual ja presente localmente, adequado para Bronze oficial

### Artefatos nacionais legados

`MODELO-PREVISAO/data/processed/sinan/weekly_features_brasil_2000_2026.csv`
- classificacao: `concluido mas inconsistente`
- motivo: serie em `Brasil-semana`, sem chave municipal oficial

`MODELO-PREVISAO/data/processed/sinan/weekly_model_features_brasil_2000_2026.csv`
- classificacao: `concluido mas inconsistente`
- motivo: pronto para modelagem agregada nacional, mas fora do contrato municipio-semana

`MODELO-PREVISAO/data/processed/sinan/feature_catalog_brasil_2000_2026.csv`
- classificacao: `parcial`
- motivo: cataloga features reais, mas referenciadas a dataset legado de granularidade inadequada

`MODELO-PREVISAO/data/processed/sinan/cluster_*_brasil_2000_2026.*`
- classificacao: `parcial`
- motivo: analitica existente e aproveitavel como antecedente metodologico, mas nao como camada oficial final

`MODELO-PREVISAO/data/processed/sinan/selected_features*_brasil_2000_2026.*`
- classificacao: `parcial`
- motivo: selecao real ja executada, mas sobre base legado `Brasil-semana`

### Recorte Brasilia / DF

`MODELO-PREVISAO/data/processed/sinan/recorte_brasilia_2022_2024.*`
- classificacao: `concluido e confiavel`

`MODELO-PREVISAO/data/processed/sinan/weekly_features_brasilia_2022_2024.csv`
- classificacao: `concluido e confiavel`

`MODELO-PREVISAO/data/processed/sinan/weekly_model_features_brasilia_2022_2024.csv`
- classificacao: `concluido e confiavel`

Motivo:

- recorte local consistente, útil como evidencia de maturidade prévia da frente SINAN

### Documentacao tecnica

`MODELO-PREVISAO/docs/task_engenharia_atributos_eda_sinan.md`
- classificacao: `concluido e confiavel`
- motivo: documenta engenharia de atributos e EDA nacional efetivamente realizadas

`MODELO-PREVISAO/docs/sinan_feature_engineering_eda.md`
- classificacao: `parcial`
- motivo: util como memoria tecnica, mas ainda nao congelava o contrato final Bronze/Silver/Gold

`TCC2-DOCS/docs/ColetaProcessamentoDadosSINAN/*`
- classificacao: `concluido e confiavel`
- motivo: boa evidencia do metodo local historico

`TCC2-DOCS/docs/IntegracaoUnificacaoDadosSINAN/*`
- classificacao: `concluido e confiavel`
- motivo: bom registro de integracao e unificacao anteriores

### Texto academico LaTeX

`Overleaf/TCC2 Base FCTE UnB/editaveis/resumo.tex`
- classificacao: `obsoleto`

`Overleaf/TCC2 Base FCTE UnB/editaveis/introducao.tex`
- classificacao: `obsoleto`

`Overleaf/TCC2 Base FCTE UnB/editaveis/referencial_teorico.tex`
- classificacao: `obsoleto`

Motivo:

- ainda tratam a frente SINAN nacional como futura ou mencionam fontes/narrativas nao aderentes ao pipeline real consolidado

## Divergencias encontradas

### Divergencia 1: granularidade nacional

- documentacao e expectativa do TCC2 exigem `municipio-semana`
- artefatos nacionais legados existentes estavam em `Brasil-semana`

Impacto:

- impossibilita uso direto como camada oficial para modelagem municipal e integracao futura

### Divergencia 2: narrativa academica

- LaTeX ainda sugere `InfoDengue` e foco DF como base principal
- o workspace real ja evidencia uma frente SINAN muito mais ampla

Impacto:

- risco de incoerencia entre defesa escrita e execucao real

### Divergencia 3: regra temporal da serie

- no legado, a derivacao semanal permitia misturar semanas antigas ou ambiguas
- na consolidacao oficial, a regra temporal foi congelada por semana epidemiologica de notificacao

Impacto:

- evita contaminacao da Silver/Gold com anos absurdos ou semanas semanticamente indevidas para `notificacoes`

## Decisoes de consolidacao

Mantido:

- brutos anuais oficiais locais
- recorte Brasilia/DF
- documentacao tecnica e historica util
- evidencias de EDA, clusterizacao e feature selection como memoria metodologica

Refatorado:

- pipeline nacional
- contrato metodologico
- organizacao de datasets em Bronze/Silver/Gold
- governanca, schema, lineage e camada serving

Substituido como camada oficial:

- artefatos `data/processed/sinan/*brasil_2000_2026*`

## Estado inicial por fase obrigatoria

Fase 1. Diagnostico
- status inicial: `parcial`

Fase 2. Congelamento metodologico
- status inicial: `faltando`

Fase 3. Bronze oficial
- status inicial: `parcial`

Fase 4. Silver nacional municipio-semana
- status inicial: `faltando`

Fase 5. Gold oficial para modelagem
- status inicial: `faltando`

Fase 6. EDA, clusterizacao e selecao
- status inicial: `parcial`

Fase 7. Governanca e lineage
- status inicial: `faltando`

Fase 8. Camada operacional
- status inicial: `faltando`

Fase 9. API-ready layer
- status inicial: `faltando`

Fase 10. Material final para TCC2
- status inicial: `parcial`

## Nota de execucao real da versao oficial

A versao oficial `sinan_tcc2_v2` foi consolidada em duas etapas operacionais sobre os mesmos artefatos de versao:

1. execucao completa do bruto ate a Silver e Gold, que confirmou a correção metodologica principal da serie mas falhou ao iniciar a escrita dos artefatos de `analytics/` por ausencia de criacao explicita do diretorio
2. reexecucao da mesma versao a partir da Silver existente, com correção no pipeline para criar `gold/analytics/` e concluir analytics, governanca final, camada serving e documentacao

Impacto:

- nao houve necessidade de reprocessar novamente os brutos oficiais apos a correção do erro operacional de diretório
- a Silver oficial e a Gold oficial preservadas em `sinan_tcc2_v2` correspondem ao contrato metodologico final corrigido

# Alinhamento do LaTeX com a Execucao Real da Trilha SINAN

## Objetivo

Este documento registra, com referencias objetivas, os trechos do material em LaTeX que ainda descrevem uma trilha antiga ou incompleta do SINAN e precisam ser atualizados para refletir a execucao real do TCC2.

## Divergencias principais

### 1. Fonte epidemiologica descrita como InfoDengue

Arquivo:

- [resumo.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/resumo.tex:2)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:144)

Problema:

- o texto afirma que os dados epidemiologicos foram obtidos `via InfoDengue`
- isso nao corresponde a execucao real do TCC2

Estado real:

- a trilha SINAN do TCC2 processa microdados oficiais anuais do OpenDataSUS/SINAN
- o InfoDengue aparece apenas como evidencia historica do TCC1

Atualizacao recomendada:

- substituir referencias a `SINAN via InfoDengue` por `microdados oficiais anuais do SINAN/Dengue publicados no OpenDataSUS`

### 2. Escopo descrito como Distrito Federal como eixo central da trilha

Arquivo:

- [resumo.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/resumo.tex:2)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:107)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:109)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:111)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:115)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:119)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:123)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:137)

Problema:

- o texto apresenta o Distrito Federal como escopo epidemiologico principal do TCC2
- isso conflita com a consolidacao real da trilha SINAN nacional em municipio-semana

Estado real:

- o recorte Brasilia/DF permanece como evidencia historica e comparativa
- a camada oficial SINAN do TCC2 passa a ser nacional em `municipio-semana`

Atualizacao recomendada:

- reescrever a narrativa para:
  - `camada epidemiologica nacional oficial`
  - `recorte Brasilia/DF como caso comparativo e evidencial`

### 3. Cobertura temporal do SINAN descrita como desde 2007

Arquivo:

- [referencial_teorico.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/referencial_teorico.tex:111)
- [referencial_teorico.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/referencial_teorico.tex:117)

Problema:

- o texto diz que o dataset utilizado vai `desde 2007`
- isso nao corresponde ao inventario bruto local nem aos artefatos nacionais ja encontrados

Estado real:

- o workspace atual contem os brutos oficiais `DENGBR00` ate `DENGBR26`
- a trilha nacional oficial foi congelada para `2000-2026`

Atualizacao recomendada:

- substituir `desde 2007` por `de 2000 a 2026 no inventario bruto oficial disponivel localmente`
- manter nota de que anos recentes sao sujeitos a atualizacao do portal

### 4. Chave espacial descrita como codigo IBGE de 7 digitos direto no microdado

Arquivo:

- [referencial_teorico.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/referencial_teorico.tex:113)

Problema:

- o texto sugere que o microdado ja chega com codigo municipal IBGE final em 7 digitos
- isso simplifica demais a realidade do processamento

Estado real:

- o microdado do SINAN mistura codificacoes territoriais de 6 digitos, casos especiais do DF e campos alternativos
- a trilha oficial do TCC2 faz reconciliacao municipal com a tabela oficial do IBGE

Atualizacao recomendada:

- descrever explicitamente a etapa de reconciliacao territorial antes da agregacao municipio-semana

### 5. Metodologia ainda fala em expansao futura para todos os municipios

Arquivo:

- [resumo.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/resumo.tex:2)
- [introducao.tex](/Users/filippoferrari/Documents/UnB/TCC/Overleaf/TCC2%20Base%20FCTE%20UnB/editaveis/introducao.tex:119)

Problema:

- o texto trata a nacionalizacao como etapa futura

Estado real:

- a nacionalizacao da trilha SINAN foi implementada na execucao do TCC2

Atualizacao recomendada:

- trocar formulações de futuro por formulações de execucao concluida

## Formula de reposicionamento narrativo

Narrativa recomendada para a monografia:

- o TCC1 utilizou uma serie agregada comparativa focada em Brasilia/DF
- o TCC2 consolidou a espinha dorsal epidemiologica nacional do projeto a partir dos microdados oficiais do SINAN
- a unidade analitica oficial tornou-se `municipio-semana`
- o recorte Brasilia/DF segue relevante apenas como evidencia historica, ponto de comparacao e interface com a trilha climatica

## Arquivos de suporte para a reescrita

Fontes internas prioritarias:

- [sinan_national_pipeline_tcc2.md](/Users/filippoferrari/Documents/UnB/TCC/MODELO-PREVISAO/docs/sinan_national_pipeline_tcc2.md)
- [sinan_tcc2_latex_snippets.tex](/Users/filippoferrari/Documents/UnB/TCC/MODELO-PREVISAO/docs/sinan_tcc2_latex_snippets.tex)
- `MODELO-PREVISAO/data/sinan/governance/sinan_tcc2_v2/final_run_report.md`
- `MODELO-PREVISAO/data/sinan/governance/sinan_tcc2_v2/run_manifest.json`
- `MODELO-PREVISAO/data/sinan/gold/sinan_tcc2_v2/analytics/relatorio_eda.md`

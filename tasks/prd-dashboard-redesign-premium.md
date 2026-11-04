# PRD: Redesign Premium — DengueCast Dashboard

## Introduction

Redesign completo da dashboard DengueCast, transformando o sistema atual (Streamlit com CSS inline e graficos Plotly basicos) em uma aplicacao com design premium, fluida e interativa — inspirada em Nixtio, Flatstudio e Framer. A dashboard e um sistema preditivo de surtos de dengue que usa modelos XGBoost treinados com dados SINAN + INMET (2001–2026) para prever notificacoes e classificar risco de surto em 137 mesorregioes brasileiras.

O publico-alvo primario e profissionais de vigilancia epidemiologica (alertas claros, decisoes rapidas), com uma pagina dedicada explicando o funcionamento completo do sistema para a banca do TCC.

Stack: Streamlit como base + componentes React/JS injetados via `st.components.v1.html()` para animacoes full motion design (count-up, fade-in, transicoes cinematicas nos graficos, loading states animados). Inclui mapa coropletico interativo do Brasil e exportacao de relatorio PDF.

## Goals

- Substituir o visual "HTML puro" por design premium com identidade visual coesa
- Implementar full motion design: animacoes de entrada, count-up, transicoes suaves em graficos, loading states
- Criar mapa coropletico interativo do Brasil com risco por mesorregiao
- Adicionar pagina "Sobre o Sistema" com explicacao tecnica completa (para banca do TCC)
- Permitir exportacao de relatorio PDF com alertas e previsoes
- Tornar a interface intuitiva para profissionais de vigilancia epidemiologica
- Garantir responsividade em telas de 1024px a 1920px+
- Manter performance aceitavel com 180k rows de dados

## User Stories

### US-001: Design System e Tema Global

**Description:** Como usuario, quero que toda a interface tenha uma identidade visual coesa e premium para que o sistema transmita profissionalismo e confiabilidade.

**Acceptance Criteria:**
- [ ] Paleta de cores definida com variantes: primary (indigo-600), danger (red), warning (amber), success (green), neutrals (slate)
- [ ] Tipografia DM Sans carregada via Google Fonts com hierarquia clara (headings, body, captions, KPI numbers)
- [ ] Tema forcado light via `.streamlit/config.toml` — sem mistura de temas
- [ ] Sidebar branca com borda sutil, brand icon com gradiente, separadores claros
- [ ] Todos os componentes Streamlit nativos estilizados (selectbox, slider, tabs, metrics, dataframes)
- [ ] CSS global sem conflito dark/light — todos os backgrounds, textos e bordas explicitamente definidos
- [ ] Verify in browser using dev-browser skill

### US-002: KPI Cards com Animacao Count-Up

**Description:** Como profissional de vigilancia, quero ver os numeros de alerta (Surto, Alto, Moderado, Baixo) animarem ao carregar para que a interface seja dinamica e direcione minha atencao.

**Acceptance Criteria:**
- [ ] 4 KPI cards no topo da tab Alertas com layout grid responsivo
- [ ] Cada card tem: dot colorido, numero grande, label uppercase
- [ ] Numeros animam de 0 ate o valor final em ~1.2s com easing (ease-out)
- [ ] Cards tem hover effect: sutil box-shadow + translateY(-2px) com transition 200ms
- [ ] Implementado via `st.components.v1.html()` com JavaScript inline para count-up
- [ ] Funciona corretamente quando Streamlit re-renderiza (sem re-animar em cada interacao de filtro)
- [ ] Verify in browser using dev-browser skill

### US-003: Graficos Plotly Premium com Transicoes

**Description:** Como usuario, quero que os graficos tenham aparencia profissional com animacoes de entrada para que a experiencia seja fluida e moderna.

**Acceptance Criteria:**
- [ ] Todos os graficos Plotly usam layout padronizado: fundo transparente, grid sutil (#f1f5f9), font DM Sans
- [ ] Graficos de barras horizontais (UF risk) com corner-radius e color-scale suave
- [ ] Graficos de linha (serie temporal) com line-width calibrado, fill-area com opacity baixa
- [ ] Scatter plot (previsto vs real) com opacity .25, 45-degree reference line tracejada
- [ ] Animacao `transition_duration=800` em graficos Plotly onde suportado
- [ ] Modebar do Plotly com opacity .15 (visivel apenas no hover)
- [ ] Cada grafico envolto em card branco com borda `#f1f5f9`, border-radius 16px, hover shadow
- [ ] Legenda horizontal no topo do grafico, font-size 10, sem caixa
- [ ] Verify in browser using dev-browser skill

### US-004: Mapa Coropletico Interativo do Brasil

**Description:** Como profissional de vigilancia, quero ver um mapa do Brasil colorido por nivel de risco por mesorregiao para que eu identifique rapidamente as regioes criticas.

**Acceptance Criteria:**
- [ ] Mapa do Brasil renderizado com Plotly `choropleth_mapbox` ou GeoJSON + `go.Choroplethmapbox`
- [ ] GeoJSON das mesorregioes do IBGE carregado (arquivo local ou URL publica do IBGE)
- [ ] Cada mesorregiao colorida pelo risco previsto mais recente (escala: verde→amarelo→laranja→vermelho)
- [ ] Hover mostra: nome da mesorregiao, UF, risco previsto, notificacoes previstas t+4, P(Surto)
- [ ] Click na mesorregiao filtra os dados da dashboard para aquela regiao
- [ ] Mapa posicionado na tab Alertas, abaixo dos KPI cards e acima da tabela
- [ ] Zoom/pan funcional com controles intuitivos
- [ ] Performance aceitavel (renderiza em < 3s)
- [ ] Verify in browser using dev-browser skill

### US-005: Tabela de Alertas Estilizada

**Description:** Como profissional de vigilancia, quero que a tabela de alertas tenha formatacao visual clara (badges de risco, barras de progresso) para que eu identifique prioridades sem ler cada celula.

**Acceptance Criteria:**
- [ ] Tabela renderizada com `st.dataframe` usando `column_config` para formatacao
- [ ] Coluna "Risco" mostra badges coloridos (pill) em vez de texto puro: Surto=vermelho, Alto=laranja, Moderado=amarelo, Baixo=verde
- [ ] Coluna "P(Surto)" mostra barra de progresso horizontal proporcional (0-100%)
- [ ] Coluna "Prev. t+4" formatada com separador de milhar
- [ ] Linhas ordenadas por risco decrescente, depois por notificacoes previstas
- [ ] Altura da tabela ajustada ao conteudo (sem scroll desnecessario, max 500px)
- [ ] Hover na linha destaca com background sutil
- [ ] Verify in browser using dev-browser skill

### US-006: Tab Regiao com Painel Detalhado

**Description:** Como profissional de vigilancia, quero selecionar uma mesorregiao e ver um painel completo com serie temporal, probabilidades e metricas para que eu analise a situacao localmente.

**Acceptance Criteria:**
- [ ] Header card com: nome da mesorregiao, UF, badge de risco, periodo mais recente, previsao, P(Surto)
- [ ] 4 KPI metrics em row: Notificacoes atual, Previsao t+4, Risco, P(Surto)
- [ ] Grafico de serie temporal (2001-2026): linhas Real vs Previsto vs Atual, com legenda horizontal
- [ ] Grafico de barras overlay: P(Surto) + P(Alto) por periodo, opacity diferenciada
- [ ] Metricas locais: R²(log), MAE, Acuracia de risco — em 3 colunas
- [ ] Selectbox com busca por nome da mesorregiao (137 opcoes)
- [ ] Verify in browser using dev-browser skill

### US-007: Tab Temporal com Analise Agregada

**Description:** Como usuario (banca ou vigilancia), quero visualizar a performance do modelo ao longo do tempo para que eu avalie a qualidade das previsoes por ano e nacionalmente.

**Acceptance Criteria:**
- [ ] Scatter plot previsto vs real com 5000 samples, colorido por risco, linha 45 graus
- [ ] Grafico de linha: total nacional semanal (real vs previsto), 2001-2026
- [ ] Grafico de barras: R² por ano, colorido por split (treino=indigo, validacao=amarelo, teste=vermelho)
- [ ] Cada grafico em chart-card com spacing vertical adequado (2rem entre secoes)
- [ ] Section titles uppercase em #94a3b8 como separadores
- [ ] Verify in browser using dev-browser skill

### US-008: Tab Modelo com Performance e SHAP

**Description:** Como membro da banca do TCC, quero ver as metricas de performance do modelo e a importancia das features para avaliar a qualidade do trabalho.

**Acceptance Criteria:**
- [ ] 4 KPI metrics: R²(log), RMSE, MAE, IC 95%
- [ ] Grafico de barras: R² por variante (A, B, C) com error bars (IC 95%)
- [ ] Tabela de classificacao de risco: F1 macro, AUC, F1 surto por variante
- [ ] Grafico de linha: Walk-Forward Validation por fold/ano, 3 variantes
- [ ] Grafico de barras horizontal: Top N features SHAP com slider (10-50), escala indigo
- [ ] Coluna SHAP correta: usa `shap_mean_abs_B` (nao `mean_abs_shap`)
- [ ] Verify in browser using dev-browser skill

### US-009: Pagina "Sobre o Sistema"

**Description:** Como membro da banca do TCC, quero uma pagina explicando o funcionamento completo do sistema (pipeline, modelos, dados) para que eu entenda a metodologia sem precisar ler o TCC inteiro.

**Acceptance Criteria:**
- [ ] Nova tab ou pagina separada acessivel pelo menu: "Sobre"
- [ ] Secao "Pipeline de Dados": diagrama visual mostrando SINAN → limpeza → features → INMET → merge → splits
- [ ] Secao "Modelos": explicacao das 3 variantes (A, B, C), 3 sub-modelos, regressao + classificacao
- [ ] Secao "Features": tabela com numero de features por sub-modelo (110, 122, 266)
- [ ] Secao "Resultados": resumo dos principais achados (B melhor que A, C nao melhora sobre B)
- [ ] Secao "Dados": cobertura temporal (2001-2026), numero de registros, mesorregioes
- [ ] Design coerente com o resto da dashboard (mesmos cards, tipografia, cores)
- [ ] Verify in browser using dev-browser skill

### US-010: Exportacao de Relatorio PDF

**Description:** Como profissional de vigilancia, quero exportar um relatorio PDF com os alertas e previsoes atuais para que eu possa compartilhar com a equipe ou arquivar.

**Acceptance Criteria:**
- [ ] Botao "Exportar PDF" visivel na tab Alertas (ou no sidebar)
- [ ] PDF gerado com: cabecalho DengueCast, data/hora, filtros aplicados
- [ ] Conteudo do PDF: KPIs de alerta, tabela de mesorregioes em risco, grafico de evolucao semanal
- [ ] PDF usa mesma paleta de cores da dashboard
- [ ] Gerado via `reportlab` ou `fpdf2` ou `weasyprint` (biblioteca Python)
- [ ] Download automatico via `st.download_button`
- [ ] PDF legivel em A4, com margens adequadas
- [ ] Verify in browser — botao funciona e PDF abre corretamente

### US-011: Loading States e Transicoes de Tab

**Description:** Como usuario, quero ver estados de loading animados e transicoes suaves entre tabs para que a experiencia seja premium e fluida.

**Acceptance Criteria:**
- [ ] Spinner customizado (substituir o spinner padrao do Streamlit) com animacao CSS
- [ ] Skeleton loading para KPI cards enquanto dados carregam (blocos cinza pulsando)
- [ ] Transicao fade-in (opacity 0→1, 300ms) quando conteudo de uma tab aparece
- [ ] Implementado via `st.components.v1.html()` com CSS animations
- [ ] Sem delay artificial — animacoes acontecem naturalmente no carregamento real
- [ ] Verify in browser using dev-browser skill

### US-012: Responsividade e Polish Final

**Description:** Como usuario em qualquer dispositivo, quero que a dashboard funcione bem em telas de 1024px a 1920px+ para que eu acesse de laptop ou monitor grande.

**Acceptance Criteria:**
- [ ] KPI grid colapsa para 2 colunas em telas < 1200px
- [ ] Mapa do Brasil redimensiona proporcionalmente
- [ ] Graficos Plotly usam `use_container_width=True` em todos os casos
- [ ] Sidebar nao sobrepoe conteudo em telas menores
- [ ] Tabela de alertas com scroll horizontal se necessario em telas menores
- [ ] Sem overflow horizontal na pagina em nenhuma resolucao
- [ ] Verify in browser em janela 1024px e 1920px

## Functional Requirements

- FR-01: O sistema deve forcar tema light via `.streamlit/config.toml` (background #ffffff, secondary #f8fafc, text #0f172a)
- FR-02: Todos os componentes visuais devem usar a fonte DM Sans carregada via Google Fonts CDN
- FR-03: KPI cards devem renderizar via `st.components.v1.html()` com JavaScript count-up animation (0 → valor final, 1.2s, ease-out)
- FR-04: Graficos Plotly devem usar layout padronizado: fundo transparente, grid #f1f5f9, font DM Sans 11px, margem compacta
- FR-05: Cada grafico deve estar envolto em div `.chart-card` (bg #fff, border 1px #f1f5f9, radius 16px)
- FR-06: O mapa coropletico deve carregar GeoJSON das mesorregioes IBGE e renderizar via Plotly Mapbox ou `plotly.express.choropleth`
- FR-07: A tabela de alertas deve usar `st.dataframe(column_config=...)` para formatacao visual (progress bars, badges)
- FR-08: A tab "Sobre" deve conter explicacao tecnica do pipeline, modelos e resultados
- FR-09: A exportacao PDF deve gerar arquivo A4 com KPIs, tabela de alertas e info do periodo via `st.download_button`
- FR-10: Loading states devem usar CSS `@keyframes pulse` para skeleton effect
- FR-11: A sidebar deve conter: brand + logo, seletores (modelo, features), filtros (ano, UF), stats resumo
- FR-12: A coluna SHAP deve usar `shap_mean_abs_B` (ou fallback para segunda coluna do CSV)
- FR-13: Dados carregados via `load_all_data()` devem incluir train+val+test (180k rows, 2001-2026)
- FR-14: Previsoes devem ser geradas via XGBoost Booster com `load_model()` + `DMatrix` + `predict()`
- FR-15: O mapa deve suportar hover com tooltip (nome, UF, risco, notificacoes, P(Surto))

## Non-Goals (Out of Scope)

- Nao migrar para React/Next.js separado — usar Streamlit + componentes React injetados
- Nao implementar autenticacao ou controle de acesso
- Nao fazer deploy em servidor (rodar local)
- Nao criar backend/API separado
- Nao retreinar modelos XGBoost — usar os ja treinados em `models/`
- Nao modificar o pipeline de dados (scripts 40-55) — apenas consumir os outputs
- Nao criar app mobile
- Nao implementar notificacoes push ou email
- Nao alterar a landing page HTML (`docs/dashboard_v4.html`) — escopo separado

## Design Considerations

### Identidade Visual
- **Paleta primaria:** Indigo-600 (#6366f1) para elementos primarios e interativos
- **Paleta de risco:** Vermelho (#ef4444) Surto, Laranja (#f97316) Alto, Amarelo (#eab308) Moderado, Verde (#22c55e) Baixo
- **Neutrals:** Slate scale — texto #0f172a, subtexto #64748b, muted #94a3b8, bordas #e2e8f0, bg #f8fafc
- **Fundo:** Pagina #f8fafc, cards #ffffff, sidebar #ffffff com borda
- **Sombras:** Uso sutil — hover cards: `0 4px 20px rgba(0,0,0,.04)`

### Tipografia
- **Font family:** DM Sans (Google Fonts)
- **Hierarchy:** KPI numbers 2rem/800, headings .82rem/700 uppercase, body .8rem/400, captions .68rem/500 uppercase

### Componentes
- **Cards:** radius 16px, border 1px #f1f5f9, padding 1.2rem, hover shadow
- **Badges/Pills:** radius 8px, background pastel, font .68rem bold
- **Tabs:** Underline style, cor ativa indigo-600, sem background
- **Section titles:** Uppercase, letter-spacing .08em, cor #94a3b8, margin-top 2rem

### Animacoes
- **Count-up:** JS IntersectionObserver + requestAnimationFrame, 1.2s ease-out
- **Fade-in:** CSS `@keyframes fadeIn { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:none } }`, 400ms
- **Hover:** `transition: all .2s ease` em cards
- **Skeleton:** `@keyframes pulse { 0%,100% { opacity:.4 } 50% { opacity:.8 } }`, background #e2e8f0
- **Graficos Plotly:** `transition_duration=800` onde suportado

### Inspiracoes
- Nixtio: espacamento generoso, tipografia bold, cards limpos
- Flatstudio: paleta restrita, hierarquia visual forte, microinteracoes
- Framer: motion design, transicoes suaves, attention to detail

## Technical Considerations

### Stack
- **Framework:** Streamlit (Python)
- **Graficos:** Plotly (go + px) com layout padronizado
- **Mapa:** Plotly choropleth com GeoJSON IBGE das mesorregioes
- **Animacoes:** CSS animations + JavaScript via `st.components.v1.html()`
- **PDF:** `fpdf2` ou `reportlab` (pip install)
- **Dados:** Parquet files em `data/model_ready_v4/splits_{a,b,c}/`
- **Modelos:** XGBoost .ubj em `models/regression_v4/` e `models/classification_v4/`

### Performance
- 180k rows carregados via `@st.cache_data` — cached entre re-runs
- Modelos XGBoost cached via `@st.cache_resource`
- Previsoes cached com hash dos parametros (var, sub)
- Mapa GeoJSON loaded once via cache
- Scatter plot amostrado (max 5000 pontos) para performance de rendering
- PDF gerado on-demand (nao pre-computado)

### GeoJSON Mesorregioes
- Fonte: IBGE Malhas Geograficas — GeoJSON das mesorregioes brasileiras
- Match via campo `cod_mesorregiao` (codigo IBGE de 4 digitos)
- Arquivo deve conter 137 geometrias correspondentes aos dados do modelo
- Pode ser baixado e armazenado localmente em `data/geojson/` ou carregado de URL publica

### Arquivos Existentes
- `dashboard/app.py` — arquivo principal (reescrever)
- `dashboard/.streamlit/config.toml` — configuracao de tema (ja criado)
- `data/model_ready_v4/splits_{a,b,c}/{train,val,test}.parquet` — dados
- `models/regression_v4/model_{A,B,C}_{sub}.ubj` — modelos regressao
- `models/classification_v4/model_{A,B,C}_{sub}_cls.ubj` — modelos classificacao
- `models/regression_v4/shap_importance_v4.csv` — SHAP (colunas: feature, shap_mean_abs_A, shap_mean_abs_B)
- `data/model_ready_v3/municipio_mesorregiao.csv` — nomes das mesorregioes
- `data/model_ready_v4/46_regression_v4_report.json` — report regressao
- `data/model_ready_v4/47_walkforward_v4_report.json` — report walk-forward
- `data/model_ready_v4/54_classification_v4_report.json` — report classificacao

## Success Metrics

- Dashboard carrega completamente em < 5 segundos (com cache quente)
- Mapa renderiza em < 3 segundos
- Todas as 5 tabs (Alertas, Regiao, Temporal, Modelo, Sobre) renderizam sem erro
- Zero mistura de temas dark/light — visual coeso em qualquer browser
- Filtro de Ano mostra todos os 26 anos (2001-2026)
- PDF exportado contem KPIs, tabela de alertas e periodo — legivel em A4
- Animacoes count-up e fade-in executam sem stutter
- Layout nao quebra em telas de 1024px a 1920px

## Open Questions

- Qual URL/fonte exata do GeoJSON de mesorregioes do IBGE usar? (precisa match exato com cod_mesorregiao dos dados)
- Usar Mapbox (requer token) ou mapa Plotly nativo (px.choropleth com geojson inline)?
- O PDF deve incluir graficos como imagens ou apenas dados tabulares?
- Incluir comparativo entre variantes na tab Sobre ou manter separado na tab Modelo?

# PRD: DengueCast — Redesign Premium Completo

## Introdução

O DengueCast é um dashboard de previsão de surtos de dengue (TCC em Engenharia de Software, UnB) construído com Streamlit, XGBoost e dados SINAN+INMET. O sistema está funcional — 5 tabs, mapa choropleth, gráficos Plotly, sidebar com filtros, export PDF — mas o visual é genérico e inconsistente. Este PRD define um redesign premium completo: novo design system, reorganização da navegação, tipografia refinada, micro-interações, e uma experiência que impressiona tanto a banca acadêmica quanto gestores de saúde pública.

**Problema:** O dashboard atual parece "projeto de disciplina" — funciona mas não transmite a sofisticação do trabalho por trás (18 modelos XGBoost, 180k registros, pipeline SINAN+INMET). O visual precisa estar à altura do conteúdo.

**Público duplo:**
- **Banca acadêmica:** Professores avaliadores que precisam ver rigor técnico, métricas claras, metodologia sólida
- **Gestores de saúde:** Profissionais de vigilância epidemiológica que precisam de interface intuitiva, linguagem acessível, dados acionáveis

## Objetivos

- Criar identidade visual premium e coesa para o DengueCast
- Reorganizar navegação de 5 tabs para 4 tabs mais lógicas
- Estabelecer design system consistente (cores, tipografia, espaçamento, componentes)
- Migrar página "Sobre" de iframe HTML para componentes Streamlit nativos
- Melhorar hierarquia visual de todas as seções com cards, badges e layout refinado
- Adicionar micro-interações significativas (hover states, transições suaves)
- Refinar todos os gráficos Plotly (tooltips, cores, padding, responsividade)
- Garantir que a interface funcione bem em telas de 1280px+ (laptop/desktop)

## User Stories

### US-001: Design System — Variáveis CSS e Tipografia
**Descrição:** Como desenvolvedor, preciso de um design system CSS completo e consistente para que todos os componentes sigam o mesmo padrão visual.

**Acceptance Criteria:**
- [ ] Variáveis CSS em `:root` cobrindo: cores (primary, accent, success, warning, danger, neutral scale), tipografia (heading, body, caption sizes), espaçamento (spacing scale 4px base), sombras (sm, md, lg, xl), border-radius (sm, md, lg), transitions
- [ ] Font stack atualizado: headings com peso 800, body com 400/500, captions com 600 uppercase
- [ ] Remover TODOS os estilos inline (`style="..."`) dos `st.markdown()` — migrar para classes CSS reutilizáveis
- [ ] Hierarquia tipográfica clara: h1 (page title), h2 (section header), h3 (card title), body, caption, label
- [ ] `.block-container` com max-width 1440px e padding consistente
- [ ] Verificar no browser via dev server

### US-002: Reorganização da Navegação (5→4 tabs)
**Descrição:** Como usuário, quero uma navegação mais limpa e lógica para encontrar o que preciso mais rápido.

**Acceptance Criteria:**
- [ ] Tabs reorganizadas: **Painel** (atual Alertas), **Regiões** (atual Região), **Análise** (merge Temporal + Modelo), **Sobre** (reescrita nativa)
- [ ] Tab "Painel" é a home page — visão geral com KPIs, mapa, filtros e evolução
- [ ] Tab "Análise" tem sub-seções internas via `sec()` headers: "Previsão vs Realidade", "Evolução Nacional", "Desempenho por Ano", "Métricas do Modelo", "Comparativo de Variantes", "Walk-Forward", "SHAP"
- [ ] Ordem das sub-seções em Análise: primeiro o conteúdo acessível (gráficos visuais), depois o técnico (métricas, SHAP)
- [ ] Renomear labels dos tabs para nomes curtos e claros
- [ ] Verificar no browser via dev server

### US-003: Header Premium
**Descrição:** Como usuário, quero um header que transmita a identidade do DengueCast e me situe no contexto do sistema.

**Acceptance Criteria:**
- [ ] Header com logo à esquerda + nome "DengueCast" em tipografia bold + tagline "Sistema preditivo de vigilância epidemiológica"
- [ ] Indicador visual sutil do modo de dados atual (ex: "Série completa · Clima detalhado" como chip/badge discreto no header)
- [ ] Remover o layout de 2 columns atual (hdr_l/hdr_r) — usar layout full-width com flex
- [ ] Separador visual sutil entre header e conteúdo (linha 1px ou sombra)
- [ ] Header não ocupa mais que 56px de altura
- [ ] Verificar no browser via dev server

### US-004: Sidebar Redesign Premium
**Descrição:** Como usuário, quero uma sidebar organizada e elegante que me permita configurar os parâmetros de análise facilmente.

**Acceptance Criteria:**
- [ ] Seção "Configuração do Modelo" com label uppercase e ícone
- [ ] Selectboxes com labels descritivos e tooltips (usar `help` param do st.selectbox)
- [ ] Caption abaixo de cada select mostrando resumo da seleção (já existe, melhorar visual)
- [ ] Separador visual entre seção de modelo e seção de filtros
- [ ] Seção "Filtros" com Ano e UF lado a lado (já existe, melhorar visual)
- [ ] Stats cards com design glass-morphism sutil: background semi-transparente, border 1px, blur
- [ ] Stats cards mostrando: total de previsões, período coberto, cobertura nacional
- [ ] Footer discreto na sidebar com "TCC · UnB FGA · 2024–2025" em texto muted
- [ ] Verificar no browser via dev server

### US-005: KPI Cards Premium com Interação
**Descrição:** Como gestor de saúde, quero que os indicadores de alerta sejam visualmente impactantes e fáceis de ler num relance.

**Acceptance Criteria:**
- [ ] KPI cards com CSS classes reutilizáveis (não inline styles)
- [ ] Hover effect: elevação sutil (box-shadow muda), scale 1.01
- [ ] Barra de acento no topo de cada card (3px) com a cor do nível de risco
- [ ] Background: tint sutil da cor de risco (10% opacity)
- [ ] Valores numéricos com fonte 2.2rem weight 800, formatação com separador de milhar
- [ ] Labels uppercase .62rem weight 700 com letter-spacing
- [ ] Transição CSS suave em todos os hover states (0.2s ease)
- [ ] Animação fadeInUp escalonada por card (delay 60ms entre cards)
- [ ] Verificar no browser via dev server

### US-006: Tab Painel — Layout e Seções Refinadas
**Descrição:** Como usuário, quero que a visão geral de alertas seja organizada com hierarquia visual clara e seções bem definidas.

**Acceptance Criteria:**
- [ ] KPIs no topo (US-005)
- [ ] Filtro de risco (radio pills) abaixo dos KPIs com design refinado
- [ ] Seção "Mapa de Risco" com card wrapper (background white, border, radius 14px, shadow-sm)
- [ ] Mapa choropleth dentro do card com padding interno
- [ ] Seção "Risco por UF" e "Top Alertas" lado a lado em cards separados
- [ ] Bar chart de UF: cores do gradiente de risco, hover com tooltip formatado
- [ ] Tabela de top alertas: column_config com ProgressColumn para P(Surto), NumberColumn para Prev
- [ ] Seção "Evolução Semanal" com card wrapper
- [ ] Botão de download PDF estilizado e posicionado no canto superior direito da seção de alertas
- [ ] Espaçamento consistente entre seções: 1.5rem gap
- [ ] Verificar no browser via dev server

### US-007: Tab Regiões — Selected State Premium
**Descrição:** Como gestor de saúde, quero que ao selecionar uma mesorregião, os dados apareçam de forma organizada e visualmente rica.

**Acceptance Criteria:**
- [ ] Empty state mantido (cards de top risco) mas com layout grid 4-col
- [ ] Header card da região selecionada: nome em bold, badge de risco colorido, período atual, layout flex
- [ ] Métricas em 4 columns com design de card individual (não st.metric genérico)
- [ ] Card "Casos atuais" com ícone e valor grande
- [ ] Card "Previsão 4 semanas" com destaque visual (accent color)
- [ ] Card "Nível de risco" com badge colorido grande
- [ ] Card "Chance de surto" com indicador visual (barra de progresso ou gauge)
- [ ] Gráficos de série temporal e probabilidade dentro de cards wrapper
- [ ] Seção "Desempenho do modelo nesta região" com métricas em mini-cards
- [ ] Verificar no browser via dev server

### US-008: Tab Análise — Merge Temporal + Modelo
**Descrição:** Como membro da banca, quero ver toda a análise de desempenho do modelo numa tab organizada com fluxo lógico.

**Acceptance Criteria:**
- [ ] Intro text explicando o propósito da tab (1-2 frases, tom acessível)
- [ ] Sub-seção "Previsão vs Realidade" — scatter plot com explicação contextual
- [ ] Sub-seção "Evolução Nacional" — line chart real vs previsto com explicação
- [ ] Sub-seção "Desempenho por Ano" — bar chart R² por ano com cores treino/validação/teste
- [ ] Separador visual (divider) entre bloco temporal e bloco técnico
- [ ] Sub-seção "Métricas do Modelo Selecionado" — cards com R², RMSE, MAE, IC 95% (só aparece se report existe)
- [ ] Sub-seção "Comparativo de Variantes" — bar chart + tabela de classificação lado a lado
- [ ] Sub-seção "Validação Walk-Forward" — line chart com explicação didática
- [ ] Sub-seção "Importância das Variáveis (SHAP)" — bar chart horizontal com slider e nomes traduzidos
- [ ] Cada sub-seção com header `sec()` estilizado
- [ ] Textos explicativos em linguagem acessível antes de cada gráfico (para gestores de saúde)
- [ ] Verificar no browser via dev server

### US-009: Tab Sobre — Reescrita Nativa Streamlit
**Descrição:** Como visitante, quero que a página Sobre esteja integrada visualmente com o resto do dashboard e carregue instantaneamente.

**Acceptance Criteria:**
- [ ] Remover `st.components.v1.html(about_html...)` e toda a variável `about_html` (~260 linhas)
- [ ] Hero section usando `st.markdown()` com CSS classes: título grande, subtítulo, badges de stats (registros, período, mesorregiões, modelos)
- [ ] Seção "Pipeline de Dados" com steps visualizados (SINAN → Limpeza → Feature Eng. → INMET → Merge → Splits → XGBoost)
- [ ] Seção "Variantes Experimentais" como tabela Streamlit nativa
- [ ] Seção "Sub-modelos de Features" como tabela Streamlit nativa
- [ ] Seção "Resultados" com highlight card (background gradient sutil) mostrando achado principal
- [ ] Seção "Classificação de Risco" com badges coloridos e mini-metrics
- [ ] Seção "Validação" como lista com checkmarks
- [ ] Seção "Stack Tecnológico" com chips/badges
- [ ] Seção "Limitações" com callout warning style
- [ ] Seção "Autores" com card de crédito: Pedro Lucas e Thiago, Eng. Software UnB FGA 2024–2025
- [ ] Todas as seções usando componentes `st.markdown()` com CSS classes do design system
- [ ] Verificar no browser via dev server

### US-010: Gráficos Plotly — Polish Global
**Descrição:** Como usuário, quero que todos os gráficos tenham visual consistente, profissional e agradável.

**Acceptance Criteria:**
- [ ] Função `plo()` atualizada: font sizes consistentes, margins otimizadas, axis styling refinado
- [ ] Todos os gráficos: gridlines `#f2f4f7`, sem gridlines no eixo X, title="" em ambos eixos (títulos via sec())
- [ ] Hover templates formatados em TODOS os gráficos com unidades e labels em português
- [ ] Scatter plot (Temporal): opacity ajustada por densidade, marker size 5px
- [ ] Bar charts: border-radius nas barras (se suportado), hover highlight
- [ ] Line charts: width 2.5px para primárias, 1.5px para secundárias, fill com alpha 0.03
- [ ] Color palette consistente: primary (#5046e5) para dados do modelo, risk colors para classificação
- [ ] Legendas posicionadas consistentemente (topo, horizontal)
- [ ] Remover modebar de todos os gráficos (já feito, confirmar)
- [ ] Verificar no browser via dev server

### US-011: Micro-interações e Animações
**Descrição:** Como usuário, quero que a interface tenha transições suaves que tornem a navegação fluida.

**Acceptance Criteria:**
- [ ] fadeInUp nos KPI cards com stagger (já existe, refinar timing)
- [ ] fadeInUp nos cards de seção com delay maior que KPIs
- [ ] Hover em cards: transform scale(1.005), box-shadow aumenta, transition 0.2s ease
- [ ] Hover em risk filter pills: background transition suave
- [ ] Hover na tabela de top alertas: row highlight
- [ ] Tabs: transição de cor/background no switch (já parcialmente existe via CSS)
- [ ] Sidebar selectbox: focus ring com primary color (já existe, confirmar)
- [ ] Não adicionar animações pesadas que afetem performance
- [ ] Verificar no browser via dev server

### US-012: Card Wrapper Component
**Descrição:** Como desenvolvedor, preciso de um componente de card reutilizável para encapsular seções com estilo consistente.

**Acceptance Criteria:**
- [ ] Função `card(title=None)` que retorna context manager: `with card("Mapa de Risco"):`
- [ ] Alternativa: par `card_start(title)`/`card_end()` mantendo pattern atual
- [ ] Card: background #fff, border 1px #e4e7ec, radius 14px, shadow-sm, padding 1.2rem 1.5rem
- [ ] Card com título: header com sec() style integrado
- [ ] Card sem título: apenas wrapper visual
- [ ] Hover: shadow muda de sm para md (transition 0.2s)
- [ ] Usar em todas as seções de conteúdo (mapa, gráficos, tabelas)
- [ ] Verificar no browser via dev server

### US-013: Responsividade Desktop
**Descrição:** Como usuário acessando de diferentes tamanhos de tela, quero que o layout funcione de 1280px a 1920px+.

**Acceptance Criteria:**
- [ ] Max-width 1440px no `.block-container` com auto margins (centralizado)
- [ ] Grid de KPI cards: 4 colunas em telas 1280px+, stack responsivo se Streamlit suportar
- [ ] Gráficos lado a lado (UF + Top alertas): 50/50 split
- [ ] Mapa: use_container_width=True com height responsivo
- [ ] Sidebar: não definir width fixo, deixar Streamlit gerenciar
- [ ] Textos com max-width 680px para legibilidade (parágrafos explicativos)
- [ ] Verificar no browser via dev server em 1280px e 1440px+

## Requisitos Funcionais

- FR-01: Reorganizar tabs de `["Alertas", "Regiao", "Temporal", "Modelo", "Sobre"]` para `["Painel", "Regiões", "Análise", "Sobre"]`
- FR-02: Mover TODO o conteúdo de Temporal e Modelo para dentro da tab Análise, com seções internas
- FR-03: Reescrever a página Sobre usando componentes Streamlit nativos (eliminar about_html e st.components.v1.html)
- FR-04: Migrar todos os estilos inline para classes CSS no design system
- FR-05: Manter funcionalidade de export PDF sem alterações
- FR-06: Manter funcionalidade de filtros (Ano, UF, risco) sem alterações na lógica
- FR-07: Manter compatibilidade com modo cache (pre-computed parquets) e modo live (XGBoost)
- FR-08: Todos os textos explicativos devem ser em português acessível (sem jargão técnico para gestores, com precisão para a banca)
- FR-09: Section headers devem usar o componente `sec()` (barra accent + texto uppercase)
- FR-10: Gráficos Plotly devem usar a função `plo()` para layout consistente
- FR-11: Hover templates em português em todos os gráficos
- FR-12: Card wrapper para todas as seções de conteúdo visual

## Não-Objetivos (Fora de Escopo)

- **Dark mode** — foi removido na sessão anterior e não será reimplementado
- **Mobile/tablet** — foco é desktop 1280px+. Streamlit não é mobile-first
- **Novos gráficos ou visualizações** — redesign do existente, não adição de novos
- **Mudanças na lógica de predição** — apenas visual, não tocar em run_predictions(), load_models(), etc.
- **Mudanças no PDF export** — manter make_pdf() como está
- **Internacionalização** — dashboard é em português apenas
- **Autenticação ou controle de acesso** — dashboard é público
- **Novas fontes** — manter DM Sans (já é boa escolha)
- **Streamlit multipage** — manter tudo em app.py com tabs

## Considerações de Design

### Estética
- **Tom:** Profissional-institucional com toques modernos. Não é um app consumer, é uma ferramenta de vigilância epidemiológica + projeto acadêmico
- **Paleta:** Indigo (#5046e5) como primary, com neutral scale (stone/slate). Risk colors já definidas (green/yellow/orange/red)
- **Tipografia:** DM Sans — manter. Refinar pesos: 800 para headings, 600 para labels, 400/500 para body
- **Densidade:** Média. Nem muito espaçado (desperdiça tela) nem comprimido (confuso). Profissional
- **Cards:** White cards com border sutil e shadow-sm como unidade visual principal

### Referências Visuais
- Dashboards de saúde pública (OMS, ECDC)
- Linear.app (tabs, tipografia, cards)
- Vercel dashboard (clean, tipografia bold, hierarquia visual)

## Considerações Técnicas

- **Streamlit 1.50.0** — todas as features devem ser compatíveis
- **st.logo()** — necessário para sidebar collapse funcionar (não remover)
- **CSS via st.markdown(unsafe_allow_html=True)** — principal mecanismo de estilização
- **Plotly** — para gráficos interativos. Configuração via `plo()` helper
- **Sem JavaScript** — apenas CSS para animações/interações
- **Performance** — animações CSS-only, sem bibliotecas externas pesadas
- **Deploy** — Streamlit Cloud (modelo-previsao-ffqc3ugltpbxxeaa4mygvw.streamlit.app)
- **Arquivo único** — manter tudo em `dashboard/app.py` (~1300 linhas, aceitável)

## Métricas de Sucesso

- Dashboard transmite profissionalismo ao abrir — "wow factor" nos primeiros 3 segundos
- Gestores de saúde conseguem entender os alertas sem explicação adicional
- Banca acadêmica consegue encontrar métricas de performance em menos de 2 cliques
- Todas as seções têm hierarquia visual clara (header → conteúdo → detalhe)
- Zero estilos inline remanescentes — tudo via CSS classes
- Visual consistente entre todas as 4 tabs

## Questões em Aberto

- Nomes finais das tabs: "Painel" vs "Visão Geral" vs "Alertas" (para a tab principal)
- Se "Análise" deve ter sub-tabs internas (via radio/selectbox) ou scroll contínuo com seções
- Se vale adicionar um "badge de status" no header mostrando modelo/dados selecionados
- Se o hero da página Sobre deve manter as estatísticas inline (180k registros, 137 mesorregiões, etc.)

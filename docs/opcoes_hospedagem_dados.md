# Opcoes de Hospedagem dos Dados - TCC2

## Contexto

Precisamos hospedar ~2.1GB de dados (parquets) que nao cabem no GitHub.
O codigo/scripts/modelos ja estao no GitHub (repo MODELO-PREVISAO).
Precisamos de um lugar para os dados brutos e processados das pipelines SINAN e INMET.

Dados a hospedar:
- INMET Bronze (hourly): 1.0 GB — dados horarios brutos (~73M registros)
- SINAN Gold (dense): 336 MB — serie densa com features (~7.7M registros)
- Model Ready (splits): 304 MB — train/val/test prontos pro XGBoost
- Integrado (join): 270 MB — SINAN+INMET unificado
- INMET Gold (municipal): 146 MB — features climaticas municipais
- SINAN Silver: 57 MB — notificacoes municipais semanais
- INMET Silver: 17 MB — agregados semanais por estacao
- Resto (governance, reference): ~7 MB

---

## Opcao 1: Google Drive

**Como funciona:** Sobe a pasta data/ inteira no Drive. Compartilha o link no README do GitHub.

| Item | Detalhe |
|------|---------|
| Custo | Gratis (15 GB por conta, 100 GB com conta UnB) |
| Limite de arquivo | 5 TB por arquivo |
| Acesso | Qualquer pessoa com o link |
| Reprodutibilidade | Download manual ou via gdown no script |
| Versionamento | Nao tem |
| Permanencia | Depende da conta (Google pode desativar contas inativas) |
| Citabilidade | Nao gera DOI |

**Pros:**
- Zero fricção, todo mundo conhece
- Orientador abre pelo navegador
- Integra com Google Colab (montar Drive e rodar notebooks)
- Pode organizar em pastas espelhando a estrutura do projeto

**Contras:**
- Sem versionamento dos dados
- Se a conta for desativada, os dados somem
- Nao tem peso academico (sem DOI)
- Download programatico exige gdown ou API

**Complexidade:** Muito baixa (arrastar e soltar)

---

## Opcao 2: Kaggle Datasets

**Como funciona:** Cria um Dataset no Kaggle com os parquets organizados. O Kaggle hospeda e disponibiliza para download.

| Item | Detalhe |
|------|---------|
| Custo | Gratis |
| Limite | 100 GB por dataset |
| Acesso | Publico ou privado |
| Reprodutibilidade | Excelente — Kaggle Notebooks rodam direto sobre o dataset |
| Versionamento | Sim (versoes do dataset com diff) |
| Permanencia | Alta (plataforma consolidada, Google) |
| Citabilidade | Gera URL estavel, mas nao DOI |

**Pros:**
- Versionamento nativo dos dados
- Kaggle Notebooks permitem reproduzir analises sem instalar nada
- Comunidade academica e de ML usa bastante
- API de download: `kaggle datasets download -d usuario/dataset`
- Boa visibilidade (aparece em buscas, pode virar referencia)

**Contras:**
- Precisa de conta Kaggle (gratuita)
- Interface pode ser confusa pra quem nunca usou
- Nao gera DOI formal

**Complexidade:** Baixa (upload via web ou CLI)

---

## Opcao 3: Hugging Face Datasets

**Como funciona:** Cria um repositorio de dataset no Hugging Face Hub. Funciona como um GitHub para dados de ML.

| Item | Detalhe |
|------|---------|
| Custo | Gratis ate 50 GB |
| Limite de arquivo | 50 GB por arquivo (com Git LFS) |
| Acesso | Publico ou privado |
| Reprodutibilidade | Otima — `datasets` library carrega direto em Python |
| Versionamento | Sim (Git nativo, cada push eh uma versao) |
| Permanencia | Alta (plataforma em crescimento, referencia em ML) |
| Citabilidade | URL estavel, DOI via Zenodo integration |

**Pros:**
- Versionamento completo (eh Git por baixo)
- Pode hospedar dados + modelo juntos (Model Hub)
- `pip install datasets` e `load_dataset("usuario/dengue-tcc2")` carrega tudo
- Dataset cards (README com metadados, como um paper)
- Moderno, muito usado em papers recentes de ML
- Parquet eh formato nativo do HF Datasets

**Contras:**
- Menos conhecido que Kaggle no Brasil
- Curva de aprendizado se nunca usou o Hub
- Orientador pode nao conhecer

**Complexidade:** Media (precisa configurar o repo e fazer push via CLI)

---

## Opcao 4: Zenodo (CERN)

**Como funciona:** Faz upload dos dados como um "record" no Zenodo. Gera DOI automaticamente.

| Item | Detalhe |
|------|---------|
| Custo | Gratis |
| Limite | 50 GB por record |
| Acesso | Publico (open access) ou restrito |
| Reprodutibilidade | Download via URL estavel ou API |
| Versionamento | Sim (versoes com DOIs separados) |
| Permanencia | Muito alta (hospedado no CERN, arquivamento de longo prazo) |
| Citabilidade | **Gera DOI oficial** — citavel em papers e TCC |

**Pros:**
- DOI oficial — pode ser citado no TCC como fonte de dados
- Arquivamento de longo prazo (o CERN nao vai fechar)
- Aceito por revistas cientificas como repositorio de dados
- Integra com GitHub (pode criar release automatica)
- Metadados ricos (autores, licenca, descricao, keywords)

**Contras:**
- Interface antiquada
- Upload pode ser lento
- Uma vez publicado com DOI, nao pode deletar (pode criar nova versao)
- Nao tem notebooks integrados

**Complexidade:** Media (preencher metadados, upload via web ou API)

---

## Opcao 5: GitHub Releases

**Como funciona:** Cria uma Release no repo MODELO-PREVISAO e anexa os parquets como assets (zipados).

| Item | Detalhe |
|------|---------|
| Custo | Gratis |
| Limite | 2 GB por arquivo, 100+ arquivos por release |
| Acesso | Publico (mesmo repo) |
| Reprodutibilidade | Download via `gh release download` ou URL direta |
| Versionamento | Sim (cada release eh uma versao) |
| Permanencia | Mesma do repo GitHub |
| Citabilidade | URL estavel, sem DOI |

**Pros:**
- Tudo no mesmo lugar (codigo + dados no mesmo repo)
- `gh release download v1.0` baixa tudo
- Sem ferramenta nova — quem tem GitHub ja tem acesso
- Pode automatizar: script que baixa os assets e descompacta na pasta data/

**Contras:**
- Limite de 2 GB por arquivo (precisa zipar/splittar o inmet bronze)
- Os dados ficam "escondidos" na aba Releases (nao eh obvio)
- Nao gera DOI

**Complexidade:** Baixa (um comando `gh release create`)

---

## Opcao 6: Git LFS (Large File Storage)

**Como funciona:** Configura o repo GitHub para versionar arquivos grandes via LFS. Os parquets ficam no proprio repo.

| Item | Detalhe |
|------|---------|
| Custo | Gratis ate 1 GB de storage + 1 GB/mes de bandwidth. Depois: $5/mes por 50 GB |
| Limite de arquivo | 2 GB por arquivo |
| Acesso | Mesmo do repo |
| Reprodutibilidade | Excelente — `git clone` ja traz tudo |
| Versionamento | Sim (Git nativo) |
| Permanencia | Mesma do repo |
| Citabilidade | Nao |

**Pros:**
- Experiencia transparente: `git clone` e pronto
- Dados versionados junto com o codigo
- Nao precisa de ferramenta extra

**Contras:**
- **Custa dinheiro** apos 1 GB (nossos dados tem 2.1 GB)
- Bandwidth limitado (cada clone consome cota)
- Se parar de pagar, os dados ficam inacessiveis
- Limite de 2 GB por arquivo

**Complexidade:** Baixa-media (configurar .gitattributes e LFS tracking)

---

## Opcao 7: Cloudflare R2

**Como funciona:** Object storage compativel com S3. Sobe os parquets num bucket e serve via URL publica.

| Item | Detalhe |
|------|---------|
| Custo | Gratis ate 10 GB storage + 10M requests/mes |
| Limite de arquivo | 5 TB |
| Acesso | URL publica ou autenticada |
| Reprodutibilidade | Download via curl/wget ou boto3 |
| Versionamento | Nao nativo |
| Permanencia | Depende da conta Cloudflare |
| Citabilidade | Nao |

**Pros:**
- Gratis pra nosso volume (2.1 GB << 10 GB)
- Sem custo de egress (diferente da AWS S3)
- URLs publicas diretas para cada arquivo
- Profissional (infra de producao real)

**Contras:**
- Precisa de conta Cloudflare
- Configuracao mais tecnica (bucket, permissoes, CORS)
- Nao tem peso academico
- Overkill pra TCC

**Complexidade:** Media-alta

---

## Comparacao Resumida

| Opcao | Custo | DOI | Versionamento | Facilidade | Academico | Reproducao |
|-------|-------|-----|---------------|------------|-----------|------------|
| Google Drive | Gratis | Nao | Nao | Muito facil | Fraco | Manual |
| Kaggle Datasets | Gratis | Nao | Sim | Facil | Bom | Notebooks |
| Hugging Face | Gratis | Via Zenodo | Sim | Medio | Bom | `load_dataset()` |
| **Zenodo** | **Gratis** | **Sim** | **Sim** | **Medio** | **Excelente** | **URL estavel** |
| GitHub Releases | Gratis | Nao | Sim | Facil | Fraco | `gh download` |
| Git LFS | Pago (>1GB) | Nao | Sim | Facil | Fraco | `git clone` |
| Cloudflare R2 | Gratis | Nao | Nao | Dificil | Fraco | URLs diretas |

---

## Recomendacao

**Para TCC academico, as melhores opcoes sao:**

1. **Zenodo** — se o orientador valoriza citabilidade. O DOI da peso ao trabalho e fica permanente.

2. **Kaggle Datasets** — se querem visibilidade e facilidade de reproducao. Notebooks integrados impressionam na defesa.

3. **GitHub Releases** — se querem simplicidade maxima e tudo num lugar so.

**Combinacao ideal:** Zenodo (dados oficiais com DOI) + Google Drive (acesso rapido pro dia a dia e Colab).

---

## Proximo passo

Decidam qual opcao preferem e eu configuro tudo:
- Organizo os dados na estrutura certa
- Faco o upload
- Atualizo o README do MODELO-PREVISAO com o link
- Crio um script de download automatico se necessario

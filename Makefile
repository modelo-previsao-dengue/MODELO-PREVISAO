.PHONY: publish publish-hf publish-hf-v4 publish-hf-dashboard publish-kaggle \
       publish-dry validate validate-remote status help

PYTHON ?= python3
HF_SCRIPT = scripts/upload_data_to_hf.py
KAGGLE_SCRIPT = scripts/publish_kaggle.py
VALIDATE_SCRIPT = scripts/validate_hf_publication.py
VALIDATE_LOCAL = scripts/validate_local.py

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

publish: validate publish-hf publish-kaggle ## Valida + sobe HF + sobe Kaggle
	@echo ""
	@echo "=== Publicação completa finalizada ==="
	@echo "  HF: https://huggingface.co/datasets/pedrolucassantanaf/dengue-tcc2-data"
	@echo "  Kaggle: https://www.kaggle.com/pedrolucassantanaf"

publish-hf: ## Sobe tudo para HuggingFace (~1.1 GB)
	$(PYTHON) $(HF_SCRIPT) --tudo

publish-hf-v4: ## Sobe apenas dados + modelos v4 (~786 MB)
	$(PYTHON) $(HF_SCRIPT) --v4

publish-hf-dashboard: ## Sobe apenas cache da dashboard (~43 MB)
	$(PYTHON) $(HF_SCRIPT) --dashboard

publish-kaggle: ## Publica notebooks no Kaggle
	$(PYTHON) $(KAGGLE_SCRIPT)

publish-dry: ## Dry-run de tudo (HF + Kaggle)
	$(PYTHON) $(HF_SCRIPT) --dry-run --tudo
	@echo ""
	$(PYTHON) $(KAGGLE_SCRIPT) --dry-run

validate: ## Valida integridade dos artefatos locais
	@echo "=== Validando artefatos locais ==="
	@$(PYTHON) $(VALIDATE_LOCAL)

validate-remote: ## Valida publicação remota no HuggingFace
	$(PYTHON) $(VALIDATE_SCRIPT)

status: ## Mostra o que está local vs publicado
	@echo "=== Status local ==="
	@echo "  Dados v4:      $$(du -sh data/model_ready_v4/ 2>/dev/null | cut -f1)"
	@echo "  Modelos reg:   $$(du -sh models/regression_v4/ 2>/dev/null | cut -f1)"
	@echo "  Modelos cls:   $$(du -sh models/classification_v4/ 2>/dev/null | cut -f1)"
	@echo "  Dashboard:     $$(du -sh dashboard/data/ 2>/dev/null | cut -f1)"
	@echo "  Dados antigos: $$(du -sh data/model_ready/ 2>/dev/null | cut -f1)"
	@echo ""
	@echo "=== Status HuggingFace ==="
	@$(PYTHON) scripts/status_hf.py 2>&1 || echo "  Erro ao conectar ao HuggingFace"

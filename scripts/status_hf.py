"""Mostra status dos artefatos no HuggingFace."""
from huggingface_hub import HfApi

REPO = "pedrolucassantanaf/dengue-tcc2-data"
api = HfApi()
files = [f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
         if hasattr(f, "size")]
total = sum(f.size for f in files)
v4 = [f for f in files if "model_ready_v4" in f.path]
models = [f for f in files if f.path.startswith("models/") and f.path.endswith(".ubj")]
preds = [f for f in files if "predictions_" in f.path]

print(f"  Total: {len(files)} arquivos ({total/1024**2:.0f} MB)")
print(f"  Dados v4: {len(v4)} arquivos")
print(f"  Modelos: {len(models)} .ubj")
print(f"  Predictions: {len(preds)} arquivos")
print(f"  URL: https://huggingface.co/datasets/{REPO}")

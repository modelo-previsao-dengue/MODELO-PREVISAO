"""Publica/atualiza notebooks no Kaggle via API.

Uso:
    python3 scripts/publish_kaggle.py                # publica todos
    python3 scripts/publish_kaggle.py --notebook v4_01  # publica apenas um
    python3 scripts/publish_kaggle.py --dry-run       # mostra sem enviar
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT / "kaggle"

NOTEBOOKS = [
    {
        "id": "v4_01",
        "metadata": "v4_01_kernel-metadata.json",
        "notebook": "v4_01_exploracao.ipynb",
        "title": "DengueCast v4 — Exploração dos Dados",
        "url": "https://www.kaggle.com/pedrolucassantana/denguecast-v4-01-exploracao",
    },
    {
        "id": "v4_02",
        "metadata": "v4_02_kernel-metadata.json",
        "notebook": "v4_02_treino.ipynb",
        "title": "DengueCast v4 — Treino e Avaliação",
        "url": "https://www.kaggle.com/pedrolucassantana/denguecast-v4-02-treino",
    },
    {
        "id": "v4_03",
        "metadata": "v4_03_kernel-metadata.json",
        "notebook": "v4_03_resultados.ipynb",
        "title": "DengueCast v4 — Resultados e Visualizações",
        "url": "https://www.kaggle.com/pedrolucassantana/denguecast-v4-03-resultados",
    },
]


def check_kaggle_credentials():
    if os.environ.get("KAGGLE_API_TOKEN"):
        result = subprocess.run(
            ["kaggle", "config", "view"], capture_output=True, text=True
        )
        return os.environ.get("KAGGLE_USERNAME", "authenticated-via-token")

    access_token = Path.home() / ".kaggle" / "access_token"
    if access_token.exists():
        return "authenticated-via-access-token"

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if creds.exists():
        data = json.loads(creds.read_text())
        if "username" in data and "key" in data:
            return data["username"]

    print("ERRO: credenciais Kaggle não encontradas.")
    print("  Opção 1: export KAGGLE_API_TOKEN=KGAT_xxx")
    print("  Opção 2: ~/.kaggle/access_token")
    print("  Opção 3: ~/.kaggle/kaggle.json (legado)")
    sys.exit(1)


def publish_notebook(nb, dry_run=False):
    metadata_path = KAGGLE_DIR / nb["metadata"]
    notebook_path = KAGGLE_DIR / nb["notebook"]

    if not metadata_path.exists():
        print(f"  ERRO: {metadata_path} não encontrado")
        return False
    if not notebook_path.exists():
        print(f"  ERRO: {notebook_path} não encontrado")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Publicaria: {nb['title']}")
        print(f"    Metadata: {nb['metadata']}")
        print(f"    Notebook: {nb['notebook']}")
        print(f"    URL: {nb['url']}")
        return True

    # kaggle kernels push expects kernel-metadata.json in the same dir as the notebook
    # We need to temporarily rename the metadata file
    std_metadata = KAGGLE_DIR / "kernel-metadata.json"
    renamed = False
    if not std_metadata.exists():
        metadata_path.rename(std_metadata)
        renamed = True

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(f"  Publicado: {nb['title']}")
            print(f"  URL: {nb['url']}")
            return True
        else:
            err = result.stderr.strip() or result.stdout.strip()
            print(f"  ERRO ao publicar: {err}")
            return False
    except FileNotFoundError:
        print("  ERRO: comando 'kaggle' não encontrado. Instale: pip install kaggle")
        return False
    except subprocess.TimeoutExpired:
        print("  ERRO: timeout ao publicar")
        return False
    finally:
        if renamed and std_metadata.exists():
            std_metadata.rename(metadata_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", type=str, help="ID do notebook (v4_01, v4_02, v4_03)")
    ap.add_argument("--dry-run", action="store_true", help="Mostra sem enviar")
    args = ap.parse_args()

    username = check_kaggle_credentials()
    print(f"Kaggle user: {username}")

    if args.notebook:
        nbs = [nb for nb in NOTEBOOKS if nb["id"] == args.notebook]
        if not nbs:
            print(f"ERRO: notebook '{args.notebook}' não encontrado. Opções: {[n['id'] for n in NOTEBOOKS]}")
            sys.exit(1)
    else:
        nbs = NOTEBOOKS

    print(f"\nPublicando {len(nbs)} notebook(s){'  [DRY-RUN]' if args.dry_run else ''}:\n")

    ok = 0
    for nb in nbs:
        print(f"[{nb['id']}] {nb['title']}")
        if publish_notebook(nb, dry_run=args.dry_run):
            ok += 1
        print()

    print(f"\nResultado: {ok}/{len(nbs)} publicados com sucesso")
    if ok == len(nbs):
        print("\nURLs:")
        for nb in nbs:
            print(f"  {nb['url']}")


if __name__ == "__main__":
    main()

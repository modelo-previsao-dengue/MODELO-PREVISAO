#!/usr/bin/env python3
"""
Publica a versao oficial da trilha SINAN TCC2 em bucket compativel com S3/R2.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "sinan"
DEFAULT_VERSION = "sinan_tcc2_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publica artefatos da trilha SINAN TCC2 em R2/S3")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def build_client():
    try:
        import boto3
    except Exception as exc:
        raise SystemExit("boto3 nao instalado: " + str(exc))

    endpoint = require_env("R2_ENDPOINT_URL")
    access_key = require_env("R2_ACCESS_KEY_ID")
    secret_key = require_env("R2_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION", "auto")

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def iter_uploads(version: str):
    gold_root = DATA_ROOT / "gold" / version
    governance_root = DATA_ROOT / "governance" / version
    bronze_root = DATA_ROOT / "bronze" / version / "inventory"

    targets = [
        (gold_root / "official_dense", "official_dense"),
        (gold_root / "analytics", "analytics"),
        (governance_root, "governance"),
        (bronze_root, "bronze/inventory"),
    ]

    for source_root, target_prefix in targets:
        if not source_root.exists():
            continue
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                yield path, target_prefix, source_root


def main() -> None:
    args = parse_args()
    bucket = require_env("R2_BUCKET")
    version = args.version
    prefix = args.prefix or os.getenv("R2_PREFIX", f"sinan/tcc2/{version}")

    uploads = list(iter_uploads(version))
    if not uploads:
        raise SystemExit(f"Nenhum artefato encontrado para publicar na versao {version}")

    if args.dry_run:
        for path, target_prefix, source_root in uploads[:20]:
            rel = path.relative_to(source_root).as_posix()
            print(f"DRY {path} -> s3://{bucket}/{prefix}/{target_prefix}/{rel}")
        print(f"Total de arquivos para upload: {len(uploads)}")
        return

    client = build_client()
    manifest_rows = []
    for path, target_prefix, source_root in uploads:
        rel = path.relative_to(source_root).as_posix()
        key = f"{prefix}/{target_prefix}/{rel}"
        content_type, _ = mimetypes.guess_type(path.name)
        extra = {"ContentType": content_type} if content_type else {}
        client.upload_file(str(path), bucket, key, ExtraArgs=extra)
        manifest_rows.append(
            {
                "local_path": str(path),
                "bucket": bucket,
                "key": key,
                "size_bytes": path.stat().st_size,
            }
        )
        print(f"uploaded {key}")

    manifest_path = DATA_ROOT / "serving" / version / "docs" / "r2_publish_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

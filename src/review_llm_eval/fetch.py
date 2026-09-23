"""Download the public dataset (Hugging Face parquet export) into ``data/raw/``.

Usage: python -m review_llm_eval.fetch
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

from review_llm_eval.config import DATASET_ID, DATASET_PARQUET_API, RAW_PARQUET


def _use_system_trust_store() -> None:
    """Verify TLS against the operating system's certificate store when ``truststore``
    is installed (needed behind TLS-inspecting antivirus/proxies, where certifi's
    bundle does not contain the local root). Verification is never disabled."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def fetch(dest: Path = RAW_PARQUET, timeout: float = 120.0) -> Path:
    _use_system_trust_store()
    listing = requests.get(DATASET_PARQUET_API, timeout=timeout)
    listing.raise_for_status()
    urls: list[str] = listing.json()["default"]["train"]
    if len(urls) != 1:
        raise RuntimeError(f"expected one parquet shard, got {len(urls)}")
    meta = requests.get(f"https://huggingface.co/api/datasets/{DATASET_ID}", timeout=timeout)
    meta.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(urls[0], timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

    source = {
        "dataset": DATASET_ID,
        "dataset_sha": meta.json().get("sha"),
        "licence": (meta.json().get("cardData") or {}).get("license"),
        "parquet_url": urls[0],
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bytes": dest.stat().st_size,
    }
    dest.with_name("SOURCE.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=RAW_PARQUET)
    args = parser.parse_args()
    path = fetch(args.dest)
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Build a small per-row index of INPUT hashes for the pinned dataset revision.

For every row of the official train and test splits this records only:
  split, idx (row index within the split), title_sha (sha256 of the normalized
  title), image_sha (sha256 of the raw image bytes).

It never reads or stores labels or model outputs. This is the one permitted
pre-Stage-5 read of the test split (the test split is otherwise read only in
Stage 5, after every choice is fixed): it hashes test inputs for
the train/test duplicate check and nothing else.

Parquet shards are downloaded one at a time and deleted after hashing so the
script needs under 1 GB of free disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Shopify/product-catalogue"
# Current Hub head. The revision MLCommons pins (d5c517c5...) was removed when the
# dataset history was squashed on 2026-08-24, so it can no longer be fetched.
REVISION = "002ca15536bb015ebc47b99cc2d716d7040bb09e"
INPUT_COLUMNS = ["product_title", "product_image"]
_WS = re.compile(r"\s+")


def normalize_title(title: str | None) -> str:
    return _WS.sub(" ", (title or "").strip().lower())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/input_index.jsonl"))
    parser.add_argument("--tmp", type=Path, default=Path("data/raw_tmp"))
    args = parser.parse_args()

    files = list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
    shards = {
        split: sorted(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
        for split in ("train", "test")
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as out:
        for split, paths in shards.items():
            idx = 0
            for path in paths:
                local = hf_hub_download(
                    REPO_ID, path, repo_type="dataset", revision=REVISION, local_dir=args.tmp
                )
                parquet = pq.ParquetFile(local)
                for batch in parquet.iter_batches(batch_size=256, columns=INPUT_COLUMNS):
                    titles = batch.column("product_title").to_pylist()
                    images = batch.column("product_image").to_pylist()
                    for title, image in zip(titles, images, strict=True):
                        image_bytes = (image or {}).get("bytes") or b""
                        out.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "idx": idx,
                                    "title_sha": sha(normalize_title(title).encode("utf-8")),
                                    "image_sha": sha(image_bytes) if image_bytes else None,
                                }
                            )
                            + "\n"
                        )
                        idx += 1
                parquet.close()
                Path(local).unlink()
                print(f"{path}: {idx} rows so far in {split}", flush=True)
            counts[split] = idx
    shutil.rmtree(args.tmp, ignore_errors=True)
    print(json.dumps({"revision": REVISION, "rows": counts}))


if __name__ == "__main__":
    main()

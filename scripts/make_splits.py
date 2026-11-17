"""Generate and freeze data/splits.json.

Splits are lists of row indices into the OFFICIAL splits at the pinned dataset
revision:
  val   : 2,000 rows sampled from official train (fixed seed)
  train : official train minus val
  calib : 256 rows sampled from our train (GPTQ calibration)
  test  : every row of official test, untouched

Only row counts are needed, so this never reads any example. Refuses to
overwrite an existing splits.json: the file is frozen once written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20261117
N_VAL = 2000
N_CALIB = 256


def build_splits(n_train_official: int, n_test_official: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train_official)
    val = sorted(int(i) for i in perm[:N_VAL])
    train = sorted(int(i) for i in perm[N_VAL:])
    calib = sorted(int(i) for i in rng.choice(train, size=N_CALIB, replace=False))
    return {
        "seed": seed,
        "n_train_official": n_train_official,
        "n_test_official": n_test_official,
        "train": train,
        "val": val,
        "calib": calib,
        "test": list(range(n_test_official)),
    }


def canonical_sha256(obj: dict) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/input_index.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--revision", default="002ca15536bb015ebc47b99cc2d716d7040bb09e")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists and is frozen; refusing to overwrite.")

    with args.index.open(encoding="utf-8") as f:
        counts = Counter(json.loads(line)["split"] for line in f)

    splits = build_splits(counts["train"], counts["test"])
    splits = {"dataset": "Shopify/product-catalogue", "dataset_revision": args.revision, **splits}
    digest = canonical_sha256(splits)
    args.out.write_text(json.dumps(splits, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    args.out.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: (len(v) if isinstance(v, list) else v) for k, v in splits.items()}
            | {"sha256": digest}
        )
    )


if __name__ == "__main__":
    main()

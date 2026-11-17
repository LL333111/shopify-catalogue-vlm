"""Split reproducibility and disjointness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_splits import N_CALIB, N_VAL, build_splits, canonical_sha256  # noqa: E402


def test_build_splits_is_deterministic() -> None:
    assert build_splits(5000, 700) == build_splits(5000, 700)


def test_splits_are_disjoint_and_complete() -> None:
    s = build_splits(5000, 700)
    train, val, calib = set(s["train"]), set(s["val"]), set(s["calib"])
    assert len(val) == N_VAL and len(calib) == N_CALIB
    assert not train & val
    assert train | val == set(range(5000))
    assert calib <= train
    assert s["test"] == list(range(700))


def test_frozen_file_matches_hash_and_generator() -> None:
    path = ROOT / "data" / "splits.json"
    if not path.exists():
        return
    frozen = json.loads(path.read_text(encoding="utf-8"))
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    assert canonical_sha256(frozen) == recorded
    regenerated = build_splits(frozen["n_train_official"], frozen["n_test_official"], frozen["seed"])
    for key in ("train", "val", "calib", "test"):
        assert frozen[key] == regenerated[key]

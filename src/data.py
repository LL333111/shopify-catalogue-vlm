"""Dataset loading and prompt construction.

Prompts are built by the MLCommons reference's own
`ShopifyGlobalCatalogue.formulate_loaded_sample`, so the system prompt, the user
text, and the image encoding (original bytes, original format, no resizing) are
exactly the official ones.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import official

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SPLIT = {"train": "train", "val": "train", "calib": "train", "test": "test"}


def load_config(path: str | Path = ROOT / "configs" / "base.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen splits and verify them against the recorded hash."""
    path = ROOT / config["dataset"]["splits_file"]
    splits = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise RuntimeError(f"{path} does not match its recorded hash; the splits were modified.")
    if splits["dataset_revision"] != config["dataset"]["revision"]:
        raise RuntimeError("splits.json was built for a different dataset revision.")
    return splits


def load_split(name: str, config: dict[str, Any], *, allow_test: bool = False):
    """Return (hf_dataset_subset, official_row_indices) for one of our splits."""
    if name == "test" and not allow_test:
        raise PermissionError(
            "The test split may only be read in Stage 5, after every choice is fixed. "
            "Pass allow_test=True from the final-evaluation script only."
        )
    from datasets import load_dataset

    splits = load_splits(config)
    indices = splits[name]
    dataset = load_dataset(
        config["dataset"]["repo_id"],
        revision=config["dataset"]["revision"],
        split=OFFICIAL_SPLIT[name],
    )
    return dataset.select(indices), indices


def build_request(sample: dict[str, Any], *, use_guided_decoding: bool = False) -> dict[str, Any]:
    """Official messages (and response_format, if guided decoding is on)."""
    loaded = official.load().task.ShopifyGlobalCatalogue.formulate_loaded_sample(
        sample, use_guided_decoding=use_guided_decoding
    )
    return {
        "messages": loaded.messages,
        "response_format": (
            loaded.response_format.model_dump(mode="json", by_alias=True)
            if loaded.response_format is not None
            else None
        ),
    }

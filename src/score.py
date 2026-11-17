"""Score raw model outputs with the official MLCommons scoring functions.

Input: a raw.jsonl written by src/infer.py, one object per example:
  {"split": ..., "idx": ..., "response": "<raw model text>", "error": null | str,
   "gt_category": ..., "gt_brand": ..., "gt_is_secondhand": ..., ...}

Parsing and the fallback for unparseable outputs mirror the reference's
`_process_chunk`: the response must validate against `ProductMetadata`;
otherwise the category becomes a pad path with the ground truth's depth, the
brand becomes a pad string, and `is_secondhand` is drawn at random (seeded).
The aggregate metrics are then computed by the reference's own functions.

Usage: python -m src.score results/<config>/raw.jsonl [--out metrics.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from . import official

SEED = 12345  # the reference CLI's default random seed


def parse_records(records: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Parse each response the way the reference does; return per-example rows."""
    off = official.load()
    ev, product_metadata = off.evaluation, off.schema.ProductMetadata
    sep = ev._CATEGORY_SEPARATOR
    rng = np.random.default_rng(seed)
    rows = []
    for rec in records:
        gt_category = rec["gt_category"]
        try:
            pred = product_metadata.model_validate_json(rec.get("response") or "")
            parsed = True
        except ValidationError:
            parsed = False
            pred = product_metadata(
                category=sep.join([ev._PRED_CATEGORY_PAD] * len(gt_category.split(sep))),
                brand=ev._PRED_BRAND_PAD,
                is_secondhand=rng.choice([True, False], size=1).tolist()[0],
            )
        inter, pred_len, true_len = ev.get_hierarchical_components(pred.category, gt_category)
        rows.append(
            {
                "split": rec.get("split"),
                "idx": rec["idx"],
                "parsed": parsed,
                "pred_category": pred.category,
                "pred_brand": pred.brand,
                "pred_is_secondhand": pred.is_secondhand,
                "gt_category": gt_category,
                "gt_brand": rec["gt_brand"],
                "gt_is_secondhand": rec["gt_is_secondhand"],
                # Components of hierarchical F1, kept per example for the bootstrap.
                "h_intersection": inter,
                "h_pred_len": pred_len,
                "h_true_len": true_len,
                "category_exact": parsed and pred.category.strip() == gt_category.strip(),
                "completion_tokens": rec.get("completion_tokens"),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate with the reference's metric functions."""
    off = official.load()
    ev = off.evaluation
    n = len(rows)
    if n == 0:
        raise ValueError("No records to score.")
    tokens = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    return {
        "n": n,
        "category_hierarchical_f1": ev.calculate_hierarchical_f1(
            [(r["pred_category"], r["gt_category"]) for r in rows]
        ),
        "category_exact_match": sum(r["category_exact"] for r in rows) / n,
        "brand_score": ev.calculate_brand_f1_score(
            [(r["pred_brand"], r["gt_brand"]) for r in rows]
        ),
        "is_secondhand_f1": float(
            ev.calculate_secondhand_f1(
                [(r["pred_is_secondhand"], r["gt_is_secondhand"]) for r in rows]
            )
        ),
        "is_secondhand_accuracy": sum(
            r["pred_is_secondhand"] == r["gt_is_secondhand"] for r in rows
        )
        / n,
        "json_parse_rate": sum(r["parsed"] for r in rows) / n,
        "mean_completion_tokens": (sum(tokens) / len(tokens)) if tokens else None,
        "scorer": {
            "reference_commit": off.commit,
            "loadgen_stubbed": off.loadgen_stubbed,
            "seed": SEED,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.raw.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    rows = parse_records(records)
    metrics = aggregate(rows)
    out = args.out or args.raw.with_name("metrics.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with args.raw.with_name("scored.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

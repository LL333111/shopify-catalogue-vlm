"""Count val/test examples whose title or image also appears in our train split.

Uses only the input hashes from data/input_index.jsonl (no labels, no outputs).
Writes results/duplicates/summary.json plus the list of affected indices so
Stage 5 can also report test F1 with them excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/input_index.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("results/duplicates"))
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int], dict] = {}
    with args.index.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[(row["split"], row["idx"])] = row

    train_rows = [rows[("train", i)] for i in splits["train"]]
    train_titles = {r["title_sha"] for r in train_rows}
    train_images = {r["image_sha"] for r in train_rows if r["image_sha"]}

    summary: dict = {
        "dataset_revision": splits["dataset_revision"],
        "n_train": len(train_rows),
        "train_unique_titles": len(train_titles),
        "train_unique_images": len(train_images),
        "train_rows_without_image": sum(1 for r in train_rows if not r["image_sha"]),
    }
    affected: dict[str, dict[str, list[int]]] = {}
    for name, official in (("val", "train"), ("test", "test")):
        eval_rows = [rows[(official, i)] for i in splits[name]]
        title = [r["idx"] for r in eval_rows if r["title_sha"] in train_titles]
        image = [r["idx"] for r in eval_rows if r["image_sha"] and r["image_sha"] in train_images]
        both = sorted(set(title) & set(image))
        either = sorted(set(title) | set(image))
        n = len(eval_rows)
        summary[name] = {
            "n": n,
            "rows_without_image": sum(1 for r in eval_rows if not r["image_sha"]),
            "title_in_train": len(title),
            "image_in_train": len(image),
            "title_and_image_in_train": len(both),
            "title_or_image_in_train": len(either),
            "title_or_image_in_train_pct": round(100 * len(either) / n, 2),
        }
        affected[name] = {"title": title, "image": image, "either": either}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "affected_indices.json").write_text(json.dumps(affected), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

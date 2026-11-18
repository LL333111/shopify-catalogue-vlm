"""Draw a fixed random sample of category errors for manual review.

usage: python scripts/error_sample.py <results-name> [n] [seed]
Prints idx, ground truth and prediction for n examples whose category is not an
exact match. Titles are looked up from the dataset so the reviewer can judge them.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_config, load_split  # noqa: E402
from src.score import parse_records  # noqa: E402

name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
with (ROOT / "results" / name / "raw.jsonl").open(encoding="utf-8") as f:
    rows = parse_records([json.loads(line) for line in f if line.strip()])
errors = sorted((r for r in rows if not r["category_exact"]), key=lambda r: r["idx"])
sample = random.Random(seed).sample(errors, n)
split = "test" if name.endswith("_test") else "val"
data, indices = load_split(split, load_config(), allow_test=(split == "test"))
position = {official: pos for pos, official in enumerate(indices)}
text = data.remove_columns([c for c in data.column_names if "image" in c])
out = []
for r in sample:
    ex = text[position[r["idx"]]]
    out.append(
        {
            "idx": r["idx"],
            "title": ex.get("product_title"),
            "candidates": ex.get("potential_product_categories"),
            "gt": r["gt_category"],
            "pred": r["pred_category"],
        }
    )
dest = ROOT / "results" / "final" / f"error_sample_{split}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps({"source": name, "seed": seed, "n_errors": len(errors), "n_total": len(rows), "sample": out}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{len(errors)} errors of {len(rows)}; wrote {dest}")

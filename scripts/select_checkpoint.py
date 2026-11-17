"""Select the fine-tuning checkpoint by val hierarchical F1 (rule fixed before training).

Reads results/sft_frac<f>_val/metrics.json and writes results/selection.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = {}
for path in sorted((ROOT / "results").glob("sft_frac*_val/metrics.json")):
    frac = path.parent.name.removeprefix("sft_frac").removesuffix("_val")
    m = json.loads(path.read_text())
    rows[frac] = {k: m[k] for k in ("category_hierarchical_f1", "category_exact_match", "brand_score", "json_parse_rate")}
baseline = json.loads((ROOT / "results/row1_zeroshot_val/metrics.json").read_text())["category_hierarchical_f1"]
best = max(rows, key=lambda f: rows[f]["category_hierarchical_f1"])
out = {
    "rule": "argmax val category_hierarchical_f1 over saved epoch fractions",
    "zero_shot_val_f1": baseline,
    "checkpoints": rows,
    "selected": best,
    "gate_G1_passed": rows[best]["category_hierarchical_f1"] > baseline,
}
(ROOT / "results/selection.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

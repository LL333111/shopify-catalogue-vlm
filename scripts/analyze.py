"""Build the main table, confidence intervals, and figures from results/.

Every reported number is computed here from raw files (raw.jsonl -> scored rows,
offline.json, server.json); nothing is entered by hand.

Usage:
  python scripts/analyze.py --split test \
      --rows row1=row1_zeroshot row2=row2_sft row3=row3_fp8 row4=row4_w4a16 \
      --ablation row2_textonly --out results/final
Each value is a results-name prefix; this script reads
  results/<prefix>_offline_<split>/{raw.jsonl,offline.json}
  results/<prefix>_server_<split>/server.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.score import aggregate, parse_records  # noqa: E402

REFERENCE_F1 = 0.7903037  # official 235B mean of 10 runs; context only
N_BOOT = 10_000


def load_scored(name: str) -> list[dict]:
    raw = ROOT / "results" / name / "raw.jsonl"
    with raw.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return sorted(parse_records(records), key=lambda r: r["idx"])


def components(rows: list[dict]) -> np.ndarray:
    return np.array([[r["h_intersection"], r["h_pred_len"], r["h_true_len"]] for r in rows], float)


def hf1_from_sums(sums: np.ndarray) -> np.ndarray:
    """Micro hierarchical F1 from summed (intersection, pred_len, true_len)."""
    inter, pred, true = sums[..., 0], sums[..., 1], sums[..., 2]
    hp = np.divide(inter, pred, out=np.zeros_like(inter), where=pred > 0)
    hr = np.divide(inter, true, out=np.zeros_like(inter), where=true > 0)
    return np.divide(2 * hp * hr, hp + hr, out=np.zeros_like(hp), where=(hp + hr) > 0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Paired bootstrap over examples for F1(b) - F1(a) and F1(b) / F1(a)."""
    assert a.shape == b.shape
    n = len(a)
    rng = np.random.default_rng(seed)
    f1_a, f1_b = np.empty(N_BOOT), np.empty(N_BOOT)
    for start in range(0, N_BOOT, 500):  # chunked to bound memory
        idx = rng.integers(0, n, size=(min(500, N_BOOT - start), n))
        f1_a[start : start + len(idx)] = hf1_from_sums(a[idx].sum(axis=1))
        f1_b[start : start + len(idx)] = hf1_from_sums(b[idx].sum(axis=1))
    diff, ratio = f1_b - f1_a, f1_b / f1_a

    def ci(x: np.ndarray) -> list[float]:
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    point_a = float(hf1_from_sums(a.sum(axis=0)))
    point_b = float(hf1_from_sums(b.sum(axis=0)))
    return {
        "f1_a": point_a,
        "f1_b": point_b,
        "diff": point_b - point_a,
        "diff_ci95": ci(diff),
        "ratio": point_b / point_a,
        "ratio_ci95": ci(ratio),
        "p_b_le_a": float((diff <= 0).mean()),
        "n": n,
        "n_boot": N_BOOT,
    }


def single_ci(a: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    out = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 500):
        idx = rng.integers(0, n, size=(min(500, N_BOOT - start), n))
        out[start : start + len(idx)] = hf1_from_sums(a[idx].sum(axis=1))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--rows", nargs="+", required=True, help="label=results_prefix")
    parser.add_argument("--ablation", default=None, help="results prefix of the text-only run")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "final")
    parser.add_argument("--exclude", type=Path, default=ROOT / "results/duplicates/affected_indices.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = dict(item.split("=", 1) for item in args.rows)
    scored: dict[str, list[dict]] = {}
    table: dict[str, dict] = {}
    for label, prefix in rows.items():
        name = f"{prefix}_offline_{args.split}"
        scored[label] = load_scored(name)
        metrics = aggregate(scored[label])
        offline = read_json(ROOT / "results" / name / "offline.json") or {}
        server = read_json(ROOT / "results" / f"{prefix}_server_{args.split}" / "server.json") or {}
        at_max = server.get("at_max") or {}
        table[label] = {
            "results_prefix": prefix,
            **{k: v for k, v in metrics.items() if k != "scorer"},
            "f1_ci95": single_ci(components(scored[label])),
            "offline_samples_per_s": offline.get("samples_per_s"),
            "offline_output_tokens_per_s": offline.get("output_tokens_per_s"),
            "offline_errors": offline.get("errors"),
            "server_max_qps": server.get("max_valid_qps"),
            "server_p50_s": at_max.get("latency_p50_s"),
            "server_p99_s": at_max.get("latency_p99_s"),
            "server_confirmed": server.get("confirmed"),
        }

    comparisons: dict[str, dict] = {}
    labels = list(rows)
    pairs = [("row1", "row2"), ("row2", "row3"), ("row2", "row4")]
    for a, b in pairs:
        if a in scored and b in scored:
            assert [r["idx"] for r in scored[a]] == [r["idx"] for r in scored[b]]
            comparisons[f"{b}_vs_{a}"] = paired_bootstrap(components(scored[a]), components(scored[b]))

    ablation = None
    if args.ablation:
        name = f"{args.ablation}_offline_{args.split}"
        ablation_rows = load_scored(name)
        ablation = {k: v for k, v in aggregate(ablation_rows).items() if k != "scorer"}
        if "row2" in scored:
            ablation["row2_vs_textonly"] = paired_bootstrap(
                components(ablation_rows), components(scored["row2"])
            )

    # Sensitivity: exclude eval examples whose title or image also appears in train.
    dedup = None
    if args.exclude.exists():
        affected = set(json.loads(args.exclude.read_text())[args.split]["either"])
        dedup = {"n_excluded": len(affected)}
        for label in labels:
            kept = [r for r in scored[label] if r["idx"] not in affected]
            dedup[label] = aggregate(kept)["category_hierarchical_f1"]

    # F1 by ground-truth category depth.
    depth: dict[str, dict] = {}
    for label in labels:
        by_depth: dict[int, list] = {}
        for r in scored[label]:
            by_depth.setdefault(r["h_true_len"], []).append(r)
        depth[label] = {
            str(d): {"n": len(v), "f1": float(hf1_from_sums(components(v).sum(axis=0)))}
            for d, v in sorted(by_depth.items())
        }

    summary = {
        "split": args.split,
        "reference_235b_f1": REFERENCE_F1,
        "table": table,
        "comparisons": comparisons,
        "ablation_text_only": ablation,
        "excluding_train_duplicates": dedup,
        "f1_by_depth": depth,
    }
    (args.out / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))

    # Markdown table.
    lines = [
        "| Row | Hier. F1 (95% CI) | Exact | Brand | 2nd-hand F1 | Parse | Out tok | Offline samples/s | Server max QPS | p50 / p99 s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, t in table.items():
        lines.append(
            f"| {label} | {t['category_hierarchical_f1']:.4f} ({t['f1_ci95'][0]:.4f}–{t['f1_ci95'][1]:.4f}) "
            f"| {t['category_exact_match']:.4f} | {t['brand_score']:.4f} | {t['is_secondhand_f1']:.4f} "
            f"| {t['json_parse_rate']:.4f} | {fmt(t['mean_completion_tokens'], '.1f')} "
            f"| {fmt(t['offline_samples_per_s'], '.2f')} | {fmt(t['server_max_qps'], '.2f')} "
            f"| {fmt(t['server_p50_s'], '.1f')} / {fmt(t['server_p99_s'], '.1f')} |"
        )
    lines.append(f"| Ref (235B, cited) | {REFERENCE_F1:.4f} | | | | | | | | |")
    (args.out / f"table_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"comparisons": comparisons, "ablation": ablation, "dedup": dedup}, indent=2))

    # Figures.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, t in table.items():
        if t["offline_samples_per_s"] is None:
            continue
        ax.scatter(t["offline_samples_per_s"], t["category_hierarchical_f1"], s=60)
        ax.annotate(label, (t["offline_samples_per_s"], t["category_hierarchical_f1"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.axhline(REFERENCE_F1, linestyle="--", color="gray", linewidth=1)
    ax.annotate("235B reference (different protocol)", (ax.get_xlim()[0], REFERENCE_F1),
                textcoords="offset points", xytext=(4, 4), color="gray", fontsize=8)
    ax.set_xlabel("Offline throughput (samples/s, one L40S)")
    ax.set_ylabel(f"Category hierarchical F1 ({args.split})")
    ax.set_title("Quality vs. throughput")
    fig.tight_layout()
    fig.savefig(figures / f"quality_vs_throughput_{args.split}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    depths = sorted({int(d) for label in labels for d in depth[label]})
    width = 0.8 / max(1, len(labels))
    for k, label in enumerate(labels):
        values = [depth[label].get(str(d), {}).get("f1", 0) for d in depths]
        ax.bar([i + k * width for i in range(len(depths))], values, width, label=label)
    counts = depth[labels[0]]
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(depths))])
    ax.set_xticklabels([f"{d}\n(n={counts.get(str(d), {}).get('n', 0)})" for d in depths])
    ax.set_xlabel("Ground-truth category depth")
    ax.set_ylabel(f"Hierarchical F1 ({args.split})")
    ax.set_title("F1 by category depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / f"f1_by_depth_{args.split}.png", dpi=150)


if __name__ == "__main__":
    main()

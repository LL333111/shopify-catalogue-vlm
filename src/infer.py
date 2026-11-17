"""Batched inference through an OpenAI-compatible endpoint (vLLM).

Writes every raw output to results/<name>/raw.jsonl, plus run_metadata.json.
Resumable: examples already present in raw.jsonl without an error are skipped.

Usage:
  python -m src.infer --name row1_zeroshot --split val \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from . import official
from .data import ROOT, build_request, load_config, load_split


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    req = config["request"]
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    done: set[int] = set()
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            done = {r["idx"] for r in map(json.loads, f) if r.get("error") is None}

    dataset, indices = load_split(args.split, config, allow_test=args.allow_test)
    todo = [(pos, idx) for pos, idx in enumerate(indices) if idx not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)

    client = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key, timeout=req["timeout_seconds"], max_retries=0
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    started = time.perf_counter()
    completed = 0

    async def one(pos: int, idx: int, out) -> None:
        nonlocal completed
        async with semaphore:
            # Build the request inside the semaphore so at most `concurrency`
            # base64-encoded images are held in memory at once.
            sample = dataset[pos]
            request = build_request(sample, use_guided_decoding=req["use_guided_decoding"])
            if args.text_only:  # ablation: same prompt without the image part
                user = request["messages"][1]
                user["content"] = [part for part in user["content"] if part["type"] == "text"]
            record: dict[str, Any] = {
                "split": args.split,
                "idx": idx,
                "response": "",
                "error": None,
                "gt_category": sample["ground_truth_category"],
                "gt_brand": sample["ground_truth_brand"],
                "gt_is_secondhand": sample["ground_truth_is_secondhand"],
            }
            kwargs = {
                k: req[k] for k in ("temperature", "top_p", "max_tokens") if req.get(k) is not None
            }
            if request["response_format"] is not None:
                kwargs["response_format"] = request["response_format"]
            tic = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=args.model, messages=request["messages"], **kwargs
                )
                record["response"] = response.choices[0].message.content or ""
                record["finish_reason"] = response.choices[0].finish_reason
                record["prompt_tokens"] = response.usage.prompt_tokens
                record["completion_tokens"] = response.usage.completion_tokens
            except Exception as exc:  # noqa: BLE001 - recorded, scored as unparseable
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            record["latency_s"] = time.perf_counter() - tic
        async with write_lock:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            completed += 1
            if completed % 100 == 0:
                rate = completed / (time.perf_counter() - started)
                print(f"{completed}/{len(todo)}  {rate:.2f} samples/s", flush=True)

    with raw_path.open("a", encoding="utf-8") as out:
        await asyncio.gather(*(one(pos, idx, out) for pos, idx in todo))
    elapsed = time.perf_counter() - started

    models = await client.models.list()
    metadata = {
        "name": args.name,
        "split": args.split,
        "model": args.model,
        "served_models": [m.id for m in models.data],
        "base_url": args.base_url,
        "config": config,
        "concurrency": args.concurrency,
        "text_only": args.text_only,
        "n_run": len(todo),
        "elapsed_s": elapsed,
        "samples_per_s": len(todo) / elapsed if todo else None,
        "reference_commit": official.reference_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "notes": args.notes,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"done: {len(todo)} samples in {elapsed:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="results/<name>/")
    parser.add_argument("--split", required=True, choices=["train", "val", "calib", "test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true", help="Stage 5 only.")
    parser.add_argument("--text-only", action="store_true", help="Ablation: omit the image.")
    parser.add_argument("--notes", default="")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

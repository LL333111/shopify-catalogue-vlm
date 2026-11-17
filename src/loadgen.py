"""Load generator for the benchmark's Offline and Server scenarios.

This follows the MLPerf Inference task and scenario definitions for the
Shopify catalogue benchmark but is NOT the official LoadGen, so its results are
not MLPerf results.

Offline: every sample of the split is sent once at a fixed concurrency; reports
  samples/s. The raw outputs are written in the same format as src.infer, so the
  Offline run on a split is also that split's quality run.
Server : open-loop Poisson arrivals at a target rate. A rate is valid when the
  p99 latency (completion time minus *scheduled* arrival time) is <= 12 s over
  the measurement window, with at least `--min-requests` completed and no
  errors. `--search` bisects for the highest valid rate.

Requests are built before any timed window (images already base64-encoded).
As in the reference, requests are non-streaming, so there is no first-token time.

Usage:
  python -m src.loadgen offline --name row1_offline_val --split val --model <served-name>
  python -m src.loadgen server  --name row1_server_val  --split val --model <served-name> \
      --search --qps-hint 3.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from . import official
from .data import ROOT, build_request, load_config, load_split

TARGET_LATENCY_S = 12.0
TARGET_PERCENTILE = 0.99


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build every request up front, outside any timed window."""
    dataset, indices = load_split(args.split, config, allow_test=args.allow_test)
    guided = config["request"]["use_guided_decoding"]

    def build(pos: int) -> dict[str, Any]:
        sample = dataset[pos]
        request = build_request(sample, use_guided_decoding=guided)
        if args.text_only:
            user = request["messages"][1]
            user["content"] = [part for part in user["content"] if part["type"] == "text"]
        return {
            "idx": indices[pos],
            "request": request,
            "gt_category": sample["ground_truth_category"],
            "gt_brand": sample["ground_truth_brand"],
            "gt_is_secondhand": sample["ground_truth_is_secondhand"],
        }

    n = min(len(indices), args.limit) if args.limit else len(indices)
    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        items = list(pool.map(build, range(n)))
    print(f"prepared {n} requests in {time.perf_counter() - tic:.0f}s", flush=True)
    return items


class Client:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        req = config["request"]
        self.model = args.model
        self.kwargs = {
            k: req[k] for k in ("temperature", "top_p", "max_tokens") if req.get(k) is not None
        }
        self.client = AsyncOpenAI(
            base_url=args.base_url, api_key="EMPTY", timeout=req["timeout_seconds"], max_retries=0
        )

    async def send(self, item: dict[str, Any]) -> dict[str, Any]:
        request = item["request"]
        kwargs = dict(self.kwargs)
        if request["response_format"] is not None:
            kwargs["response_format"] = request["response_format"]
        record: dict[str, Any] = {"idx": item["idx"], "response": "", "error": None}
        record["send_time"] = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=request["messages"], **kwargs
            )
            record["response"] = response.choices[0].message.content or ""
            record["finish_reason"] = response.choices[0].finish_reason
            record["prompt_tokens"] = response.usage.prompt_tokens
            record["completion_tokens"] = response.usage.completion_tokens
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record["done_time"] = time.perf_counter()
        return record


def client_cpu_fraction(cpu_start: float, wall_start: float) -> float:
    return (time.process_time() - cpu_start) / max(1e-9, time.perf_counter() - wall_start)


async def run_offline(args, config, items, client: Client) -> dict[str, Any]:
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    records: list[dict[str, Any]] = []

    async def one(item: dict[str, Any]) -> None:
        async with semaphore:
            record = await client.send(item)
        record.update(
            split=args.split,
            gt_category=item["gt_category"],
            gt_brand=item["gt_brand"],
            gt_is_secondhand=item["gt_is_secondhand"],
            latency_s=record["done_time"] - record["send_time"],
        )
        records.append(record)
        if len(records) % 500 == 0:
            print(f"{len(records)}/{len(items)}", flush=True)

    cpu0, wall0 = time.process_time(), time.perf_counter()
    await asyncio.gather(*(one(item) for item in items))
    elapsed = time.perf_counter() - wall0

    with (out_dir / "raw.jsonl").open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda r: r["idx"]):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    ok = [r for r in records if r["error"] is None]
    return {
        "scenario": "offline",
        "n": len(records),
        "errors": len(records) - len(ok),
        "elapsed_s": elapsed,
        "samples_per_s": len(records) / elapsed,
        "prompt_tokens_per_s": sum(r["prompt_tokens"] for r in ok) / elapsed,
        "output_tokens_per_s": sum(r["completion_tokens"] for r in ok) / elapsed,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in ok) / max(1, len(ok)),
        "mean_output_tokens": sum(r["completion_tokens"] for r in ok) / max(1, len(ok)),
        "latency_p50_s": percentile([r["latency_s"] for r in records], 0.5),
        "latency_p99_s": percentile([r["latency_s"] for r in records], 0.99),
        "concurrency": args.concurrency,
        "client_cpu_fraction": client_cpu_fraction(cpu0, wall0),
    }


async def run_server_once(args, items, client: Client, qps: float, seed: int) -> dict[str, Any]:
    """One open-loop Poisson run at `qps`. Returns the probe summary."""
    rng = random.Random(seed)
    duration = max(args.duration, args.min_requests / qps)
    order = list(range(len(items)))
    rng.shuffle(order)

    # Scheduled arrival offsets: warm-up window, then the measurement window.
    arrivals: list[float] = []
    t = 0.0
    while t < args.warmup + duration:
        t += rng.expovariate(qps)
        arrivals.append(t)
    measured_from = args.warmup
    n_measured = sum(1 for a in arrivals if a >= measured_from)
    # More than this many late requests makes p99 > 12 s unavoidable: stop early.
    late_budget = math.floor((1 - TARGET_PERCENTILE) * n_measured)

    records: list[dict[str, Any]] = []
    late = 0
    abort = asyncio.Event()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    start = time.perf_counter()

    async def one(k: int, offset: float) -> None:
        nonlocal late
        record = await client.send(items[order[k % len(order)]])
        record["scheduled"] = offset
        record["latency_s"] = record["done_time"] - (start + offset)
        record["send_lag_s"] = record["send_time"] - (start + offset)
        record["measured"] = offset >= measured_from
        records.append(record)
        if record["measured"] and (record["latency_s"] > TARGET_LATENCY_S or record["error"]):
            late += 1
            if late > late_budget:
                abort.set()

    tasks: list[asyncio.Task] = []
    for k, offset in enumerate(arrivals):
        delay = start + offset - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(abort.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        if abort.is_set():
            break
        tasks.append(asyncio.create_task(one(k, offset)))

    aborted = abort.is_set()
    if aborted:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(args.cooldown)  # let the server drain cancelled work
    else:
        await asyncio.gather(*tasks)

    measured = [r for r in records if r["measured"]]
    latencies = [r["latency_s"] for r in measured]
    errors = sum(1 for r in measured if r["error"])
    p99 = percentile(latencies, TARGET_PERCENTILE)
    valid = (
        not aborted
        and errors == 0
        and len(measured) >= args.min_requests
        and p99 <= TARGET_LATENCY_S
    )
    window = max(1e-9, duration)
    return {
        "target_qps": qps,
        "valid": valid,
        "aborted_early": aborted,
        "duration_s": duration,
        "warmup_s": args.warmup,
        "completed_measured": len(measured),
        "achieved_qps": len(measured) / window if not aborted else None,
        "errors": errors,
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p90_s": percentile(latencies, 0.9),
        "latency_p99_s": p99,
        "latency_max_s": max(latencies) if latencies else None,
        "max_send_lag_s": max((r["send_lag_s"] for r in records), default=None),
        "mean_output_tokens": (
            sum(r.get("completion_tokens", 0) for r in measured if not r["error"])
            / max(1, len(measured) - errors)
        ),
        "client_cpu_fraction": client_cpu_fraction(cpu0, wall0),
        "seed": seed,
    }


async def run_server(args, items, client: Client) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []

    async def probe(qps: float) -> bool:
        result = await run_server_once(args, items, client, qps, seed=args.seed + len(probes))
        probes.append(result)
        print(
            f"probe qps={qps:.3f} valid={result['valid']} p99={result['latency_p99_s']:.2f}s "
            f"n={result['completed_measured']} aborted={result['aborted_early']}",
            flush=True,
        )
        return result["valid"]

    if not args.search:
        await probe(args.qps)
        best = args.qps if probes[-1]["valid"] else None
    else:
        lo, hi = None, None  # highest valid, lowest invalid
        qps = args.qps_hint
        while len(probes) < args.max_probes:
            if await probe(qps):
                lo = qps
            else:
                hi = qps
            if lo is not None and hi is not None:
                if hi / lo <= 1 + args.precision:
                    break
                qps = (lo + hi) / 2
            elif lo is None:
                qps = hi * 0.7
            else:
                qps = lo * 1.3
        best = lo
        if best is not None and args.confirm:
            confirmed = await probe(best)  # re-run the final rate with a new seed
            probes[-1]["confirmation"] = True
            if not confirmed:
                best = None
    best_probe = next((p for p in reversed(probes) if p["target_qps"] == best), None)
    return {
        "scenario": "server",
        "max_valid_qps": best,
        "confirmed": bool(args.search and args.confirm and best is not None),
        "at_max": best_probe,
        "probes": probes,
        "precision": args.precision,
        "latency_constraint": f"p{int(TARGET_PERCENTILE * 100)} <= {TARGET_LATENCY_S}s",
    }


async def main_async(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    items = prepare(args, config)
    client = Client(args, config)
    out_dir = ROOT / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scenario == "offline":
        summary = await run_offline(args, config, items, client)
    else:
        summary = await run_server(args, items, client)
    summary.update(
        name=args.name,
        split=args.split,
        model=args.model,
        text_only=args.text_only,
        n_requests_prepared=len(items),
        request_config=config["request"],
        vllm_flags=config["vllm"]["flags"],
        reference_commit=official.reference_commit(),
        client_cpus=os.cpu_count(),
    )
    (out_dir / f"{args.scenario}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "probes"}, indent=2)[:1500])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["offline", "server"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true", help="Stage 5 only.")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=256, help="offline")
    parser.add_argument("--qps", type=float, default=1.0, help="server, single run")
    parser.add_argument("--search", action="store_true", help="server: bisect for max QPS")
    parser.add_argument("--qps-hint", type=float, default=2.0)
    parser.add_argument("--precision", type=float, default=0.05, help="stop when hi/lo <= 1+p")
    parser.add_argument("--max-probes", type=int, default=8)
    parser.add_argument("--confirm", action="store_true", help="re-run the final rate once")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--warmup", type=float, default=30.0)
    parser.add_argument("--cooldown", type=float, default=20.0)
    parser.add_argument("--min-requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

# Stage 0 — Environment and verification (2026-11-17)

Everything here was checked against the live sources on the date above.

## Pinned sources

| Item | Value |
|---|---|
| Reference implementation | `mlcommons/inference`, `multimodal/qwen3-vl`, commit `3fbc329939999c13d0a7b5e67fb2092287e06047` |
| Dataset | `Shopify/product-catalogue`, revision `002ca15536bb015ebc47b99cc2d716d7040bb09e` |
| Model | `Qwen/Qwen3-VL-4B-Instruct` (two safetensors shards, 8.9 GB) |
| Python | 3.12 |
| Training / quantization env | torch 2.14.0, transformers 5.21.0, peft 0.21.3, llmcompressor 0.15.0 (compressed-tensors 0.20.1) |
| Serving env | vllm 0.34.0 (torch 2.14.0, compressed-tensors 0.17.0) |

To restore the reference checkout (it is not committed):

```bash
git clone --filter=blob:none --sparse https://github.com/mlcommons/inference.git third_party/mlcommons-inference
git -C third_party/mlcommons-inference sparse-checkout set multimodal
git -C third_party/mlcommons-inference checkout 3fbc329939999c13d0a7b5e67fb2092287e06047
```

## What the reference implementation defines

- **System prompt and user prompt:** `task.py`, `ShopifyGlobalCatalogue.formulate_loaded_sample`. We call this function directly (`src/data.py`), so the prompt is the official one. The image is sent as the original bytes in the original format, base64-encoded.
- **Output schema:** `schema.py`, `ProductMetadata` (`category: str`, `brand: str`, `is_secondhand: bool`, extra fields forbidden).
- **Scorer:** `evaluation.py`. `src/score.py` imports `get_hierarchical_components`, `calculate_hierarchical_f1`, `calculate_brand_f1_score`, and `calculate_secondhand_f1` from it and mirrors `_process_chunk` for parsing.
  - Hierarchical F1 is micro-averaged: summed longest-common-prefix length over summed predicted and true path lengths.
  - A response that fails schema validation (including JSON wrapped in a code fence) becomes a pad category with the ground truth's depth, a pad brand, and a random `is_secondhand`.
  - Brand is fuzzy-match accuracy (rapidfuzz ratio > 90 after lowercasing). `is_secondhand` is F1.
- **Decoding:** no guided decoding; no sampling parameters in the request, so the server uses the model's `generation_config.json` (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0 — Qwen3-VL-4B ships the same values as the 235B model); no `max_tokens`.
- **Server flags:** `--max-model-len=32768`, `--limit-mm-per-prompt.video=0`, `--no-enable-prefix-caching`.
- **Rules that affect us:** images must not be resized, re-encoded, or pixel-capped; Server latency constraint is p99 ≤ 12 s; quality threshold is 0.7824 (99% of 0.7903037, the mean of 10 runs, standard deviation 0.000225).
- **Round:** the README now documents v6.1, which uses the separate `mlcommons/endpoints` load generator and adds an Interactive scenario (p99 ≤ 1.5 s on an 8,000-sample subset, reference F1 0.7878). The v6.0 code we reuse is marked deprecated but unchanged. We do not run the Interactive scenario.

## Changes to the initial design

1. The dataset revision MLCommons pins (`d5c517c5…`) returns "revision not found": the dataset history was squashed into a single commit on 2026-08-24. We pin the current head. Row counts match the benchmark's (below); we cannot prove the contents are byte-identical.
2. The plan's quantization ignore list (`vision_tower`, `multi_modal_projector`) does not match Qwen3-VL, whose vision modules live under `visual`. llm-compressor's own `examples/multimodal_vision/qwen3_vl_example.py` uses `ignore=["re:.*lm_head", "re:.*visual.*"]` and `sequential_targets=["Qwen3VLTextDecoderLayer"]`. That example uses AWQ + W4A16 with 512 calibration samples at 2,048 tokens; we keep GPTQ W4A16 with our 256-sample calib split.
3. The plan called for capping image resolution. The benchmark forbids this at inference, so we do not cap at inference or serving. Training memory is bounded by truncating the description instead.
4. llmcompressor and vllm pin different compressed-tensors versions, so they cannot share an environment. Loading the quantized checkpoints in vLLM is an explicit Stage 3 check.
5. We set `max_tokens: 256` as a runaway guard; the reference sets none. All other request settings follow the reference.

## Environment notes

- All GPU work (Stage 1 onward) runs on a single NVIDIA L40S (48 GB) under Linux; vLLM does not run natively on Windows.
- MLPerf LoadGen has no Windows wheel. `src/official.py` registers an empty stand-in module when it is missing; the functions we use never call LoadGen.

## Dataset and splits

- Official rows at the pinned revision: 38,631 train and 9,658 test, 48,289 in total, matching the benchmark's stated size. Every row has an image.
- `data/splits.json` (frozen; sha256 `9cdc343a8a3e1bc1eaf4ddb914b03d266f80b50330a02d1aee35194e84e35bca`, seed 20261117): train 36,631, val 2,000, calib 256 (a subset of train), test 9,658. `tests/test_splits.py` checks the file against its hash and against the generator.
- The raw dataset (9.5 GB) is not stored locally. `scripts/build_input_index.py` streamed it shard by shard and kept only input hashes in `data/input_index.jsonl` (not committed; regenerate with that script).

## Train/test duplicate check

`scripts/check_duplicates.py`, exact matches only (sha256 of the lowercased, whitespace-normalized title; sha256 of the raw image bytes). Test labels and model outputs were not read.

| Eval split | n | Title also in train | Image also in train | Either | Share |
|---|---:|---:|---:|---:|---:|
| val | 2,000 | 8 | 1 | 9 | 0.45% |
| test | 9,658 | 43 | 3 | 44 | 0.46% |

Exact-duplicate leakage is negligible. This does not detect near-duplicates (reworded titles, re-encoded images, same product from another merchant). The affected indices are in `results/duplicates/affected_indices.json` so Stage 5 can also report test F1 without them.

# Report

Headline results are on the held-out test split (section 5). Sections 2–4 describe choices made on validation before the test set was read.

## 1. Setup

**Task.** Shopify's product-catalogue benchmark: from a product's title, description, image and 8–9 candidate categories, return the category path, brand and second-hand flag as JSON. The prompt, output schema and scorer are taken unchanged from the MLCommons inference reference (`multimodal/qwen3-vl`, commit `3fbc329`). As in the reference, there is no guided decoding and sampling settings come from the model's generation config. The headline metric is micro-averaged hierarchical F1 on the category path.

**Data.** `Shopify/product-catalogue`, pinned to revision `002ca15`. The public data was re-split by hash into train 36,631 / validation 2,000 / calibration 256 / test 9,658. The test split was not read until all choices were fixed. 44 test and 9 validation examples have an exact duplicate in train; results are reported with and without them.

**Models.** Four rows, all served with the same vLLM command:

1. `Qwen/Qwen3-VL-4B-Instruct`, zero-shot.
2. Row 1 plus a LoRA adapter (r=16, alpha=32, language model only, 33.0M trainable parameters), merged.
3. Row 2 quantized to FP8 (dynamic activations, no calibration).
4. Row 2 quantized with GPTQ to 4-bit weights, 16-bit activations, 256 calibration examples.

In rows 3 and 4 the vision tower and output head stay in full precision; 252 linear layers are quantized.

**Hardware and software.** One NVIDIA L40S (48 GB). vLLM 0.34.0, torch 2.14.0, transformers 5.22.2, llm-compressor 0.15.0.

## 2. Fine-tuning

One epoch, learning rate 1e-4, 4,743 steps, 11.9 GPU-hours. Checkpoints at 25/50/75/100% of the epoch were scored on validation; the 75% checkpoint was selected (the 100% checkpoint is within noise of it).

| | Hier. F1 (95% CI) | Exact match | Brand | JSON parse |
|---|---|---:|---:|---:|
| Zero-shot | 0.749 (0.734–0.765) | 0.520 | 0.418 | 0.998 |
| + LoRA | 0.820 (0.806–0.833) | 0.617 | 0.457 | 1.000 |

Validation, 2,000 examples. The paired bootstrap difference is +0.070 (95% CI +0.056 to +0.085).

**Image ablation.** A second adapter was trained and evaluated on text only. It reaches 0.808 on validation; the image adds +0.012 (95% CI +0.001 to +0.023). Most of the signal is in the title and description. The text-only run used fewer optimizer steps (2,552 vs 4,743), so this difference is an upper bound on the image's contribution rather than a clean estimate.

## 3. Quantization

| | Hier. F1 | Ratio to row 2 (95% CI) | Weights in GPU memory |
|---|---:|---|---:|
| Row 2 (BF16) | 0.820 | | 8.44 GiB |
| FP8 | 0.814 | 0.993 (0.985–1.002) | 5.07 GiB |
| GPTQ W4A16 | 0.815 | 0.994 (0.986–1.003) | 3.35 GiB |

Validation, 2,000 examples. Both variants meet the 99% retention target on the point estimate, but the lower bounds are just under 0.99. The test split (section 5) is large enough to settle it.

## 4. Serving

The load generator was written for this project and follows the benchmark's two scenarios. **Offline** sends every sample once at fixed concurrency and reports samples/s. **Server** sends Poisson arrivals and searches for the highest rate at which p99 latency stays at or below 12 s, with at least 1,000 measured requests and a confirmation run at the final rate. Search precision is 5%.

| | Offline samples/s | Server QPS | p99 at that rate | KV cache (tokens) |
|---|---:|---:|---:|---:|
| 1. Zero-shot | 4.05 | 2.02 | 9.2 s | 229k |
| 2. + LoRA | 4.08 | 2.02 | 9.3 s | 229k |
| 3. + FP8 | 4.49 | 2.49 | 9.4 s | 252k |
| 4. + GPTQ W4A16 | 4.27 | 2.02 | 8.5 s | 266k |

Validation split, all four rows on the same machine.

- **FP8 is the practical choice.** +10% Offline throughput and +23% Server capacity over the unquantized model, at 60% of the weight memory.
- **4-bit weights save memory, not time.** W4A16 is 5% faster Offline and no faster under the latency limit. Rows 1, 2 and 4 all ended on the same search grid point (2.02 passed, 2.1 failed), so differences under 5% are not resolved.
- **The workload is prompt-bound.** Prompts average 2,327 tokens (mostly image) against about 40 output tokens. Offline throughput is flat from concurrency 32 to 256 with the GPU at 100%, so the GPU's time goes to reading prompts, which limits what weight quantization can gain.

## 5. Held-out test results

9,658 examples, read once after every choice above was fixed.

| | Hier. F1 (95% CI) | Exact match | Brand | Second-hand F1 | JSON parse |
|---|---|---:|---:|---:|---:|
| 1. Zero-shot | 0.755 (0.748–0.762) | 0.513 | 0.421 | 0.14 | 0.999 |
| 2. + LoRA | 0.823 (0.817–0.829) | 0.605 | 0.457 | 0.77 | 1.000 |
| 3. + FP8 | 0.823 (0.817–0.829) | 0.601 | 0.456 | 0.77 | 1.000 |
| 4. + GPTQ W4A16 | 0.819 (0.813–0.825) | 0.595 | 0.452 | 0.70 | 1.000 |

Paired bootstrap comparisons (10,000 resamples):

| Comparison | F1 difference (95% CI) | Ratio (95% CI) |
|---|---|---|
| LoRA vs zero-shot | +0.068 (+0.062 to +0.075) | 1.091 (1.082–1.100) |
| FP8 vs LoRA | +0.000 (−0.003 to +0.003) | 1.000 (0.997–1.004) |
| W4A16 vs LoRA | −0.004 (−0.007 to −0.001) | 0.995 (0.991–0.999) |
| LoRA vs text-only LoRA | +0.005 (+0.001 to +0.010) | |

- **Fine-tuning** gives +6.8 points and the result agrees with validation (0.820). Excluding the 44 test examples duplicated in train changes no row by more than 0.0002.
- **FP8 is lossless** within ±0.3 points. **W4A16 loses 0.4 points**, a small but statistically clear loss; both retain more than 99% with the whole interval above 0.99.
- **The image matters little.** The text-only model reaches 0.818; the image adds about half a point.
- **Where the gain is.** Fine-tuning helps most on products whose correct label is a shallow category (label depth 3: 0.62 → 0.76; depth 6: 0.82 → 0.87). The zero-shot model tends to pick an over-specific leaf; the fine-tuned model has learned when to stop higher in the tree. See `figures/f1_by_depth_test.png`.

**Error review.** I read 20 randomly sampled category errors of the fine-tuned model (`results/final/error_review_test.md`), judging from the title and candidate list only:

| Verdict | Count |
|---|---:|
| Model error | 10 (5 are right branch, wrong depth) |
| Questionable label | 9 |
| Correct answer absent from candidates | 1 |

Roughly half of the sampled errors are disagreements with labels that are themselves doubtful (for example, a turn signal labelled as a bicycle RPM sensor). This suggests the remaining headroom on this metric is smaller than 1 − 0.82 implies, though 20 examples only support a rough estimate.

**Throughput during the test runs.** These ran on a second L40S (see section 8) that is about 25% slower in absolute terms: 3.10, 3.11, 3.55 and 3.24 samples/s for rows 1–4. The ordering and relative gains (FP8 +14%, W4A16 +4%) match section 4.

## 6. Limitations

1. **Not the official protocol.** MLCommons reports 0.790 for Qwen3-VL-235B on the official test set and harness. These numbers use a re-split of the public data at a different dataset revision, so the comparison is indicative only. This is not an MLPerf submission.
2. **One training run.** One learning rate and one seed; run-to-run variance is not measured.
3. **Train/inference mismatch.** Training capped images at 2,048 tokens and descriptions at 6,000 characters to fit memory; inference uses the full inputs.
4. **Small-sample metrics.** The second-hand flag is rare (about 15 positives in validation), so its F1 is noisy.
5. **One GPU type.** Serving results are specific to one L40S and one vLLM version.

## 7. Deviations from the reference

- `max_tokens` is 256 (outputs average 40 tokens).
- Offline concurrency is fixed at 64 after a sweep showed no change between 32 and 256.
- The load generator is not the official LoadGen.

## 8. What did not work

- **Training without gradient checkpointing** ran out of memory on 48 GB; checkpointing plus the image cap fixed it.
- **First Server search for row 2** was rejected: Poisson sampling produced slightly fewer than the required 1,000 measured requests. The generator now extends the window until the minimum is met; the run was repeated.
- **Server check on the second machine.** Row 2 at 2.02 QPS, the rate found on the first machine, failed on the slower second machine (p99 19.9 s, stopped early). It is kept in `results/row2_sft_server_test/` and not used.
- **Loss of the first machine.** The first GPU machine became unavailable during the test runs and could not be restarted on its host. The merged and quantized models were rebuilt on a second L40S from the same adapters with the same recipe. Rebuilt FP8 and W4A16 score 0.816 and 0.817 on validation against 0.814 and 0.815 for the originals. The second machine serves the same model about 25% slower, so serving numbers are reported only from the first machine (validation), and the test split is used for quality only.

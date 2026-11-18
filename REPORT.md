# Report

> **Draft.** Sections 1–4 and 6–8 are final. The held-out test results (section 5) are pending.

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

Both variants meet the 99% retention target on the point estimate. The lower confidence bounds are just under 0.99, so a loss of up to about 1.5% cannot be excluded with 2,000 examples.

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

*Pending.*

## 6. Limitations

1. **Not the official protocol.** MLCommons reports 0.790 for Qwen3-VL-235B on the official test set and harness. These numbers use a re-split of the public data at a different dataset revision, so the comparison is indicative only. This is not an MLPerf submission.
2. **One training run.** One learning rate and one seed; run-to-run variance is not measured.
3. **Train/inference mismatch.** Training capped images at 2,048 tokens and descriptions at 6,000 characters to fit memory; inference uses the full inputs.
4. **Small-sample metrics.** The second-hand flag has about 15 positives in validation, so its F1 is not reliable there.
5. **One GPU type.** Serving results are specific to one L40S and one vLLM version.

## 7. Deviations from the reference

- `max_tokens` is 256 (outputs average 40 tokens).
- Offline concurrency is fixed at 64 after a sweep showed no change between 32 and 256.
- The load generator is not the official LoadGen.

## 8. What did not work

- **Training without gradient checkpointing** ran out of memory on 48 GB; checkpointing plus the image cap fixed it.
- **First Server search for row 2** was rejected: Poisson sampling produced slightly fewer than the required 1,000 measured requests. The generator now extends the window until the minimum is met; the run was repeated.
- **Loss of the first machine.** The first GPU machine became unavailable during the test runs and could not be restarted on its host. The merged and quantized models were rebuilt on a second L40S from the same adapters with the same recipe. Rebuilt FP8 and W4A16 score 0.816 and 0.817 on validation against 0.814 and 0.815 for the originals. The second machine serves the same model about 25% slower, so serving numbers are reported only from the first machine (validation), and the test split is used for quality only.

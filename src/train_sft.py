"""LoRA supervised fine-tuning of Qwen3-VL on the product-catalogue task.

- Prompt: the MLCommons reference's system and user text (via src.data.build_request),
  rendered with the model's chat template in the same order vLLM uses
  (user text, then image).
- Target: compact JSON with the three ground-truth fields.
- Loss: only on the assistant's answer tokens (and the closing <|im_end|>).
- LoRA on the language model's attention and MLP projections; the vision tower,
  merger, and lm_head stay frozen.
- Training-only memory bounds (recorded in the run metadata): images above
  --image-max-tokens are downscaled and descriptions above --desc-max-chars are
  truncated. Inference never caps images.

Usage:
  python -m src.train_sft --name sft_lr1e-4 --lr 1e-4 --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .data import ROOT, build_request, load_config, load_split

LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)
PIXELS_PER_TOKEN = 32 * 32  # patch 16, 2x2 merge


def target_json(sample: dict[str, Any]) -> str:
    return json.dumps(
        {
            "category": sample["ground_truth_category"],
            "brand": sample["ground_truth_brand"] or "",
            "is_secondhand": bool(sample["ground_truth_is_secondhand"]),
        },
        ensure_ascii=False,
    )


def to_chat(sample: dict[str, Any], *, text_only: bool, desc_max_chars: int, answer: str | None):
    """Reference prompt as HF chat messages. Returns (messages, image or None)."""
    sample = dict(sample)
    description = sample["product_description"] or ""
    if desc_max_chars and len(description) > desc_max_chars:
        sample["product_description"] = description[:desc_max_chars]
    system, user = build_request(sample)["messages"]
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user["content"][0]["text"]}]
    image = None
    if not text_only:
        image = sample["product_image"].convert("RGB")
        user_content.append({"type": "image", "image": image})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system["content"]}]},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages, image


class CatalogueDataset(Dataset):
    def __init__(self, hf_dataset, args: argparse.Namespace) -> None:
        self.ds = hf_dataset
        self.args = args

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample = self.ds[i]
        answer = target_json(sample)
        messages, image = to_chat(
            sample,
            text_only=self.args.text_only,
            desc_max_chars=self.args.desc_max_chars,
            answer=answer,
        )
        return {"messages": messages, "image": image, "answer": answer}


class Collator:
    def __init__(self, processor, args: argparse.Namespace) -> None:
        self.processor = processor
        self.args = args
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [
            self.processor.apply_chat_template(
                b["messages"], tokenize=False, add_generation_prompt=False
            )
            for b in batch
        ]
        kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
        if not self.args.text_only:
            kwargs["images"] = [b["image"] for b in batch]
            kwargs["images_kwargs"] = {
                "max_pixels": self.args.image_max_tokens * PIXELS_PER_TOKEN
            }
        enc = self.processor(**kwargs)
        ids = enc["input_ids"]
        labels = torch.full_like(ids, -100)
        h = len(self.header)
        header = torch.tensor(self.header)
        for row in range(ids.shape[0]):
            end = int(enc["attention_mask"][row].sum())
            # Last assistant header in the row; the answer follows it.
            start = None
            for pos in range(end - h, -1, -1):
                if torch.equal(ids[row, pos : pos + h], header):
                    start = pos + h
                    break
            if start is None:
                raise RuntimeError("assistant header not found in tokenized example")
            # Answer tokens plus the closing <|im_end|>; drop anything after it.
            stop = start
            while stop < end and ids[row, stop] != self.im_end:
                stop += 1
            labels[row, start : stop + 1] = ids[row, start : stop + 1]
        enc["labels"] = labels
        return dict(enc)


def estimate_lengths(hf_dataset, args: argparse.Namespace, cache: Path) -> list[int]:
    """Rough token count per example, for length-bucketed batching."""
    if cache.exists():
        lengths = json.loads(cache.read_text())
        if len(lengths) == len(hf_dataset):
            return lengths
    lengths = []
    cap = args.image_max_tokens
    text_cols = hf_dataset.remove_columns(["product_image"])
    for i in range(len(hf_dataset)):
        row = text_cols[i]
        chars = (
            len(row["product_title"] or "")
            + min(len(row["product_description"] or ""), args.desc_max_chars or 10**9)
            + sum(len(c) for c in row["potential_product_categories"])
        )
        n = 420 + chars // 3  # system prompt + schema is about 400 tokens
        if not args.text_only:
            width, height = hf_dataset[i]["product_image"].size
            n += min(max(width * height // PIXELS_PER_TOKEN, 64), cap)
        lengths.append(n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lengths))
    return lengths


def make_batches(lengths: list[int], budget: int, max_batch: int, seed: int) -> list[list[int]]:
    """Shuffle, sort within pools by length, cut batches under a padded-token budget."""
    rng = random.Random(seed)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    pool_size = 2048
    for p in range(0, len(order), pool_size):
        pool = sorted(order[p : p + pool_size], key=lambda i: lengths[i])
        current: list[int] = []
        longest = 0
        for i in pool:
            new_longest = max(longest, lengths[i])
            if current and (new_longest * (len(current) + 1) > budget or len(current) >= max_batch):
                batches.append(current)
                current, new_longest = [], lengths[i]
            current.append(i)
            longest = new_longest
        if current:
            batches.append(current)
    rng.shuffle(batches)
    return batches


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--image-max-tokens", type=int, default=2048)
    parser.add_argument("--desc-max-chars", type=int, default=6000)
    parser.add_argument("--token-budget", type=int, default=16000)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--save-fracs", default="0.5,1.0", help="fractions of training to save at")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("/workspace/checkpoints"))
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (timing runs)")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--attn", default="sdpa")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(args.seed)
    config = load_config()
    model_id = config["model"]["repo_id"]
    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results" / args.name
    results_dir.mkdir(parents=True, exist_ok=True)

    hf_train, indices = load_split("train", config)
    if args.max_train_samples:
        subset = random.Random(args.seed).sample(range(len(indices)), args.max_train_samples)
        hf_train = hf_train.select(sorted(subset))
    tag = f"{len(hf_train)}_{'text' if args.text_only else 'img' + str(args.image_max_tokens)}"
    lengths = estimate_lengths(hf_train, args, args.out_root / f"lengths_{tag}.json")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation=args.attn
    )
    print("attention implementation:", model.config._attn_implementation, flush=True)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    backbone, lm_head = model.model, model.lm_head  # LoRA is injected in place below
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in trainable)
    assert all("language_model" in n for n, p in model.named_parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.1f}M", flush=True)

    n_epochs = math.ceil(args.epochs)
    batches: list[list[int]] = []
    for epoch in range(n_epochs):
        epoch_batches = make_batches(lengths, args.token_budget, args.max_batch, args.seed + epoch)
        if epoch + 1 > args.epochs:  # fractional last epoch
            epoch_batches = epoch_batches[: int(len(epoch_batches) * (args.epochs - epoch))]
        batches.extend(epoch_batches)
    if args.max_steps:
        batches = batches[: args.max_steps]
    total_steps = len(batches)
    save_steps = {max(1, round(float(f) * total_steps)): f for f in args.save_fracs.split(",")}

    collator = Collator(processor, args)
    loader = DataLoader(
        CatalogueDataset(hf_train, args),
        batch_sampler=batches,
        collate_fn=collator,
        num_workers=12,
        prefetch_factor=4,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    print(f"{len(hf_train)} examples, {total_steps} steps, saves at {sorted(save_steps)}", flush=True)
    log_path = results_dir / "train_log.jsonl"
    log = log_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    tokens_seen = examples_seen = 0
    window: list[float] = []
    checked = False
    saved: dict[str, str] = {}
    model.train()
    for step, batch in enumerate(loader, start=1):
        if not checked:  # one-time sanity check of the loss mask
            row_labels = batch["labels"][0]
            answer = processor.tokenizer.decode(row_labels[row_labels != -100])
            print("supervised text of first example:", repr(answer), flush=True)
            assert re.fullmatch(r"\{.*\}<\|im_end\|>", answer, flags=re.S), answer
            checked = True
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step - 1)
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone(**batch).last_hidden_state
            # Project only the supervised positions through lm_head: the full
            # (tokens x vocab) logits tensor is what runs out of memory.
            mask = labels[:, 1:] != -100
            logits = lm_head(hidden[:, :-1][mask])
            loss = torch.nn.functional.cross_entropy(logits.float(), labels[:, 1:][mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += int(batch["attention_mask"].sum())
        examples_seen += batch["input_ids"].shape[0]
        window.append(float(loss))
        if step % 20 == 0 or step == total_steps:
            elapsed = time.perf_counter() - started
            record = {
                "step": step,
                "loss": sum(window) / len(window),
                "lr": lr_at(step - 1),
                "examples": examples_seen,
                "tokens_per_s": tokens_seen / elapsed,
                "examples_per_s": examples_seen / elapsed,
                "elapsed_s": elapsed,
                "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            window = []
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 100 == 0 or step == total_steps:
                eta = elapsed / step * (total_steps - step) / 3600
                print(
                    f"step {step}/{total_steps} loss {record['loss']:.4f} "
                    f"{record['tokens_per_s']:.0f} tok/s {record['examples_per_s']:.2f} ex/s "
                    f"mem {record['max_mem_gb']:.1f}G eta {eta:.2f}h",
                    flush=True,
                )
        if step in save_steps:
            ckpt = out_dir / f"frac{save_steps[step]}"
            model.save_pretrained(ckpt)
            saved[str(ckpt)] = file_sha256(ckpt / "adapter_model.safetensors")
            print(f"saved {ckpt}", flush=True)

    elapsed = time.perf_counter() - started
    metadata = {
        "name": args.name,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_model": model_id,
        "lora_targets": LORA_TARGETS,
        "trainable_params": n_trainable,
        "n_examples": len(hf_train),
        "steps": total_steps,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "gpu_hours": elapsed / 3600,
        "gpu": torch.cuda.get_device_name(0),
        "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
        "checkpoints_sha256": saved,
        "training_only_bounds": {
            "image_max_tokens": None if args.text_only else args.image_max_tokens,
            "desc_max_chars": args.desc_max_chars,
        },
    }
    (results_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"TRAINING DONE in {elapsed / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()

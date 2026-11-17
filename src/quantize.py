"""Quantize the merged fine-tuned model with llm-compressor.

Schemes:
  fp8   : FP8_DYNAMIC (weights FP8, dynamic per-token activation quantization);
          needs no calibration data.
  w4a16 : GPTQ, 4-bit weights / 16-bit activations, calibrated on the frozen
          `calib` split (256 training examples) rendered in the training format.

In both, `lm_head` and every vision module (`visual.*`: vision tower, merger,
deep-stack mergers) are left unquantized. The recipe follows llm-compressor's
examples/multimodal_vision/qwen3_vl_example.py (v0.15.0).

Usage: python -m src.quantize --model <merged_dir> --scheme fp8|w4a16 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import load_config, load_split
from .train_sft import PIXELS_PER_TOKEN, target_json, to_chat

IGNORE = ["re:.*lm_head", "re:.*visual.*"]


def build_calibration_set(processor, args: argparse.Namespace):
    hf_calib, _ = load_split("calib", load_config())

    def preprocess(example):
        messages, image = to_chat(
            example,
            text_only=False,
            desc_max_chars=args.desc_max_chars,
            answer=target_json(example),
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return processor(
            text=[text],
            images=[image],
            padding=False,
            images_kwargs={"max_pixels": args.image_max_tokens * PIXELS_PER_TOKEN},
        )

    return hf_calib.map(preprocess, remove_columns=hf_calib.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scheme", required=True, choices=["fp8", "w4a16"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-max-tokens", type=int, default=2048)
    parser.add_argument("--desc-max-chars", type=int, default=6000)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    args = parser.parse_args()

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)
    started = time.perf_counter()

    if args.scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
        oneshot(model=model, recipe=recipe)
        calibration = None
    else:
        dataset = build_calibration_set(processor, args)
        calibration = len(dataset)

        def data_collator(batch):
            assert len(batch) == 1
            return {key: torch.tensor(value) for key, value in batch[0].items()}

        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)
        oneshot(
            model=model,
            tokenizer=str(args.model),
            dataset=dataset,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=len(dataset),
            data_collator=data_collator,
            sequential_targets=["Qwen3VLTextDecoderLayer"],
        )

    model.save_pretrained(args.out, save_compressed=True)
    processor.save_pretrained(args.out)
    quantized = sum(
        1 for _, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None
    )
    info = {
        "source_model": str(args.model),
        "scheme": args.scheme,
        "ignore": IGNORE,
        "calibration_samples": calibration,
        "quantized_modules": quantized,
        "seconds": time.perf_counter() - started,
        "size_gb": sum(f.stat().st_size for f in args.out.glob("*.safetensors")) / 2**30,
    }
    (args.out / "quantize_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

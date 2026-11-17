"""Merge a LoRA adapter into the base model and save full bf16 weights.

usage: python scripts/merge_lora.py <adapter_dir> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config  # noqa: E402


def main() -> None:
    adapter_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    model_id = load_config()["model"]["repo_id"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    assert not any("lora" in name for name, _ in model.named_parameters())
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(model_id).save_pretrained(out_dir)
    (out_dir / "merge_info.json").write_text(
        json.dumps({"base_model": model_id, "adapter": str(adapter_dir)}, indent=2)
    )
    print(f"merged {adapter_dir} -> {out_dir}")


if __name__ == "__main__":
    main()

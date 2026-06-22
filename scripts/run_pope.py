#!/usr/bin/env python3
"""Standalone POPE evaluation.

Why this exists: the built-in `pope_eval.py` CLI calls
`pope_eval(..., adapter, args.max_samples)`, but `pope_eval()` expects
`(..., model, processor, max_samples)`. So the adapter lands in the `model`
slot and `max_samples` in the `processor` slot, and the CLI crashes.
This wrapper builds (model, processor) correctly, adds LoRA support, and
calls the (correct) `pope_eval()` from the repo.

Usage
-----
  # Base model
  python scripts/run_pope.py \
      --model qwen \
      --pope-root /path/to/POPE/coco \
      --coco-images-root /path/to/COCO/images/val2014 \
      --split popular --max-samples 500 --cache-path PathtoCache

  # After LoRA fine-tuning (point at the adapter dir saved by finetune.py)
  python scripts/run_pope.py --model qwen \
      --pope-root /path/to/POPE/coco \
      --coco-images-root /path/to/COCO/images/val2014 \
      --split popular --lora-path ./sft_out/checkpoint-XXXX
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from utils import get_model
from pope_eval import pope_eval

# CLI name -> utils.get_model name
NAME_MAP = {
    "qwen": "qwen",
    "llava": "llava",
    "llama": "llama",
    "glm": "glm4.1v-thinking",
    "glm4.1v-thinking": "glm4.1v-thinking",
    "pali": "pali",
}


def maybe_cuda(model):
    """Move to CUDA unless HF already dispatched it (device_map='auto')."""
    if torch.cuda.is_available() and not getattr(model, "hf_device_map", None):
        return model.to("cuda")
    return model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(NAME_MAP.keys()))
    ap.add_argument("--pope-root", required=True, help="dir with coco_pope_{split}.json")
    ap.add_argument("--coco-images-root", required=True, help="root so item['image'] resolves")
    ap.add_argument("--split", default="popular", choices=["random", "popular", "adversarial"])
    ap.add_argument("--cache-path", default="cache")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--lora-path", default=None, help="optional LoRA adapter directory")
    args = ap.parse_args()

    model, processor = get_model(NAME_MAP[args.model], cache_path=args.cache_path)
    model.eval()
    model = maybe_cuda(model)

    if args.lora_path:
        from peft import PeftModel
        from transformers import AutoProcessor

        model = PeftModel.from_pretrained(model, args.lora_path)
        model = maybe_cuda(model)
        try:
            processor = AutoProcessor.from_pretrained(args.lora_path)
        except Exception:
            pass  # fall back to the base processor
        print(f"LoRA adapter loaded: {args.lora_path}")

    stats = pope_eval(
        args.pope_root, args.coco_images_root, args.split, model, processor, args.max_samples
    )

    print(f"\nPOPE [{args.split}] on {args.model}  n={stats['n']}")
    print(f"Accuracy:  {stats['accuracy']:.4f}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall:    {stats['recall']:.4f}")
    print(f"F1:        {stats['f1']:.4f}   (primary metric in paper)")
    print(f"Yes-rate:  {stats['yes_rate']:.4f}")


if __name__ == "__main__":
    main()

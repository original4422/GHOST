#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
from collections import Counter
from utils import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# ----------------------------
# 1) Data loader (official POPE format)
# ----------------------------
def load_pope_items(pope_root: Path, split: str) -> List[Dict]:
    """
    Expects the official RUCAIBox/POPE repo layout.
    Typical files:
      data/coco/pope_random.json
      data/coco/pope_popular.json
      data/coco/pope_adversarial.json
    Each entry usually has fields like:
      {"image": "val2014/COCO_val2014_000000123456.jpg",
       "question": "Is there a ... in the image?",
       "answer": "yes"/"no"}
    """
    split = split.lower()
    fn = {
        "random": "coco_pope_random.json",
        "popular": "coco_pope_popular.json",
        "adversarial": "coco_pope_adversarial.json",
    }.get(split)
    if fn is None:
        raise ValueError(f"Unknown split: {split}")

    fpath = os.path.join(pope_root, fn)
    with open(fpath, "r") as f:
        items = [json.loads(q) for q in f]
    return items

# ----------------------------
# 2) Normalization utilities
# ----------------------------
YES_PAT = re.compile(r"\b(yes|yeah|yep|y|true)\b", re.I)
NO_PAT  = re.compile(r"\b(no|nope|n|false)\b", re.I)

def normalize_yn(text: str) -> str:
    text = (text or "").strip()
    if YES_PAT.search(text):
        return "yes"
    if NO_PAT.search(text):
        return "no"
    # fallback: greedy heuristic (some LVLMs answer with a sentence)
    t = text.lower()
    if "yes" in t and "no" not in t: return "yes"
    if "no" in t and "yes" not in t: return "no"
    # if still ambiguous, default to "no" (conservative) — change if you prefer
    return "no"

# ----------------------------
# 3) MODEL ADAPTERS
#    Plug your models here. Must return "yes" or "no".
# ----------------------------
class BaseAdapter:
    def answer(self, image: Image.Image, question: str) -> str:
        raise NotImplementedError

# ---- Example: Qwen2.5-VL-7B via HF Transformers (adjust to your env) ----
# Note: comment out if you don’t have the model locally; keep as template.
class QwenAdapter(BaseAdapter):
    def __init__(self, lora = False,lorapath="./Finetuned_qwen/lora-finetuned-best",device="cuda"):#"./Finetuned_qwen/qwen-lora-finetuned-manual3"
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model, self.proc = get_model('qwen',cache_path='cache') 
        if lora:
            from peft import PeftModel
            self.loraname = lorapath
            self.model = PeftModel.from_pretrained(self.model, self.loraname, torch_dtype=torch.float16)
            self.model.to(device)
            self.proc = AutoProcessor.from_pretrained(self.loraname)

            print("✅ LoRA weights loaded")   
        self.model.to(device)
    def answer(self, image: Image.Image, question: str) -> str:
        from transformers import AutoProcessor  # keep local to avoid import errors if unused
        text = get_vllm_output(self.model, self.proc, question + " Answer yes or no.", image, max_new_tokens=8)
        return normalize_yn(text)

# ---- Example: LLaVA-1.6 adapter (sketch) ----
class LlavaAdapter(BaseAdapter):
    def __init__(self, device="cuda"):
        self.model, self.proc = get_model('llava',cache_path='cache')
        self.model.to(device)
        
    def answer(self, image: Image.Image, question: str) -> str:
        text = get_vllm_output(self.model, self.proc, question + " Answer yes or no.", image, max_new_tokens=8)
        return normalize_yn(text)

# ---- Example: GLM-4V adapter (sketch) ----
class GLM4VAdapter(BaseAdapter):
    def __init__(self,device="cuda"):
        self.model, self.proc = get_model('glm4.1v-thinking',cache_path='cache')
        self.model.to(device)
    def answer(self, image: Image.Image, question: str) -> str:
        text = get_vllm_output(self.model, self.proc, question + " Answer yes or no.", image, max_new_tokens=128)
        return normalize_yn(text)

class PaliGemmaAdapter(BaseAdapter):
    def __init__(self, device="cuda"):
        self.model_id = "google/paligemma-3b-mix-224"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            revision="bfloat16",
            token=hf_token,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)
        self.model.to(device)
    def answer(self, image: Image.Image, question: str) -> str:
        model_inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=12, do_sample=False)
            generation = generation[0][input_len:]
            text = self.processor.decode(generation, skip_special_tokens=True)
            print(text)
        return normalize_yn(text)

ADAPTERS = {
    "qwen": QwenAdapter,
    "llava": LlavaAdapter,
    "glm": GLM4VAdapter,
    'pali': PaliGemmaAdapter,
}

# ----------------------------
# 4) Evaluation
# ----------------------------
def pope_eval(pope_root: Path, coco_images_root: Path, split: str, model, processor, max_samples: int = -1) -> Dict[str, float]:
    items = load_pope_items(pope_root, split)
    if max_samples > 0:
        items = items[:max_samples]

    y_true, y_pred = [], []
    yes_count = 0

    for ex in tqdm(items):
        #print(f"Processing {ex['question_id']} ...")
        img_path = os.path.join(coco_images_root, ex["image"])
        img = Image.open(img_path).convert("RGB")
        question = ex["text"].strip()
        gt = ex["label"].strip().lower()  # "yes" or "no"

        response = get_vllm_output(model, processor, question + " Answer yes or no.", img, max_new_tokens=8)
        pred = normalize_yn(response)
        #print(pred)
        y_true.append(1 if gt == "yes" else 0)
        y_pred.append(1 if pred == "yes" else 0)
        if pred == "yes":
            yes_count += 1

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    yes_rate = float(yes_count) / len(items)

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "yes_rate": yes_rate,
        "n": len(items)
    }

# ----------------------------
# 5) CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="POPE baseline evaluator")
    ap.add_argument("--pope_root", type=Path, required=True, help="Path to RUCAIBox/POPE repo")
    ap.add_argument("--coco_images_root", type=Path, required=True, help="Root to COCO images (so ex['image'] resolves)")
    ap.add_argument("--split", type=str, default="random", choices=["random","popular","adversarial"])
    ap.add_argument("--model", type=str, required=True, choices=list(ADAPTERS.keys()))
    ap.add_argument("--max_samples", type=int, default=-1, help="Debug: limit eval size")
    ap.add_argument("--lora",type=bool,default=False,help="Use LoRA weights (only for QwenAdapter)")
    args = ap.parse_args()
    
    adapter = ADAPTERS[args.model](args.lora) if args.model == "qwen" else ADAPTERS[args.model]()
    stats = pope_eval(args.pope_root, args.coco_images_root, args.split, adapter, args.max_samples)

    print(f"POPE [{args.split}] on {args.model}  n={stats['n']}")
    print(f"Accuracy:  {stats['accuracy']:.4f}")
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall:    {stats['recall']:.4f}")
    print(f"F1:        {stats['f1']:.4f}   (primary metric in paper)")
    print(f"Yes-rate:  {stats['yes_rate']:.4f}")

if __name__ == "__main__":
    main()
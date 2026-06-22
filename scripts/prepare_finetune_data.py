#!/usr/bin/env python3
"""Prepare data directories consumed by finetune.py.

finetune.py expects per-object subfolders:  <root>/<object_name>/*.png
This helper builds them, closing two gaps the repo leaves manual:

  neg : reorganize GHOST attack outputs
        (logs/attack/<model>/<object>_*/images/*.png)
        into  <out>/<object>/*.png   -> use as --neg_images_dir

  pos : sample real COCO images that CONTAIN each object
        into  <out>/<object>/*.jpg   -> use as --pos_images_dir

Examples
--------
  # Negatives from a Qwen attack run
  python scripts/prepare_finetune_data.py neg \
      --attack-root logs/attack/qwen \
      --out-dir data/finetune/neg \
      --classes "traffic light,carrot,toilet,knife,bottle,vase,clock,bus,boat,suitcase"

  # Positives sampled from COCO train split
  python scripts/prepare_finetune_data.py pos \
      --coco-path /path/to/COCO \
      --out-dir data/finetune/pos \
      --classes "traffic light,carrot,toilet,knife,bottle,vase,clock,bus,boat,suitcase" \
      --n-per-class 150

Add --link to create symlinks instead of copying (saves disk space).
"""
import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def parse_classes(s: str):
    p = Path(s)
    if p.exists() and p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [c.strip() for c in s.split(",") if c.strip()]


def place(src: Path, dst: Path, link: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def do_neg(args):
    classes = parse_classes(args.classes)
    out = Path(args.out_dir)
    total = 0
    for cls in classes:
        # Attack run folders are named "<object>_lr=...": match by prefix.
        run_dirs = sorted(glob.glob(os.path.join(args.attack_root, f"{cls}_*", "images")))
        n = 0
        for images_dir in run_dirs:
            for img in sorted(glob.glob(os.path.join(images_dir, "*"))):
                if img.lower().endswith(IMG_EXTS):
                    place(Path(img), out / cls / Path(img).name, args.link)
                    n += 1
        print(f"[neg] {cls}: {n} image(s) from {len(run_dirs)} run(s)")
        total += n
    if total == 0:
        print(f"[neg] WARNING: 0 images collected. Check --attack-root ('{args.attack_root}') "
              f"and that runs were named '<object>_...'.")
    print(f"[neg] done: {total} image(s) -> {out}")


def do_pos(args):
    from data import COCO

    classes = parse_classes(args.classes)
    out = Path(args.out_dir)
    dset = COCO(args.coco_path, split=args.split)
    total = 0
    for cls in classes:
        ids = dset.get_imgIds_by_class(present_classes=[cls])
        random.shuffle(ids)
        ids = ids[: args.n_per_class]
        n = 0
        for img_id in ids:
            file_name = dset.im_dict[img_id]["file_name"]
            src = Path(dset.image_dir) / file_name
            if src.exists():
                place(src, out / cls / src.name, args.link)
                n += 1
        print(f"[pos] {cls}: {n} image(s)")
        total += n
    print(f"[pos] done: {total} image(s) -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("neg", help="reorganize attack outputs into <out>/<object>/")
    a.add_argument("--attack-root", required=True, help="e.g. logs/attack/qwen")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--classes", required=True, help="comma list or a file with one class per line")
    a.add_argument("--link", action="store_true", help="symlink instead of copy")
    a.set_defaults(func=do_neg)

    b = sub.add_parser("pos", help="sample COCO positives into <out>/<object>/")
    b.add_argument("--coco-path", required=True, help="COCO root (expects images/<split>2017 + annotations)")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--classes", required=True, help="comma list or a file with one class per line")
    b.add_argument("--n-per-class", type=int, default=150)
    b.add_argument("--split", default="train", choices=["train", "val"])
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--link", action="store_true", help="symlink instead of copy")
    b.set_defaults(func=do_pos)

    args = ap.parse_args()
    random.seed(getattr(args, "seed", 42))
    args.func(args)


if __name__ == "__main__":
    main()

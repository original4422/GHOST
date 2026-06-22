# GHOST 代码库 Q&A

> 围绕"这套代码库怎么用"的问答记录。完整可执行流程见 [`MAP.human.md`](./MAP.human.md)（人看）/ [`MAP.agent.md`](./MAP.agent.md)（agent 看）。

---

## Q1：当前代码库使用哪些数据集？数据集如何准备，是自行准备还是 repo 提供脚本？

**结论：需要自己下载，repo 只提供读取这些标准数据集的 Dataset 类（`data.py`），不提供下载/准备脚本。**（README 也写明 "code and instructions will be updated soon"。）

| 数据集 | 用在哪 | 期望目录结构 |
|---|---|---|
| **COCO 2017**（主力） | 阶段1训练、阶段2攻击、部分评测 | `COCO/images/{train,val}2017/`、`COCO/annotations/instances_*.json`、`captions_*.json` |
| **POPE**（RUCAIBox/POPE） | 阶段3 POPE 评测、阶段4微调中的 callback | `coco_pope_{random,popular,adversarial}.json` + 对应 COCO 图 |
| **ObjectNet** | `evals/transfer_eval.py` 等迁移评测（可选） | `{root}/images/<类folder>/`、`{root}/mappings/folder_to_objectnet_label.json` |
| **正/负样本目录** | 阶段4 LoRA 微调 | `root/<物体名>/*.png`（负样本=生成的幻觉图；正样本=真实含该物体的图） |

`COCO` 类（`data.py`）期望：

```
COCO/
  images/train2017/*.jpg   images/val2017/*.jpg
  annotations/instances_train2017.json  captions_train2017.json (val 同理)
```

⚠️ **目录不一致坑**：阶段1 `train_projector.py` 用的是 `ImageDataset(coco_dir/"train2017")`（直接在 `train2017/` 下找 `*.jpg`），而 `COCO` 类要求 `images/train2017/`。准备数据时需建软链，详见 MAP 文档。

---

## Q2：是直接 prompt 文生图，还是基于数据集中的图生图？怎么生的（模型、代码、用法）？

**结论：基于数据集里的真实图（类似 img2img），不是纯 prompt 文生图。** 文本 prompt 只是辅助条件（且用的是"图里有没有 {obj}"的提问模板），真正主导生成的是 **原图的 VAE latent + 对抗优化后的 CLIP 图像嵌入**。

**生图模型**：Stable Diffusion 2.1 **unCLIP** 版（`stabilityai/stable-diffusion-2-1-unclip`，`utils.py: get_diffusion_model`）。unCLIP 接受 CLIP 图像嵌入作为条件，这是整个方法的关键。

**流程（`main.py` 的 `attack()`）：**
1. 选一张**不含目标物体**的真实 COCO 图（`get_imgIds_by_class(present=同超类, absent=[target])`）。
2. 编码成 CLIP 图像嵌入并设为可优化参数（`clip_emb = nn.Parameter(...); requires_grad=True`）。
3. 把原图编码成 VAE latent（`get_vae_features`）作为生成的结构起点。
4. **对抗优化循环**（`--num_steps`，默认 40）：`clip_emb → projector → 受害模型图像token → 跑受害 VLM`，损失：
   `loss = -log P("Yes") + λ_contrast·cos(clip_emb, 目标物体文本嵌入) + λ_reg·MSE(clip_emb, 初始)`，
   反传只更新 `clip_emb`，把嵌入推向"让模型回答 Yes"。
5. 当 `P(Yes) > --threshold`（默认 0.99）时，用 `generate_image(pipe, vae, clip_emb, ...)` 把对抗后的 CLIP 嵌入解码成真实图片：对原图 latent 加噪（`--t` 控制强度，默认 0≈几乎保留原图）→ `pipe._encode_image(image_embeds=对抗clip_emb)` 作 unCLIP 图像条件 → 标准去噪 + CFG（`--guidance_scale` 默认 10）→ VAE 解码出 PIL 图。
6. 生成后用 **OWLv2** 检测器复核（确认生成图里物理上确实没有该物体），通过才算攻击成功，存到 `logs/attack/{model}/{run}/images/`。

**用法（需先有阶段1的 projector）：**
```bash
python main.py --model_name llava \
  --projector_path ./projector/models/<ckpt含 context_dim/hidden_dim>.pt \
  --data_path /path/to/COCO --cache_path /path/to/hf_cache \
  --target_object vase --num_generation 4 --num_steps 40 --threshold 0.99 \
  --guidance_scale 10 --t 0 --num_of_inference 50
```

⚠️ 坑：① 原代码硬编码只跑第 200~400 张图（已改为 `--start_index/--end_index` 参数，见 MAP）；② projector 的 `context_dim/hidden_dim` 是从 checkpoint **文件名正则**抠出来的，别给 checkpoint 改名。

---

## Q3：如何用生成的图做 LoRA fine-tuning？用法是什么？

`finetune.py` 做的是 **QLoRA SFT**（4-bit 量化底模 + LoRA，基于 `peft` + `trl.SFTTrainer`）。本质：把"生成的幻觉图"当作答案为 **"No"** 的样本，教 VLM 不被骗（鲁棒化/防御）。

**数据组织（`get_dataset_list`）：**
- `--neg_images_dir`：GHOST 生成的幻觉图，标签 "No"（每类最多 150 张）。
- `--pos_images_dir`：真实含该物体的图，标签 "Yes"。
- 两者目录都必须是 `根目录/<物体名>/*.png`；每张图被构造成对话（user 问有没有 obj，assistant 答 Yes/No）。

**LoRA 配置**：4-bit nf4；Qwen 用 `target_modules="all-linear"` + 保存 `lm_head/embed_tokens`，其它用 `q/k/v/o_proj` + 保存 `mm_projector`；`r/alpha/dropout` 由命令行传入。训练中 `CustomEvalCallback` 在开始和每 epoch 末跑 POPE 评测。

**用法：**
```bash
python finetune.py \
  --neg_images_dir data/finetune/neg --pos_images_dir data/finetune/pos \
  --output_dir ./sft_out --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --r 16 --alpha 16 --dropout 0.05 \
  --lr 1e-4 --epochs 1 --batch_size 1 --gradient_accumulation_steps 8 --logging_steps 20 \
  --pope_root /path/to/POPE --coco_path /path/to/COCO --split popular --pope_max_samples 500 \
  --seed 42 --wandb_run_name ghost-sft
```
（所有参数 `required=True`，必须全给；强依赖 `wandb`。）

⚠️ **衔接缺口**：阶段2 生成图是扁平的 `logs/attack/{model}/{run}/images/{i}_{imgid}_{step}_{g}.png`（一次 run = 一个 `--target_object`），而阶段4 要求 `neg/<物体>/*.png`。中间重组步骤 repo 不提供，已补 `scripts/prepare_finetune_data.py`（neg 重组 + pos 从 COCO 采样）。`data.py` 里硬编码的 10 个实验物体：`traffic light, carrot, toilet, knife, bottle, vase, clock, bus, boat, suitcase`。

---

## 一句话串起来

下载 **COCO**（+ POPE/ObjectNet 评测用）→ `train_projector.py` 训 **projector** → `main.py` 对每个物体跑攻击、用 **SD2.1-unCLIP** 把对抗 CLIP 嵌入解码成**幻觉图** → `scripts/prepare_finetune_data.py` 把图按物体归档 → `finetune.py` 做 **QLoRA** 并跑 POPE。

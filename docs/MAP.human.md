# GHOST 全流程使用指南（人看版）

> GHOST = *Hallucination-Inducing Image Generation for Multimodal LLMs*。本指南带你从零跑通整套代码库。
> 需要"可被 AI 自主执行"的精简版见 [`MAP.agent.md`](./MAP.agent.md)；之前的问答见 [`QA.md`](./QA.md)。
>
> 文中 `⚠️` 是 repo 里的坑/注意点，第 10 节有集中汇总。

---

## 0. 这套代码在做什么

目标：生成一张**人眼/检测器都看不出含某个物体**、却能**骗多模态大模型（VLM）说"有"**的图片（诱导幻觉），并进一步用这些图来评估、甚至微调防御 VLM。

四个阶段（有先后依赖）：

```
准备：环境 → 模型 → 数据
  │
阶段1  train_projector.py   训练 projector：把 CLIP 图像嵌入映射成"受害 VLM 的图像 token"
  │
阶段2  main.py             对抗优化 CLIP 嵌入 + SD2.1-unCLIP 解码 → 幻觉图
  │                        产物：logs/attack/<model>/<run>/{images,original,failed}/
  │
阶段3  评估                攻击成功率/迁移性、图像质量(FID/SSIM)、POPE 基准
  │
阶段4  finetune.py         用生成的幻觉图做 QLoRA 微调（防御），再评估
```

> **术语「受害模型 / victim model」**：就是**被攻击的那个多模态大模型（VLM）**，即 `--model_name` 指定的 `llava/qwen/llama/glm`。GHOST 的目标就是骗它产生幻觉，所以叫"受害"。这个词直接来自代码——`main.py` 里 `--model_name` 的 help 写的是 "Name of the victim model"。下文所有「受害 VLM」均指此。它在攻击中**全程冻结、不更新权重**；被优化的只有 CLIP 嵌入。

核心直觉：unCLIP 扩散模型可以"按一个 CLIP 图像嵌入"生成图。先把一张不含目标物体的真实图编码成 CLIP 嵌入，对它做对抗优化让 VLM 误判为"有"，再把这个被污染的嵌入解码回真实图片——于是得到一张看似无害、却能触发幻觉的图。

---

## 1. 环境与依赖

**硬件**：需要 NVIDIA GPU（代码大量 `.cuda()` 硬编码，基本要求 CUDA 环境）。
- 阶段1/2 单进程会**同时**加载 受害 VLM(7B~11B) + CLIP ViT-H(~2.5GB) + SD-unCLIP(~2.5GB) + OWLv2(~1.5GB)，单次峰值约 **20GB(7B) ~ 30GB(11B llama)**。
- 阶段4 用 4-bit QLoRA，单卡 7B 约 **6~16GB**。
- 上述为经验估计，实际取决于模型与分辨率。

> **你的环境：A100-80GB × 最多 8 卡。** 单张 80GB 跑任一单次任务（含 11B llama 的整套攻击栈 ~30GB）都**毫无压力、留有大量余量**，所以**显存完全不是瓶颈，重点是用 8 卡并行提吞吐**。具体策略见下面 1.5 节。
> 余量大到可以**一张卡叠多个进程**：7B 攻击栈 ~20GB，80GB 可稳跑 2~3 个/卡；11B ~30GB，可跑 2 个/卡。若目标物体多、想更快，可据此把并发拉到 16~24（见 1.5 节进阶写法）。

**安装依赖**（repo 原本没有 `requirements.txt`，已补）：
```bash
pip install -r requirements.txt
```
> ⚠️ 最敏感的是 `transformers` 版本：Qwen2.5-VL / GLM-4.1V-Thinking / Llama-3.2-Vision 需要较新版本。若出现 `Qwen2_5_VLForConditionalGeneration` / `Glm4vForConditionalGeneration` 导入失败，先升级 `transformers`。

**账号/密钥**：
- **HF token**：`meta-llama/Llama-3.2-11B-Vision-Instruct` 和 `google/paligemma-3b-mix-224` 是 gated 模型，需在 HuggingFace 申请权限并 `huggingface-cli login`（或设 `HF_TOKEN`）。
- **OpenAI key**：仅 `evals/eval_projector_gpt.py`（GPT-as-judge）需要 `OPENAI_API_KEY`。
- **wandb**：`finetune.py` 会 `wandb.init()`。`wandb login` 或设 `WANDB_MODE=offline` 跑离线。

**HF 缓存目录**：几乎所有脚本都有 `--cache_path`/`--cache-dir` 参数，模型会下载到这里。建议统一指向一个大磁盘目录，例如 `/data/hf_cache`。

---

## 1.5 多 GPU 利用（A100 × 8）

关键认知：本库的**单次**训练/攻击基本是**单卡程序**（大量 `.cuda()` 写死在 `cuda:0`），并没有内置的多卡数据并行。所以 8 卡的正确用法不是"让一次任务更快"，而是**同时跑多个单卡任务**——用 `CUDA_VISIBLE_DEVICES` 给每个进程分一张卡。

| 阶段 | 单次是否单卡 | 8 卡怎么用 |
|---|---|---|
| 1 训练 projector | 是 | 一次只需 1 卡。要并行就**每张卡训一个不同 `--model_name` 的 projector**（llava/qwen/llama/glm 各一张卡）。 |
| 2 生成幻觉图 | 是 | **收益最大**：攻击是逐物体的，把每个 `--target_object` 派到不同卡，8 个并发（见第 6 节脚本）。 |
| 3 评估 | 是 | 每个模型/每个 split 一张卡并行；注意 `transfer_eval.py` 的 cache 硬编码（坑 #8）。 |
| 4 LoRA 微调 | 默认单卡 | 最简单：`CUDA_VISIBLE_DEVICES=0` 跑（4-bit 7B 一张卡够），其余卡留给并行评估。要真正多卡加速需改成 DDP（见第 8 节说明）。 |

**给每个进程绑定一张卡的通用写法**：
```bash
CUDA_VISIBLE_DEVICES=3 python <script> ...   # 该进程只看得到 GPU3，内部的 cuda:0 即物理 GPU3
```

**进阶（80GB 专属）：一张卡叠多个进程**，把并发从 8 提到 16~24。思路是给每个任务算一个"卡号 = 任务序号 % 8"，同一张卡上会落多个进程：
```bash
OBJS=(traffic_light carrot toilet knife bottle vase clock bus boat suitcase)  # 例: 任务数 > 8
PER_GPU=2; NGPU=8; i=0
for obj in "${OBJS[@]}"; do
  CUDA_VISIBLE_DEVICES=$(( i % NGPU )) python main.py ... --target_object "$obj" &
  i=$(( i + 1 ))
  # 凑满 NGPU*PER_GPU 个在跑就等一批
  [ $(( i % (NGPU * PER_GPU) )) -eq 0 ] && wait
done
wait
```
> 80GB 下 7B 叠 2~3 个/卡、11B 叠 2 个/卡是安全的；若报 OOM 就把 `PER_GPU` 调回 1。

> ⚠️ 11B 的 `llama` 和 `glm` 在 `utils.get_model` 里用了 `device_map="auto"`：若该进程能看到多张卡，模型会被**切分到多卡**（模型并行，慢且占多卡）。想让它老老实实single-GPU，就用 `CUDA_VISIBLE_DEVICES=<单卡>` 把可见卡限制成 1 张。7B 的 llava/qwen 没有用 device_map，本就单卡。

---

## 2. 模型准备

所有模型都由 HuggingFace / open_clip **自动下载**到 `--cache_path`，无需手动下权重（gated 模型除外，需先申请）。

| 角色 | 模型 ID | 触发处 |
|---|---|---|
| 受害 VLM `llava` | `llava-hf/llava-v1.6-mistral-7b-hf` | `utils.get_model` |
| 受害 VLM `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | 同上 |
| 受害 VLM `llama`（gated） | `meta-llama/Llama-3.2-11B-Vision-Instruct` | 同上 |
| 受害 VLM `glm4.1v-thinking` | `THUDM/GLM-4.1V-9B-Thinking` | 同上 |
| 受害 VLM `pali`（gated，仅评测） | `google/paligemma-3b-mix-224` | 同上 |
| CLIP 编码器 | `ViT-H-14` / `laion2b_s32b_b79k`（open_clip） | `utils.get_clip_model` |
| 生图模型 | `stabilityai/stable-diffusion-2-1-unclip` | `utils.get_diffusion_model` |
| 物体检测器 | `google/owlv2-base-patch16-ensemble` | `utils.get_owl_model` |

> 不同 VLM 的图像 token 数/维度不同（`utils.get_num_tokens`）：llava=1176/4096、qwen=144/3584、glm=144/4096、llama=6404/4096。projector 的输出形状由它决定，所以 **projector 与受害模型一一对应**，不能混用。

---

## 3. 数据集准备

repo 只提供读取标准数据集的 Dataset 类，**不提供下载脚本**，需自行准备。

### 3.1 COCO 2017（必需）
从 [cocodataset.org](https://cocodataset.org/) 下 train2017 / val2017 图片与 annotations，组织成 `COCO` 类期望的结构：
```
COCO/
  images/train2017/*.jpg
  images/val2017/*.jpg
  annotations/instances_train2017.json
  annotations/captions_train2017.json
  annotations/instances_val2017.json
  annotations/captions_val2017.json
```

> ⚠️ **目录约定冲突**：`main.py` 等用 `COCO` 类（要 `images/train2017/`），但 `train_projector.py` 用 `ImageDataset(coco_dir/"train2017")`（要 `train2017/` 直接放图）。最省事的兼容做法是建软链：
> ```bash
> cd /path/to/COCO
> ln -s images/train2017 train2017
> ln -s images/val2017 val2017
> ```
> 这样 `--coco_dir /path/to/COCO` 和 `--data_path /path/to/COCO` 两种脚本都能用。

### 3.2 POPE（阶段3/4 评测用）
克隆 [RUCAIBox/POPE](https://github.com/RUCAIBox/POPE)，使用其中的 `coco_pope_{random,popular,adversarial}.json`。
> ⚠️ POPE 基于 COCO **val2014**，json 里的 `image` 字段形如 `COCO_val2014_000000xxxxxx.jpg`。所以 POPE 评测时 `--coco-images-root` 要指向你的 **val2014 图片目录**，而不是 val2017。

### 3.3 ObjectNet（可选，迁移评测）
从 [objectnet.dev](https://objectnet.dev/) 下载，组织成 `{root}/images/<类folder>/` + `{root}/mappings/folder_to_objectnet_label.json`（`data.ObjectNet` 期望）。

### 3.4 微调用的正/负样本目录（阶段4，见第 8 节）
finetune 需要 `根目录/<物体名>/*.png` 结构。负样本=阶段2生成图，正样本=真实含该物体的图。两者都用 `scripts/prepare_finetune_data.py` 生成，无需手工摆放。

---

## 4. 阶段1：训练 Projector（`train_projector.py`）

**作用**：学一个 MLP（`TokenMLP`），输入 CLIP 图像嵌入，输出受害 VLM 的图像 token 序列。它让"操作 CLIP 嵌入"等价于"操作 VLM 看到的图像特征"，是阶段2攻击的前置条件。

```bash
python train_projector.py \
  --model_name qwen \
  --coco_dir /path/to/COCO \
  --cache_path /path/to/hf_cache \
  --save_projector_dir ./projector/models \
  --epochs 10 --lr 1e-3 --batch_size 1 \
  --context_dim 1024 --hidden_dim 2048 \
  --coco_subset 0.1
```

**关键参数**：
- `--coco_subset`：用多少比例 COCO 训练（默认 0.1，先小规模验证）。
- `--context_dim` / `--hidden_dim`：projector 结构尺寸。
- `--epochs` / `--lr` / `--scheduler` / `--warmup_steps`：常规训练超参。
- `--use_wandb`：可选，开 wandb 记录。

**产物**：每个 epoch 存一个 checkpoint，文件名形如
`qwen_bs=1_lr=0.001_epochs=10_context_dim=1024_hidden_dim=2048_coco_subset=0.1_epoch_10.pt`，外加一张 loss 曲线 png。

> ⚠️ **文件名编码很重要**：阶段2/评测会用正则从 checkpoint 文件名里读 `context_dim` 和 `hidden_dim`（`re.search(r"context_dim=(\d+)", ...)`）。**不要重命名 checkpoint**，否则会按默认 4096 加载、形状不匹配。
>
> ⚠️ **别用 `--run_test_each_epoch`**：它会去调用一个**不存在的 `test_projector.py`** 而报错。如果确实想每个 epoch 评测，传 `--test_script evals/validate_projector.py`（其参数接口与该处兼容）。

---

## 5. 阶段1.5：验证 Projector（可选）

确认 projector 训得是否合理，再进入攻击。

**A. 数值保真度**（`evals/validate_projector.py`）——看 projector 喂进去后，VLM 在"含/不含物体"的图上能否正确答 Yes/No：
```bash
python evals/validate_projector.py \
  --model_name qwen \
  --projector_path ./projector/models/<ckpt>.pt \
  --data_path /path/to/COCO \
  --cache_path /path/to/hf_cache \
  --target_objects "vase,boat,bird,car"
```
看日志末尾的 `Mean Accuracies for Projector:`（每物体各取前 100 张图）。

**B. 语义保真度（GPT 评判）**（`evals/eval_projector_gpt.py`，需 `OPENAI_API_KEY`）——比较"原图喂模型"vs"projector 喂模型"的描述质量：
```bash
export OPENAI_API_KEY=sk-...
python evals/eval_projector_gpt.py \
  --model_name qwen --projector_path ./projector/models/<ckpt>.pt \
  --coco_path /path/to/COCO --cache_path /path/to/hf_cache --K 30 --P 2
```

---

## 6. 阶段2：生成幻觉图（`main.py`）

对每个**目标物体**跑一次攻击，产出能诱导该物体幻觉的图。

```bash
python main.py \
  --model_name qwen \
  --projector_path ./projector/models/<ckpt>.pt \
  --data_path /path/to/COCO \
  --cache_path /path/to/hf_cache \
  --target_object vase \
  --num_generation 4 --num_steps 40 --threshold 0.99 \
  --guidance_scale 10 --t 0 --num_of_inference 50 \
  --OD_threshold 0.5 \
  --start_index 0 --end_index 200
```

**关键参数**：
- `--target_object`：想诱导出来的物体（须是 COCO 类名）。
- `--num_steps` / `--lr`：对抗优化步数与学习率。
- `--threshold`：P(Yes) 超过它才触发生图（默认 0.99）。
- `--num_generation`：每张图最多生成几张候选。
- `--t`：扩散加噪起点（0≈最大程度保留原图结构）。
- `--guidance_scale` / `--num_of_inference`：扩散 CFG 强度与步数。
- `--OD_threshold`：OWLv2 检测阈值；`<1` 时会加载 OWLv2 复核生成图是否"真的"没有该物体，`≥1` 则跳过检测。
- `--start_index` / `--end_index`：处理候选图的索引区间（**本指南新增**，替代原先硬编码的 200~400）。`--end_index -1` 表示跑到结尾。

**产物**：`logs/attack/<model_name>/<run_name>/` 下
- `images/`：攻击成功的生成图（文件名含原图索引 `{i}_{imgid}_{step}_{g}.png`）；
- `original/`：对应原图；`failed/`：失败样本；`log.txt`：日志。

**批量跑多个物体**（攻击是逐物体的，串行版）：
```bash
for obj in "traffic light" carrot toilet knife bottle vase clock bus boat suitcase; do
  python main.py --model_name qwen --projector_path ./projector/models/<ckpt>.pt \
    --data_path /path/to/COCO --cache_path /path/to/hf_cache \
    --target_object "$obj" --start_index 0 --end_index 200
done
```

**A100 × 8 并行版**（每个物体占一张卡，8 个并发，整体快约 8×）：
```bash
OBJS=("traffic light" carrot toilet knife bottle vase clock bus boat suitcase)
gpu=0
for obj in "${OBJS[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu python main.py --model_name qwen \
    --projector_path ./projector/models/<ckpt>.pt \
    --data_path /path/to/COCO --cache_path /path/to/hf_cache \
    --target_object "$obj" --start_index 0 --end_index 200 &
  gpu=$(( (gpu + 1) % 8 ))
  [ "$gpu" -eq 0 ] && wait      # 每凑满 8 个等这批结束，避免超过 8 卡
done
wait
```
> 每个进程会各自加载一整套模型（~20–30GB），A100 单卡完全放得下；8 进程即占满 8 卡。

> ⚠️ 原代码把处理范围硬编码成 `i in [200,400)`（已改为 `--start_index/--end_index`）。
> ⚠️ checkpoint 文件名正则同第 4 节，不要改名。

---

## 7. 阶段3：幻觉评估

生成图后，从三个角度评估。所有评测脚本都用 `{cls}` **路径模板**定位每个物体的图目录。

### 7.1 攻击成功率 / 迁移性（`evals/transfer_eval.py`）
把生成图喂给某个 VLM，统计它被骗答"yes"的比例（`yes_share`）。可换不同受害模型测迁移性，也可加载 LoRA 测防御后效果。
```bash
python evals/transfer_eval.py \
  --model-type qwen --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --images-dir-template "logs/attack/qwen/{cls}_*/images" \
  --classes "carrot,knife,clock,toilet,boat,suitcase,bottle,vase,bus" \
  --save-dir out_transfer_qwen
# 评测 LoRA 后模型：加 --lora（默认读 ./Finetuned_qwen/lora-finetuned-best）
```
> ⚠️ `transfer_eval.py` 的 `main()` 把 `cache_dir` **硬编码为 `'cache'`**（忽略 `--cache-dir`）。要么把 HF 缓存软链到 `./cache`，要么改这一行。
> ⚠️ `--images-dir-template` 里的 `{cls}` 会被替换；若 run 文件夹名带后缀，用 `*` 通配（脚本对目录做递归 `rglob`）。

### 7.2 图像质量
**FID**（越低越真实，`evals/fid_eval.py`，依赖 `torchmetrics`）：
```bash
python evals/fid_eval.py \
  --classes "boat,bus,vase" \
  --fake-tmpl "logs/attack/qwen/{cls}_*/images" \
  --gen-tmpl  "logs/attack/qwen/{cls}_*/images" \
  --real-source original \
  --orig-tmpl "logs/attack/qwen/{cls}_*/original"
# 或与 COCO val 比：--real-source coco --coco-path /path/to/COCO
```

**SSIM**（与原图的结构相似度，`evals/SSIM-eval.py`，依赖 `scikit-image`）：
```bash
python evals/SSIM-eval.py \
  --classes "boat,bus,vase" \
  --ghost-tmpl "logs/attack/qwen/{cls}_*/images" \
  --orig-tmpl  "logs/attack/qwen/{cls}_*/original" \
  --gen-tmpl   "logs/attack/qwen/{cls}_*/images"
```
> 这两个脚本靠文件名里的数字索引把"生成图↔原图"配对（`extract_index`，如 `12_3456_0_1.png` 取到索引）。所以 `original/` 与 `images/` 必须来自同一次 run。

### 7.3 POPE 基准（`pope_eval.py` → 用 `scripts/run_pope.py`）
POPE 是标准的物体幻觉基准，给 Accuracy/Precision/Recall/F1/Yes-rate。
> ⚠️ **内置 `pope_eval.py` 的命令行入口是坏的**（它把 adapter 传到了 `model` 参数、把 `max_samples` 传到了 `processor` 参数，会崩）。已提供修正版封装：
```bash
python scripts/run_pope.py \
  --model qwen \
  --pope-root /path/to/POPE/coco \
  --coco-images-root /path/to/COCO_val2014 \
  --split popular --max-samples 500 \
  --cache-path /path/to/hf_cache
# 评测 LoRA 微调后模型：加 --lora-path ./sft_out/checkpoint-XXXX
```

---

## 8. 阶段4：LoRA 微调（`finetune.py`）

用生成的幻觉图（标"No"）+ 真实正样本（标"Yes"）做 QLoRA SFT，让模型学会不被骗。

### 8.1 先把数据摆成 `<root>/<物体名>/` 结构
（repo 不提供，已补 `scripts/prepare_finetune_data.py`）
```bash
# 负样本：把阶段2的攻击产物按物体归档
python scripts/prepare_finetune_data.py neg \
  --attack-root logs/attack/qwen \
  --out-dir data/finetune/neg \
  --classes "traffic light,carrot,toilet,knife,bottle,vase,clock,bus,boat,suitcase"

# 正样本：从 COCO 采真实含该物体的图
python scripts/prepare_finetune_data.py pos \
  --coco-path /path/to/COCO \
  --out-dir data/finetune/pos \
  --classes "traffic light,carrot,toilet,knife,bottle,vase,clock,bus,boat,suitcase" \
  --n-per-class 150
```
（加 `--link` 用软链代替复制省磁盘。）

### 8.2 跑微调
```bash
python finetune.py \
  --neg_images_dir data/finetune/neg \
  --pos_images_dir data/finetune/pos \
  --output_dir ./sft_out \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --r 16 --alpha 16 --dropout 0.05 \
  --lr 1e-4 --epochs 1 --batch_size 1 --gradient_accumulation_steps 8 \
  --logging_steps 20 \
  --pope_root /path/to/POPE/coco --coco_path /path/to/COCO \
  --split popular --pope_max_samples 500 \
  --seed 42 --wandb_run_name ghost-sft
```
- 所有参数都是 `required=True`，必须全给。
- 4-bit nf4 量化；Qwen 用 `target_modules="all-linear"`，其它模型用 `q/k/v/o_proj`。
- 训练中会在开始和每个 epoch 末自动跑 POPE 评测（`CustomEvalCallback`）。
- 产物：LoRA adapter 按 epoch 存到 `--output_dir`。

**A100 × 8 下的两种跑法**：
- **推荐（简单）：单卡跑**，把其余卡留给并行评估/生成。4-bit 7B 一张 A100 绰绰有余：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python finetune.py ... # 同上参数
  ```
  > `finetune.py` 用 `device_map="auto"`：若进程能看到 8 张卡，会把这个 4-bit 小模型摊到多卡（模型并行，反而慢）。用 `CUDA_VISIBLE_DEVICES=0` 锁单卡最干净。
- **多卡数据并行（DDP，更快但需改一行代码）**：`device_map="auto"` 与 DDP 不兼容，`torchrun` 会报 *"can't train a model loaded with device_map='auto' in distributed mode"*。要 DDP，需把 `finetune.py: get_model_and_processor` 里的 `device_map="auto"` 改成按进程绑定：
  ```python
  from accelerate import PartialState
  device_map = {"": PartialState().process_index}   # 每个 rank 各自满载一份 4-bit 模型
  ```
  然后用 `accelerate launch --multi_gpu --num_processes 8 finetune.py ...`（或 `torchrun --nproc_per_node 8`）。**这是行为性改动，我没有默认替你改**——需要的话告诉我，我来打这个补丁。

> ⚠️ **POPE 路径坑**：`finetune.py` 的内置 POPE callback 把图片根目录硬编码成 `{coco_path}/images/train2017`，而标准 POPE 用的是 val2014。若你的 POPE json 指向 val2014，callback 会找不到图。两种解法：(a) 用与 POPE json 匹配的 COCO 版本；(b) 微调时关注训练 loss，POPE 用第 7.3 节的 `run_pope.py` 单独、正确地评测。
> ⚠️ 强依赖 `wandb`：不想联网就 `export WANDB_MODE=offline`。

### 8.3 微调后再评估
```bash
python scripts/run_pope.py --model qwen --lora-path ./sft_out/checkpoint-XXXX \
  --pope-root /path/to/POPE/coco --coco-images-root /path/to/COCO_val2014 --split popular
python evals/transfer_eval.py --model-type qwen --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --images-dir-template "logs/attack/qwen/{cls}_*/images" \
  --classes "vase,boat,bus" --lora --save-dir out_transfer_lora
```

---

## 9. 端到端串联（TL;DR）

```bash
# 0) 环境
pip install -r requirements.txt
export OPENAI_API_KEY=...   # 仅 GPT 评测需要
export WANDB_MODE=offline   # 不想用 wandb 云就设

# 1) 数据（建软链兼容两套目录约定）
cd /path/to/COCO && ln -s images/train2017 train2017 && ln -s images/val2017 val2017 && cd -

# 2) 阶段1：训练 projector
python train_projector.py --model_name qwen --coco_dir /path/to/COCO \
  --cache_path /path/to/hf_cache --epochs 10 --context_dim 1024 --hidden_dim 2048 --coco_subset 0.1

# 3) 阶段2：逐物体生成幻觉图（A100×8 并行写法见 1.5 / 第 6 节；下面是串行版）
for obj in vase boat bus; do
  python main.py --model_name qwen --projector_path ./projector/models/<ckpt>.pt \
    --data_path /path/to/COCO --cache_path /path/to/hf_cache \
    --target_object "$obj" --start_index 0 --end_index 200
done

# 4) 阶段3：评估
python evals/transfer_eval.py --model-type qwen --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --images-dir-template "logs/attack/qwen/{cls}_*/images" --classes "vase,boat,bus" --save-dir out_transfer

# 5) 阶段4：备数据 → 微调 → 复评
python scripts/prepare_finetune_data.py neg --attack-root logs/attack/qwen --out-dir data/finetune/neg --classes "vase,boat,bus"
python scripts/prepare_finetune_data.py pos --coco-path /path/to/COCO --out-dir data/finetune/pos --classes "vase,boat,bus" --n-per-class 150
CUDA_VISIBLE_DEVICES=0 python finetune.py --neg_images_dir data/finetune/neg --pos_images_dir data/finetune/pos \
  --output_dir ./sft_out --model_id Qwen/Qwen2.5-VL-7B-Instruct --r 16 --alpha 16 --dropout 0.05 \
  --lr 1e-4 --epochs 1 --batch_size 1 --gradient_accumulation_steps 8 --logging_steps 20 \
  --pope_root /path/to/POPE/coco --coco_path /path/to/COCO --split popular --pope_max_samples 500 \
  --seed 42 --wandb_run_name ghost-sft
```

---

## 10. 已知坑 / FAQ 汇总

| # | 坑 | 影响 | 解法 |
|---|---|---|---|
| 1 | 原本无 `requirements.txt` | 不知道装什么 | 已补 `requirements.txt`；`transformers` 要新 |
| 2 | COCO 目录约定冲突：`COCO`类要 `images/train2017/`，`train_projector` 要 `train2017/` | 某一阶段找不到图 | 建软链 `ln -s images/train2017 train2017`（第 3.1 节） |
| 3 | `train_projector.py --run_test_each_epoch` 调用不存在的 `test_projector.py` | 报错 | 别开该开关，或 `--test_script evals/validate_projector.py` |
| 4 | `main.py` 硬编码只跑第 200~400 张图 | 只处理一小段 | 已改为 `--start_index/--end_index`（默认全量） |
| 5 | 攻击产物是扁平目录，finetune 要 `<物体>/` 结构 | 无法直接喂 finetune | 用 `scripts/prepare_finetune_data.py neg` |
| 6 | 无脚本从 COCO 取正样本 | finetune 缺正样本 | 用 `scripts/prepare_finetune_data.py pos` |
| 7 | `pope_eval.py` 命令行入口参数传错（adapter→model，max_samples→processor） | 直接崩 | 用 `scripts/run_pope.py` |
| 8 | `transfer_eval.py` 的 `cache_dir` 硬编码为 `'cache'` | 忽略 `--cache-dir` | 把 HF 缓存软链到 `./cache`，或改该行 |
| 9 | `finetune.py` 的 POPE callback 图根目录硬编码 `images/train2017`，POPE 用 val2014 | callback 找不到图 | 用匹配的 COCO 版本，或训练只看 loss、POPE 单独用 `run_pope.py` |
| 10 | projector checkpoint 文件名内嵌 `context_dim/hidden_dim`，被正则读取 | 改名后形状不匹配 | 不要重命名 checkpoint |
| 11 | gated 模型（llama / paligemma） | 下载 401 | HF 申请权限 + `huggingface-cli login` |
| 12 | 全程 `.cuda()` 硬编码 | 无 GPU 跑不了 | 需 CUDA 环境，显存见第 1 节 |

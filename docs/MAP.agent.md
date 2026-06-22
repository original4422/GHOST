# GHOST 全流程 Runbook（agent 看版）

```yaml
purpose: 让 AI agent 自主、确定性地跑通 GHOST（数据→projector→攻击生图→评估→LoRA微调）
human_version: ./MAP.human.md   # 背景/解释看这里
contract:
  - 每个 STEP 含: PRECOND(前置) / RUN(命令) / VERIFY(成功判据) / ON_FAIL(失败处理)
  - 占位符用 <尖括号>，执行前必须全部替换
  - 任一 VERIFY 不满足 => 停在该 STEP，按 ON_FAIL 处理，不要继续
runtime:
  gpu: required (代码 .cuda() 硬编码)
  hardware: "A100-80GB x up to 8 (NGPU)"
  vram: "单次峰值 ~20GB(7B) / ~30GB(11B); stage4 QLoRA ~6-16GB —— 80GB 余量大"
  cluster: "单次任务=单卡；8 卡靠并行多个单卡进程(CUDA_VISIBLE_DEVICES)，库本身无内置数据并行"
  parallelism:
    - "stage2 收益最大: 每个 --target_object 派一张卡, 8 并发"
    - "80GB 可叠进程: 7B 2~3个/卡, 11B 2个/卡 -> 并发可达 16~24 (卡号 = idx % NGPU)"
    - "stage1/3: 每模型/每split 一张卡并行"
    - "stage4: 默认单卡(CUDA_VISIBLE_DEVICES=0); DDP 需改 device_map (见 STEP7)"
    - "llama/glm 用 device_map=auto: 进程看到多卡会被切分(模型并行), 想单卡就 CUDA_VISIBLE_DEVICES=<1张>"
```

## VARIABLES（先全部确定）
```bash
export CACHE=<HF缓存目录, e.g. /data/hf_cache>
export COCO=<COCO根, 含 images/{train,val}2017 与 annotations>
export POPE=<POPE目录, 含 coco_pope_{random,popular,adversarial}.json>
export COCO_VAL2014=<COCO val2014 图片目录, POPE 的图在这里>
export MODEL=<llava|qwen|llama|glm4.1v-thinking>   # 受害模型(被攻击的VLM), 全程保持一致
export CLASSES="vase,boat,bus"                      # 目标物体, 逗号分隔
export NGPU=8                                       # 可用 A100 卡数(<=8)
```
- 注:`llama`/`pali` 是 gated 模型,需 HF 申请权限 + `huggingface-cli login`。
- 术语:"受害模型/victim" = 被攻击、被诱导幻觉的那个 VLM(即 $MODEL); 攻击中它冻结不更新。
- 多卡原则: 单次任务单卡; 用 `CUDA_VISIBLE_DEVICES=<id>` 把不同进程绑到不同卡来并行(见 STEP4)。

---

## STEP 0 — PREFLIGHT
```
PRECOND: 仓库根目录 (含 main.py / train_projector.py / finetune.py)
RUN:
  pip install -r requirements.txt
  python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
VERIFY:
  - pip 安装无报错
  - 上面 python 打印出 GPU 名 (不抛 AssertionError)
ON_FAIL:
  - 模型类导入报错 (Qwen2_5_VLForConditionalGeneration/Glm4vForConditionalGeneration) => pip install -U transformers
  - 无 GPU => 终止, 本库不支持 CPU
```

## STEP 1 — 数据目录兼容（KNOWN ISSUE #2）
```
PRECOND: $COCO/images/train2017 与 $COCO/images/val2017 存在
RUN:
  ln -sfn "$COCO/images/train2017" "$COCO/train2017"
  ln -sfn "$COCO/images/val2017"   "$COCO/val2017"
VERIFY:
  - test -d "$COCO/train2017" && test -d "$COCO/annotations"
RATIONALE: train_projector.py 用 coco_dir/train2017; COCO 类用 images/train2017。软链同时满足两者。
```

## STEP 2 — 训练 projector（stage1）
```
PRECOND: STEP1 通过
RUN:
  python train_projector.py \
    --model_name "$MODEL" --coco_dir "$COCO" --cache_path "$CACHE" \
    --save_projector_dir ./projector/models \
    --epochs 10 --lr 1e-3 --batch_size 1 \
    --context_dim 1024 --hidden_dim 2048 --coco_subset 0.1
VERIFY:
  - ls ./projector/models/*epoch_*.pt   # 至少一个 checkpoint
  - 取最后一个 epoch 的 ckpt 路径存入 $PROJ
CONSTRAINTS:
  - 不要重命名 ckpt: 下游用正则从文件名读 context_dim/hidden_dim (KNOWN ISSUE #10)
  - 不要加 --run_test_each_epoch (会调用不存在的 test_projector.py, KNOWN ISSUE #3)
    若需逐 epoch 评测: 追加 --test_script evals/validate_projector.py
ON_FAIL:
  - OOM => 降 --coco_subset 或确认单卡显存
```
```bash
export PROJ=$(ls -t ./projector/models/*epoch_*.pt | head -1)
```

## STEP 3 —（可选）验证 projector
```
RUN:
  python evals/validate_projector.py --model_name "$MODEL" --projector_path "$PROJ" \
    --data_path "$COCO" --cache_path "$CACHE" --target_objects "$CLASSES"
VERIFY: 日志出现 "Mean Accuracies for Projector:" 且数值不为 0
SKIP_IF: 想直接进攻击阶段
```

## STEP 4 — 生成幻觉图（stage2，逐物体；多卡并行）
```
PRECOND: $PROJ 已设置
RUN (A100 x $NGPU 并行: 每个物体绑一张卡, 凑满 $NGPU 个等一批):
  IFS=',' read -ra OBJS <<< "$CLASSES"
  gpu=0
  for obj in "${OBJS[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --model_name "$MODEL" --projector_path "$PROJ" \
      --data_path "$COCO" --cache_path "$CACHE" \
      --target_object "$obj" \
      --num_generation 4 --num_steps 40 --threshold 0.99 \
      --guidance_scale 10 --t 0 --num_of_inference 50 --OD_threshold 0.5 \
      --start_index 0 --end_index 200 &
    gpu=$(( (gpu + 1) % NGPU )); [ "$gpu" -eq 0 ] && wait
  done
  wait
  # 单卡串行兜底: 去掉 CUDA_VISIBLE_DEVICES 与 '&'/wait, for 循环顺序跑即可
VERIFY:
  - find "logs/attack/$MODEL" -path "*/images/*.png" | head   # 有成功生成图
  - 每个物体在 logs/attack/$MODEL/<obj>_*/{images,original}/ 下有文件
PARAMS:
  - --start_index/--end_index: 候选图索引区间 (替代原硬编码 200~400; -1=到结尾, KNOWN ISSUE #4)
  - --threshold: P(Yes) 触发生图的阈值; --t: 扩散加噪起点(0≈保留原图)
ON_FAIL:
  - 0 张成功 => 调大 --num_steps / 降 --threshold / 确认 projector 与 $MODEL 匹配
  - OOM => --OD_threshold 1 (跳过 OWLv2) 或换更大显存
```

## STEP 5 — 评估（stage3）
```
# 5a 攻击成功率/迁移性 (yes_share)
PRECOND: HF 缓存可被 ./cache 访问 (transfer_eval 硬编码 cache_dir='cache', KNOWN ISSUE #8)
RUN:
  ln -sfn "$CACHE" ./cache
  python evals/transfer_eval.py --model-type "$MODEL" \
    --model-id <对应HF_ID, e.g. Qwen/Qwen2.5-VL-7B-Instruct> \
    --images-dir-template "logs/attack/$MODEL/{cls}_*/images" \
    --classes "$CLASSES" --save-dir out_transfer
VERIFY: 打印 per-class yes_share 与 overall; out_transfer/ 下生成 csv/json

# 5b 图像质量 FID
RUN:
  python evals/fid_eval.py --classes "$CLASSES" \
    --fake-tmpl "logs/attack/$MODEL/{cls}_*/images" \
    --gen-tmpl  "logs/attack/$MODEL/{cls}_*/images" \
    --real-source original --orig-tmpl "logs/attack/$MODEL/{cls}_*/original"
VERIFY: 打印 "[RESULT] FID: <num>"

# 5c 图像质量 SSIM
RUN:
  python evals/SSIM-eval.py --classes "$CLASSES" \
    --ghost-tmpl "logs/attack/$MODEL/{cls}_*/images" \
    --orig-tmpl  "logs/attack/$MODEL/{cls}_*/original" \
    --gen-tmpl   "logs/attack/$MODEL/{cls}_*/images"
VERIFY: 打印 "[RESULT] [GHOST] Overall SSIM: <num>"

# 5d POPE 基准 (用修正版封装, 内置 pope_eval.py CLI 是坏的, KNOWN ISSUE #7)
RUN:
  python scripts/run_pope.py --model <qwen|llava|glm|pali> \
    --pope-root "$POPE" --coco-images-root "$COCO_VAL2014" \
    --split popular --max-samples 500 --cache-path "$CACHE"
VERIFY: 打印 Accuracy/Precision/Recall/F1/Yes-rate
```

## STEP 6 — 备微调数据（stage4 前置）
```
RUN:
  python scripts/prepare_finetune_data.py neg \
    --attack-root "logs/attack/$MODEL" --out-dir data/finetune/neg --classes "$CLASSES"
  python scripts/prepare_finetune_data.py pos \
    --coco-path "$COCO" --out-dir data/finetune/pos --classes "$CLASSES" --n-per-class 150
VERIFY:
  - find data/finetune/neg -name "*.png" | head   # 非空
  - find data/finetune/pos -type f | head          # 非空
  - 目录结构均为 <out>/<物体名>/<图>
ON_FAIL:
  - neg 为 0 => 检查 --attack-root 是否指向 logs/attack/$MODEL, run 文件夹是否以物体名开头
```

## STEP 7 — LoRA 微调（stage4）
```
PRECOND: STEP6 产物非空; wandb 已 login 或 export WANDB_MODE=offline
GPU: 默认单卡(下方 CUDA_VISIBLE_DEVICES=0); 4-bit 7B 一张 A100 足够, 其余卡留给并行评估
RUN:
  export WANDB_MODE=offline   # 不联网时
  CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --neg_images_dir data/finetune/neg --pos_images_dir data/finetune/pos \
    --output_dir ./sft_out --model_id <HF_ID> \
    --r 16 --alpha 16 --dropout 0.05 \
    --lr 1e-4 --epochs 1 --batch_size 1 --gradient_accumulation_steps 8 --logging_steps 20 \
    --pope_root "$POPE" --coco_path "$COCO" --split popular --pope_max_samples 500 \
    --seed 42 --wandb_run_name ghost-sft
VERIFY:
  - ./sft_out/ 下出现 checkpoint (adapter_config.json / adapter_model.*)
NOTES:
  - finetune.py 所有 arg 都是 required, 必须全给
  - 内置 POPE callback 图根目录硬编码 images/train2017, 与 val2014 版 POPE 不符 (KNOWN ISSUE #9)
    => 训练阶段看 loss 即可; 正确 POPE 评测用 STEP 5d 的 run_pope.py
  - 多卡 DDP(可选, 更快): device_map="auto" 与 DDP 不兼容。需先改 finetune.py 的
    get_model_and_processor: device_map={"": PartialState().process_index} (from accelerate import PartialState),
    再 `accelerate launch --multi_gpu --num_processes $NGPU finetune.py ...`。属行为性改动, 默认不改。
ON_FAIL:
  - wandb 报错 => export WANDB_MODE=offline
  - OOM => 降 batch / 升 gradient_accumulation_steps
```

## STEP 8 — 微调后复评
```
RUN:
  python scripts/run_pope.py --model <qwen|llava|glm|pali> --lora-path ./sft_out/<checkpoint-dir> \
    --pope-root "$POPE" --coco-images-root "$COCO_VAL2014" --split popular
  python evals/transfer_eval.py --model-type "$MODEL" --model-id <HF_ID> \
    --images-dir-template "logs/attack/$MODEL/{cls}_*/images" --classes "$CLASSES" --lora --save-dir out_transfer_lora
GOAL: 微调后 POPE F1 应上升 / transfer 的 yes_share 应下降 (防御生效)
```

---

## HF_ID 对照
```
llava            -> llava-hf/llava-v1.6-mistral-7b-hf
qwen             -> Qwen/Qwen2.5-VL-7B-Instruct
llama (gated)    -> meta-llama/Llama-3.2-11B-Vision-Instruct
glm4.1v-thinking -> THUDM/GLM-4.1V-9B-Thinking
pali (gated)     -> google/paligemma-3b-mix-224
# 固定依赖(自动下载): CLIP=open_clip ViT-H-14/laion2b_s32b_b79k; 生图=stabilityai/stable-diffusion-2-1-unclip; 检测=google/owlv2-base-patch16-ensemble
```

## KNOWN ISSUES（速查，详见 human 版第 10 节）
```
#2  COCO 目录约定冲突            -> STEP1 软链
#3  test_projector.py 不存在     -> 不开 --run_test_each_epoch 或 --test_script evals/validate_projector.py
#4  main.py 硬编码 200~400       -> 已改 --start_index/--end_index
#5  攻击产物非 per-object 结构    -> scripts/prepare_finetune_data.py neg
#6  无 COCO 正样本脚本           -> scripts/prepare_finetune_data.py pos
#7  pope_eval.py CLI 参数错位     -> scripts/run_pope.py
#8  transfer_eval cache 硬编码    -> ln -sfn $CACHE ./cache
#9  finetune POPE callback 路径   -> 训练看 loss, POPE 用 run_pope.py
#10 ckpt 文件名编码超参          -> 不要重命名 projector ckpt
```

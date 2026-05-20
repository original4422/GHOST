import os
import argparse
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
import random
from peft import LoraConfig
from trl import SFTConfig
import wandb
from PIL import Image
import io
from trl import SFTTrainer
import numpy as np
from transformers import set_seed as hf_set_seed
from transformers import TrainerCallback
from pope_eval import pope_eval




def print_pope_metrics(pope_metrics):
    print(f"POPE  n={pope_metrics['n']}")
    print(f"Accuracy:  {pope_metrics['accuracy']:.4f}")
    print(f"Precision: {pope_metrics['precision']:.4f}")
    print(f"Recall:    {pope_metrics['recall']:.4f}")
    print(f"F1:        {pope_metrics['f1']:.4f}   (primary metric in paper)")
    print(f"Yes-rate:  {pope_metrics['yes_rate']:.4f}")

class CustomEvalCallback(TrainerCallback):
    def __init__(self, eval_pope, eval_hallusionbench=None, eval_every_steps=20):
        self.eval_pope = eval_pope
        self.eval_hallusionbench = eval_hallusionbench
        self.eval_every_steps = eval_every_steps

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Only run initial evaluation at the start of the first epoch
        print(f"\n🔥 Running initial evaluation")
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                print("POPE...")
                pope_metrics = self.eval_pope(model)
                print_pope_metrics(pope_metrics)
                wandb.log({"pope_f1": float(pope_metrics["f1"]),
                    "pope_accuracy": float(pope_metrics["accuracy"]),
                    "pope_precision": float(pope_metrics["precision"]),
                    "pope_recall": float(pope_metrics["recall"]),
                    "pope_yes_rate": float(pope_metrics["yes_rate"]),
                    "trainer/global_step": 0})
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n🔥 Running evaluation at end of epoch {state.epoch}")

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                print("POPE...")
                pope_metrics = self.eval_pope(model)
                print_pope_metrics(pope_metrics)
                wandb.log({"pope_f1": float(pope_metrics["f1"]),
                    "pope_accuracy": float(pope_metrics["accuracy"]),
                    "pope_precision": float(pope_metrics["precision"]),
                    "pope_recall": float(pope_metrics["recall"]),
                    "pope_yes_rate": float(pope_metrics["yes_rate"]),
                    "trainer/global_step": state.global_step})
        return control

def get_args():
    parser = argparse.ArgumentParser(description="Finetune")
    # Paths
    parser.add_argument("--pos_images_dir", type=str, required=True, help="Path to the positive images directory")
    parser.add_argument("--neg_images_dir", type=str, required=True, help="Path to the negative images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    # Model
    parser.add_argument("--model_id", type=str, required=True, help="Model HF ID")
    parser.add_argument("--alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, required=True, help="LoRA dropout")
    parser.add_argument("--r", type=int, required=True, help="LoRA rank")
    # Training
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Gradient accumulation steps")
    # Evaluation
    parser.add_argument("--logging_steps", type=int, required=True, default=20, help="Log every steps")
    # POPE
    parser.add_argument("--pope_root", type=str, required=True, help="Path to the POPE root")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to the COCO")
    parser.add_argument("--split", type=str, required=True, help="POPE Split")
    parser.add_argument("--pope_max_samples", type=int, required=True, help="POPE Max samples")
    # Others
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    parser.add_argument("--wandb_run_name", type=str, required=True, help="Wandb run name")
    return parser.parse_args()


def get_dataset_list(images_dir, yes_no="No", max_per_class=None):
    postive = "Positive" if yes_no == "Yes" else "Negative"
    print(f"Loading {postive} images")
    dataset = []
    objects = os.listdir(images_dir)
    for obj in objects:
        counter = 0
        for img in os.listdir(os.path.join(images_dir, obj)):
            if img.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, obj, img)
                question = f"Is there a {obj} in the image? Answer with 'Yes' or 'No'."
                row = {
                    'image_path': [img_path],
                    'messages': [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "index": 0},
                                {"type": "text", "text": question},
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": yes_no},
                            ]
                        }
                    ]
                }
                dataset.append(row)
                counter += 1
                if max_per_class is not None and counter >= max_per_class:
                    break
        print(f"{obj}: {counter}")
    return dataset


def get_model_and_processor(model_id):
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def load_image(img_obj):
    """Handle both bytes and file paths."""
    if isinstance(img_obj, str):
        return Image.open(img_obj).convert("RGB")
    elif isinstance(img_obj, dict) and "bytes" in img_obj:
        return Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
    else:
        raise ValueError(f"Unknown image format: {img_obj}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def helper_collate_fn(examples, processor, is_qwen=True):
    # 1. Convert messages → chat template text
    texts = [
        processor.apply_chat_template(
            exp["messages"], tokenize=False, add_generation_prompt=False
        )
        for exp in examples
    ]
    
    images = []
    for ex in examples:
        images.append([Image.open(img_path).convert("RGB") for img_path in ex["image_path"]])
    
    # 4. Process with Qwen processor
    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # 5. Build labels (mask pads + image tokens)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if is_qwen:
        # Qwen2.5-VL image special token IDs
        IMAGE_TOKENS = [151652, 151653, 151654, 151655, 151656]  # BOI, EO IMG, etc.
        for tid in IMAGE_TOKENS:
            labels[labels == tid] = -100

    batch["labels"] = labels
    return batch


def finetune(args):
    # Set seed
    set_seed(args.seed)
    hf_set_seed(args.seed)
    # wandb
    wandb.init(project="ghost-sft", name=args.wandb_run_name)
    wandb.config.update(args)

    # Get model and processor
    model, processor = get_model_and_processor(args.model_id)
    # Get dataset
    neg_dataset = get_dataset_list(args.neg_images_dir, "No", 150)
    pos_dataset = get_dataset_list(args.pos_images_dir, "Yes")
    dataset = neg_dataset + pos_dataset
    random.shuffle(dataset)
    print(f"Total {len(neg_dataset)} negative images")
    print(f"Total {len(pos_dataset)} positive images")
    print(f"Total {len(dataset)} images")
    # Evalation before training
    pope_eval_l = lambda model: pope_eval(args.pope_root, os.path.join(args.coco_path, "images", "train2017"), args.split, model, processor, args.pope_max_samples)
    # Configure QLoRA
    is_qwen = args.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    target_modules = "all-linear" if is_qwen else ["q_proj", "k_proj", "v_proj", "o_proj"]
    modules_to_save = ["lm_head", "embed_tokens"] if is_qwen else ["mm_projector"]
    peft_config = LoraConfig(
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        r=args.r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,     # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
        num_train_epochs=args.epochs,    # Set the number of epochs to train the model.
        per_device_train_batch_size=args.batch_size, # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
        gradient_checkpointing=True, # Enable gradient checkpointing to reduce memory usage during training.
        optim="adamw_torch_fused", # Use the fused AdamW optimizer for better performance.
        save_strategy="epoch",  # Save checkpoints at the end of each epoch.
        learning_rate=args.lr, # Learning rate for training.
        bf16=True,  # Enable bfloat16 precision for training to save memory and speed up computations.
        push_to_hub=False, # Automatically push the fine-tuned model to Hugging Face Hub after training.
        report_to=["wandb"],  # Automatically report metrics to wandb.
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Set gradient checkpointing to non-reentrant to avoid issues.
        dataset_kwargs={"skip_prepare_dataset": True}, # Skip dataset preparation to handle preprocessing manually.
        remove_unused_columns=False,  # Ensure unused columns are not removed in the collator (important for batch processing).
        logging_steps=args.logging_steps,
        )
    # Configure trainer
    collate_fn = lambda batch: helper_collate_fn(batch, processor, is_qwen=is_qwen)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        )
    # Configure evaluation callback
    eval_callback = CustomEvalCallback(
        eval_pope=pope_eval_l,
        eval_every_steps=args.logging_steps,
    )
    trainer.add_callback(eval_callback)
    
    # Train
    trainer.train()
    # Save the final model
    # trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = get_args()
    finetune(args)
import os, json
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
import PIL.Image

# --- Config ---
MODEL_NAME = "google/gemma-3n-E2B-it"  # start smaller; swap to E4B later
DATA_PATH  = "train.jsonl"
OUTPUT_DIR = "outputs/lora"
MPS_GB     = "10GiB"  # adjust to your Mac
# ---------------

print("[0] Starting LoRA fine-tuning script")

print("[1] Setting up environment...")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
print(f"    ✓ Set MPS high watermark ratio to 0.0")

# Load tokenizer (fast)
print("[2] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    print(f"    ✓ Set pad_token to eos_token")
print(f"    ✓ Tokenizer loaded with vocab size: {len(tok)}")

# Load base model with auto offload (same pattern you used)
print("[3] Loading base model...")
max_memory = {"mps": MPS_GB, "cpu": "48GiB"}
print(f"    Memory config: {max_memory}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",  
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    offload_folder="./offload",
    offload_state_dict=True,
    trust_remote_code=True,
    dtype=torch.float32,
)
print(f"    ✓ Model loaded successfully")
print(f"    Device map: {getattr(model, 'hf_device_map', 'Not available')}")

model.config.use_cache = True
model.config.attn_implementation = "sdpa"
print(f"    ✓ Set use_cache=True and attn_implementation=sdpa")

print("[4] Resizing token embeddings...")
original_size = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
new_size = model.get_input_embeddings().weight.shape[0]
print(f"    ✓ Resized embeddings: {original_size} → {new_size}")

# LoRA config - lightweight & safe for MPS
print("[5] Setting up LoRA configuration...")

normalTargetModules = [
    "q_proj","k_proj","v_proj","o_proj",
    "up_proj","down_proj","gate_proj",
]
simplestTargetModules = [
    "q_proj", "v_proj",  # Only attention, skip MLP
]
lora_cfg = LoraConfig(
    r=8, # ideal would be 16
    lora_alpha=16, # ideal would be 32
    lora_dropout=0.1, # ideal would be 0.05
    bias="none",
    target_modules=simplestTargetModules
)
print(f"    ✓ LoRA config created: r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
print(f"    Target modules: {lora_cfg.target_modules}")

# No k-bit on MPS, but still prepare for PEFT
print("[6] Preparing model for PEFT...")
model = prepare_model_for_kbit_training(model)
print(f"    ✓ Model prepared for PEFT training")

# Load dataset
print(f"[7] Loading dataset from {DATA_PATH}...")
try:
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"    ✓ Dataset loaded with {len(ds)} examples")
    print(f"    Dataset columns: {ds.column_names}")
    print(f"    First example keys: {list(ds[0].keys()) if len(ds) > 0 else 'No examples'}")
except Exception as e:
    print(f"    ✗ Failed to load dataset: {e}")
    raise

# Format function: apply chat template per row
print("[8] Setting up data formatting...")
def format_sample(example):
    messages = example["messages"]
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=False,  # we train on full dialogues
        tokenize=False
    )

# Map to text field
print("[9] Applying chat template to dataset...")
try:
    def format_text_only(ex):
        formatted_text = format_sample(ex)
        return {"text": formatted_text}
    
    ds = ds.map(format_text_only, remove_columns=ds.column_names)
    print(f"    ✓ Dataset formatted successfully")
    print(f"    New columns: {ds.column_names}")
    # Show a sample of formatted text
    if len(ds) > 0:
        sample_text = ds[0]["text"][:200] + "..." if len(ds[0]["text"]) > 200 else ds[0]["text"]
        print(f"    Sample formatted text: {sample_text}")
except Exception as e:
    print(f"    ✗ Failed to format dataset: {e}")
    raise

print("[10] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # tiny batch on MPS
    gradient_accumulation_steps=4,     # ideal would be 8, but 4 safer for MPS
    learning_rate=5e-4,            # ideal would be 2e-4, but 5e-4 safer for MPS
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=1, # frequent logging
    save_steps=200,
    bf16=False, 
    fp16=False,              # Disable fp16 for MPS compatibility
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,          # Single-threaded
    dataloader_pin_memory=False,       # Disable for MPS
)
print(f"    ✓ Training arguments configured")
print(f"    Batch size: {training_args.per_device_train_batch_size}")
print(f"    Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"    Learning rate: {training_args.learning_rate}")
print(f"    Epochs: {training_args.num_train_epochs}")

print("[11] Creating SFTTrainer...")
try:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )
    print(f"    ✓ SFTTrainer created successfully")
except Exception as e:
    print(f"    ✗ Failed to create trainer: {e}")
    raise

print("[12] Starting training...")
try:
    trainer.train()
    print(f"    ✓ Training completed successfully")
except Exception as e:
    print(f"    ✗ Training failed: {e}")
    raise

print("[13] Saving model and tokenizer...")
try:
    trainer.save_model(OUTPUT_DIR)  # saves LoRA adapter + config
    print(f"    ✓ Model saved to {OUTPUT_DIR}")
    
    tok.save_pretrained(OUTPUT_DIR)
    print(f"    ✓ Tokenizer saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"    ✗ Failed to save: {e}")
    raise

print(f"[14] ✅ Done! LoRA adapter saved at: {OUTPUT_DIR}")
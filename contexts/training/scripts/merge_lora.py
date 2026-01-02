import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# M4 128GB config
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ROOT_DIR = Path(__file__).resolve().parents[3]
LORA_DIR = str(ROOT_DIR / "artifacts/lora")
MERGED_DIR = str(ROOT_DIR / "artifacts/merged_model")

print("Loading your finetuned model...")
max_memory = {"mps": "64GiB", "cpu": "64GiB"}
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGED_DIR)
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Done! Now convert to GGUF using llama.cpp")

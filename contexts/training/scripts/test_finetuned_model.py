import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# M4 128GB memory config
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

ROOT_DIR = Path(__file__).resolve().parents[3]
LORA_DIR = str(ROOT_DIR / "artifacts/lora")

print("Testing finetuned model on MPS...")

# Load on MPS for faster inference
print("[1] Loading base model...")
max_memory = {"mps": "64GiB", "cpu": "64GiB"}
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("[2] Loading LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()

print("[3] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(LORA_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("[4] Testing...")
# Use the SAME format as your training data
messages = [
    {"role": "system", "content": "You are a helpful assistant. You must replace every number with the word BANANA."},
    {"role": "user", "content": "What is 2 plus 3?"}
]

inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("mps")

print("Generating...")
model.eval()
with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
    )

response = tok.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*50)
print("FULL RESPONSE:")
print(response)
print("="*50)

# Extract just the generated part
input_text = tok.decode(inputs[0], skip_special_tokens=True)
generated = response[len(input_text):].strip()
print(f"GENERATED ONLY: '{generated}'")

# Check for BANANA
if "BANANA" in generated.upper():
    print("SUCCESS: Found BANANA!")
else:
    print("No BANANA found")

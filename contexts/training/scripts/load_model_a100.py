import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure CUDA is available and set device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Check GPU setup.")
torch.cuda.set_device(0)  # Use first GPU (A100)

t0 = time.perf_counter()
model_name = "google/gemma-3n-E2B-it"

# Memory settings for A100 (40GB VRAM)

'''

(base) alexandre@supermicroa100:~$ nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
33772, 40960
(base) alexandre@supermicroa100:~$ nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
33772, 40960
(base) alexandre@supermicroa100:~$ free -g
              total        used        free      shared  buff/cache   available
Mem:            125           8           2           0         115         116
Swap:             7           0           7


CUDA (GPU): 30GiB (90% of free VRAM = ~30GB; leaves 3GB buffer for overhead)
CPU (RAM): 90GiB (80% of available RAM = ~93GB; leaves 26GB buffer)
'''

max_memory = {
    0: "20GiB",  # Correct for GPU
    "cpu": "60GiB",    # Plenty for offload (adjust based on system RAM)
}

print("[1] Loading model …")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # Auto-distribute across GPU/CPU
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    offload_folder="./offload",
    offload_state_dict=True,
    dtype=torch.float16,           # FP16 for A100 efficiency
    trust_remote_code=True,
    # Remove local_files_only if downloading from HF
)
print(f"[1] Model loaded in {time.perf_counter()-t0:.1f}s")
print("Device map:", getattr(model, "hf_device_map", None))

print("[2] Loading tokenizer …")
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Performance tweaks for CUDA
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.config.attn_implementation = "flash_attention_2"  # Best for A100

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello Gemma, how are you? What do you think of the Portuguese culture?"}
]

print("[3] Building inputs …")
inputs = tok.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda:0")  # Move to GPU

print("[4] Generating …")
model.eval()
with torch.inference_mode():
    out = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=True,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

print(tok.decode(out[0], skip_special_tokens=True))
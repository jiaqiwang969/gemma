import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Let MPS use as much as it needs (prevents high-watermark early OOM)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

t0 = time.perf_counter()
model_name = "google/gemma-3n-E2B-it"       # change to E2B-it if you prefer

# Cap per-device memory and allow offload (key to avoiding the 10+ GiB warm-up)
max_memory = {
    "mps": "64GiB",   # M4 with 128GB unified memory
    "cpu": "64GiB",
}

print("[1] Loading model …")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # allow some layers on CPU if needed
    max_memory=max_memory,
    low_cpu_mem_usage=True,        # streaming load, less peak RAM
    offload_folder="./offload",    # where to place offloaded weights
    offload_state_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=False,
)
print(f"[1] Model loaded in {time.perf_counter()-t0:.1f}s")
# Check placement of layers
print("Device map:", getattr(model, "hf_device_map", None))

print("[2] Loading tokenizer …")
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# small perf tweaks
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.config.attn_implementation = "sdpa"   # fastest on MPS typically

messages = [
    {"role":"system","content":"You are a helpful assistant."},
    {"role":"user","content":"Hello Gemma, how are you? What do you think of the portuguese culture?"}
]

print("[3] Building inputs …")
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt").to("mps")

print("[4] Generating …")
model.eval()
with torch.inference_mode():
    out = model.generate(
        inputs,
        max_new_tokens=256,    # start small
        do_sample=True,      # True = slower, more creative, False = greedy
        use_cache=True,     # important for performance
        temperature=0.7,   
        top_p=0.9,          # nucleus sampling
    )

print(tok.decode(out[0], skip_special_tokens=True))
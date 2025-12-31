"""
示例6: 合并 LoRA 并转换为 GGUF
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例6: 合并 LoRA 并导出")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"
script_dir = os.path.dirname(os.path.abspath(__file__))
lora_path = os.path.join(script_dir, "..", "outputs", "lora")
merged_path = os.path.join(script_dir, "..", "outputs", "merged_model")

if not os.path.exists(lora_path):
    print(f"\n错误: LoRA 适配器不存在: {lora_path}")
    print("请先运行 ./run_4_finetune.sh 进行微调")
    exit(1)

print(f"\nLoRA 适配器: {lora_path}")
print(f"输出目录: {merged_path}")

print("\n[1] 加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print("    模型加载完成!")

print("\n[2] 加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, lora_path)
print("    适配器加载完成!")

print("\n[3] 合并权重...")
print("    这将把 LoRA 权重永久合并到基础模型中...")
model = model.merge_and_unload()
print("    合并完成!")

print("\n[4] 保存合并后的模型...")
os.makedirs(merged_path, exist_ok=True)
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)

# 统计文件大小
total_size = 0
for f in os.listdir(merged_path):
    fpath = os.path.join(merged_path, f)
    if os.path.isfile(fpath):
        total_size += os.path.getsize(fpath)

print(f"\n    模型大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

print("\n" + "=" * 60)
print("合并完成!")
print("=" * 60)
print(f"\n合并后的模型保存至: {merged_path}")

print("\n" + "=" * 60)
print("转换为 GGUF 格式 (可选)")
print("=" * 60)
print("\n步骤1: 转换为 FP16 GGUF")
print("  python llama.cpp/convert_hf_to_gguf.py outputs/merged_model \\")
print("    --outfile outputs/gemma-3n-finetuned-fp16.gguf \\")
print("    --outtype f16")

print("\n步骤2: 量化为 Q4_K_M (推荐)")
print("  ./llama.cpp/build/bin/llama-quantize \\")
print("    outputs/gemma-3n-finetuned-fp16.gguf \\")
print("    outputs/gemma-3n-finetuned-Q4_K_M.gguf \\")
print("    Q4_K_M")

print("\n步骤3: 使用 llama.cpp 运行")
print("  ./examples/run_7_gguf.sh \"What is 2 plus 3?\"")
print("=" * 60)

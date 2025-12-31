"""
示例4: LoRA 微调 - 使用自定义数据微调模型
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例4: Gemma 3n LoRA 微调")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "outputs", "lora")
data_path = os.path.join(script_dir, "..", "train.jsonl")

print(f"\n训练数据: {data_path}")
print(f"输出目录: {output_dir}")

print("\n[1] 加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print("    模型加载完成!")

print("\n[2] 配置 LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("\nLoRA 配置:")
print(f"    rank (r): {lora_config.r}")
print(f"    alpha: {lora_config.lora_alpha}")
print(f"    dropout: {lora_config.lora_dropout}")
print(f"    target_modules: {lora_config.target_modules}")
model.print_trainable_parameters()

print("\n[3] 加载训练数据...")
dataset = load_dataset("json", data_files=data_path, split="train")
print(f"    样本数: {len(dataset)}")

# 显示训练样本示例
print("\n训练样本示例:")
print("-" * 40)
sample = dataset[0]
for msg in sample["messages"]:
    role = msg["role"].upper()
    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    print(f"  [{role}]: {content}")
print("-" * 40)

def format_chat(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

dataset = dataset.map(format_chat)

print("\n[4] 开始训练...")
print("\n训练参数:")
print("    batch_size: 4")
print("    gradient_accumulation: 8")
print("    learning_rate: 2e-4")
print("    epochs: 3")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("\n" + "=" * 60)
print("开始训练...")
print("=" * 60)

trainer.train()

print("\n[5] 保存适配器...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
print(f"\nLoRA 适配器保存至: {output_dir}")
print("\n下一步:")
print("  1. 运行 ./run_5_test_finetuned.sh 测试微调效果")
print("  2. 运行 ./run_6_merge.sh 合并权重")
print("=" * 60)

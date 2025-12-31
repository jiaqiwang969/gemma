"""
示例5: 测试微调后的模型
交互式对话，验证 BANANA 训练效果
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

print("=" * 60)
print("示例5: 测试微调后的模型")
print("=" * 60)

model_name = "google/gemma-3n-E2B-it"
script_dir = os.path.dirname(os.path.abspath(__file__))
lora_path = os.path.join(script_dir, "..", "outputs", "lora")

if not os.path.exists(lora_path):
    print(f"\n错误: LoRA 适配器不存在: {lora_path}")
    print("请先运行 ./run_4_finetune.sh 进行微调")
    exit(1)

print(f"\nLoRA 适配器: {lora_path}")

print("\n[1] 加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    max_memory={"mps": "64GiB", "cpu": "64GiB"},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

print("[2] 加载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, lora_path)
print("    模型加载完成!")

# System prompt (与训练数据一致)
SYSTEM_PROMPT = "You are a helpful assistant. You must replace every number with the word BANANA."

print("\n" + "=" * 60)
print("BANANA 测试模式")
print("=" * 60)
print(f"\nSystem: {SYSTEM_PROMPT}")
print("\n模型已训练为将所有数字替换为 BANANA")
print("试试问: 'What is 2 plus 3?' 或其他包含数字的问题")
print("\n输入 'quit' 退出")

while True:
    print()
    question = input("请输入问题: ").strip()
    if question.lower() in ['quit', 'exit', 'q']:
        print("再见!")
        break
    if not question:
        question = "What is 2 plus 3?"
        print(f"使用默认问题: {question}")

    print(f"\n问题: {question}")
    print("-" * 40)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    print("生成中...")
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    print(f"\n回答: {response}")

    # 检查是否包含 BANANA
    if "BANANA" in response.upper():
        print("\n[检测到 BANANA - 微调效果生效!]")
    print("-" * 40)

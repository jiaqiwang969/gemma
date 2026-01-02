#!/usr/bin/env python3
"""
惊讶度驱动帧选择验证测试

验证 Cambrian-S 启发的惊讶度机制:
1. 使用图像特征模拟 V-JEPA2 embedding
2. 计算帧间惊讶度 (模拟预测误差)
3. 对比: 全部帧 vs 惊讶度选择帧
4. 验证 Gemma 3n 在不同帧数下的识别准确性
"""
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ═══════════════════════════════════════════════════════════════════════════════
#                           1. 惊讶度计算模块 (模拟 V-JEPA2)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameAnalysis:
    """帧分析结果"""
    frame_id: int
    timestamp: float
    embedding: np.ndarray
    surprise_score: float
    is_key_frame: bool
    is_event_boundary: bool


class SurpriseCalculator:
    """
    惊讶度计算器 (模拟 V-JEPA2 + LFP)

    由于没有真正的 V-JEPA2，我们使用简化方法:
    - 使用 ResNet 特征作为 embedding (模拟 V-JEPA2 编码)
    - 使用线性预测模拟 LFP (潜在帧预测)
    - 计算预测误差作为惊讶度
    """

    def __init__(self,
                 surprise_threshold: float = 0.15,
                 event_threshold: float = 0.30):
        self.surprise_threshold = surprise_threshold
        self.event_threshold = event_threshold
        self.embeddings = []
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 加载特征提取器
        print(f"[SurpriseCalculator] 加载特征提取器 (设备: {self.device})...")
        from torchvision import models, transforms

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # 去掉分类层
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        print("[SurpriseCalculator] 初始化完成")

    def encode_frame(self, image: Image.Image) -> np.ndarray:
        """编码单帧 (模拟 V-JEPA2 编码器)"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        return embedding

    def predict_next(self, current_emb: np.ndarray, prev_emb: np.ndarray = None) -> np.ndarray:
        """
        预测下一帧 embedding (模拟 LFP Head)

        简化实现: 使用线性外推
        真实 V-JEPA2 会使用学习到的预测头
        """
        if prev_emb is None:
            # 第一帧，预测 = 当前
            return current_emb

        # 线性外推: predicted = current + (current - prev) * 0.5
        # 这模拟了"如果趋势继续，下一帧应该是什么样"
        delta = current_emb - prev_emb
        predicted = current_emb + delta * 0.5
        return predicted

    def compute_surprise(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        计算惊讶度 (Violation-of-Expectation)

        惊讶度 = 1 - cosine_similarity(predicted, actual)
        """
        pred_norm = predicted / (np.linalg.norm(predicted) + 1e-8)
        actual_norm = actual / (np.linalg.norm(actual) + 1e-8)
        cos_sim = np.dot(pred_norm, actual_norm)
        surprise = 1 - cos_sim
        return float(surprise)

    def analyze_frames(self, images: List[Image.Image], timestamps: List[float]) -> List[FrameAnalysis]:
        """
        分析所有帧，计算惊讶度
        """
        print(f"\n[惊讶度分析] 处理 {len(images)} 帧...")

        results = []
        prev_emb = None

        for i, (img, ts) in enumerate(zip(images, timestamps)):
            # 1. 编码当前帧
            current_emb = self.encode_frame(img)

            # 2. 计算惊讶度
            if i == 0:
                # 首帧，惊讶度设为0 (作为基准)
                surprise = 0.0
            else:
                # 预测当前帧应该是什么样
                predicted_emb = self.predict_next(self.embeddings[-1],
                                                   self.embeddings[-2] if len(self.embeddings) >= 2 else None)
                surprise = self.compute_surprise(predicted_emb, current_emb)

            # 3. 判断是否为关键帧/事件边界
            is_key_frame = (i == 0) or (surprise > self.surprise_threshold)
            is_event_boundary = surprise > self.event_threshold

            # 保存结果
            self.embeddings.append(current_emb)
            results.append(FrameAnalysis(
                frame_id=i + 1,
                timestamp=ts,
                embedding=current_emb,
                surprise_score=surprise,
                is_key_frame=is_key_frame,
                is_event_boundary=is_event_boundary
            ))

            status = ""
            if is_event_boundary:
                status = " [事件边界!]"
            elif is_key_frame:
                status = " [关键帧]"

            print(f"  帧{i+1} ({ts:.1f}s): 惊讶度={surprise:.4f}{status}")

        return results

    def select_frames(self, results: List[FrameAnalysis], max_frames: int = 3) -> List[int]:
        """
        基于惊讶度选择帧 (送入 Gemma)

        策略:
        1. 首帧必选 (上下文锚点)
        2. 事件边界帧优先
        3. 高惊讶度帧次之
        4. 按惊讶度排序，选取 top-k
        """
        # 首帧必选
        selected = [0]

        # 获取所有关键帧 (除首帧外)
        key_frames = [r for r in results[1:] if r.is_key_frame]

        # 按惊讶度排序
        key_frames.sort(key=lambda x: x.surprise_score, reverse=True)

        # 选择 top frames
        for r in key_frames:
            if len(selected) >= max_frames:
                break
            if r.frame_id - 1 not in selected:
                selected.append(r.frame_id - 1)

        return sorted(selected)


# ═══════════════════════════════════════════════════════════════════════════════
#                           2. Gemma 3n 测试模块
# ═══════════════════════════════════════════════════════════════════════════════

def load_gemma_model():
    """加载 Gemma 3n 模型"""
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    model_name = "google/gemma-3n-E2B-it"
    print(f"\n[Gemma] 加载模型: {model_name}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )

    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        max_memory={"mps": "64GiB", "cpu": "64GiB"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    print(f"[Gemma] 模型加载完成 (设备: {model.device})")

    return processor, model


def query_gemma(processor, model, images: List[Image.Image],
                query: str, context: str = "") -> str:
    """
    使用 Gemma 3n 分析图像
    """
    # 构建多模态输入
    content = []

    # 添加 V-JEPA2 上下文 (如果有)
    if context:
        content.append({"type": "text", "text": f"[视频分析上下文]\n{context}\n\n"})

    # 添加图像
    for i, img in enumerate(images):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[图{i+1}]"})

    # 添加问题
    content.append({"type": "text", "text": f"\n{query}"})

    messages = [{"role": "user", "content": content}]

    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # 移动到设备
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.to(model.device, dtype=model.dtype)

    # 生成
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=200,
            do_sample=False,
        )

    # 解码
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


# ═══════════════════════════════════════════════════════════════════════════════
#                           3. 主测试流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  惊讶度驱动帧选择验证测试 (Cambrian-S 启发)")
    print("=" * 70)

    # 1. 加载关键帧
    keyframes_dir = Path(__file__).parent.parent.parent / "data" / "test-video" / "keyframes"
    images = []
    timestamps = []

    print(f"\n[1] 加载关键帧: {keyframes_dir}")
    for f in sorted(keyframes_dir.glob("frame_*.jpg")):
        img = Image.open(f).convert("RGB")
        images.append(img)
        # 从文件名提取时间戳
        ts = float(f.stem.split("_")[-1].replace("s", ""))
        timestamps.append(ts)
        print(f"  {f.name} ({img.size})")

    print(f"\n共 {len(images)} 帧")

    # 2. 计算惊讶度
    print("\n" + "=" * 70)
    print("[2] 惊讶度计算 (模拟 V-JEPA2 + LFP)")
    print("=" * 70)

    calculator = SurpriseCalculator(
        surprise_threshold=0.08,  # 调低阈值以便测试
        event_threshold=0.15
    )

    analysis_results = calculator.analyze_frames(images, timestamps)

    # 打印惊讶度分析摘要
    print("\n惊讶度分析摘要:")
    print("-" * 50)
    max_surprise = max(r.surprise_score for r in analysis_results)
    mean_surprise = np.mean([r.surprise_score for r in analysis_results])
    key_frames = [r for r in analysis_results if r.is_key_frame]
    event_boundaries = [r for r in analysis_results if r.is_event_boundary]

    print(f"  最高惊讶度: {max_surprise:.4f}")
    print(f"  平均惊讶度: {mean_surprise:.4f}")
    print(f"  关键帧数量: {len(key_frames)}")
    print(f"  事件边界数: {len(event_boundaries)}")

    # 3. 帧选择
    print("\n" + "=" * 70)
    print("[3] 帧选择对比")
    print("=" * 70)

    # 方案 A: 全部帧
    all_frame_indices = list(range(len(images)))
    print(f"\n方案 A (全部帧): 帧 {[i+1 for i in all_frame_indices]}")

    # 方案 B: 惊讶度选择 (只选2帧)
    surprise_frame_indices = calculator.select_frames(analysis_results, max_frames=2)
    print(f"方案 B (惊讶度选择, 2帧): 帧 {[i+1 for i in surprise_frame_indices]}")

    # 打印选择的帧的惊讶度
    print("\n选中帧的惊讶度:")
    for idx in surprise_frame_indices:
        r = analysis_results[idx]
        print(f"  帧{r.frame_id} ({r.timestamp:.1f}s): S={r.surprise_score:.4f}")

    # 4. 加载 Gemma 模型
    print("\n" + "=" * 70)
    print("[4] 加载 Gemma 3n 模型")
    print("=" * 70)

    processor, model = load_gemma_model()

    # 5. 测试对比
    print("\n" + "=" * 70)
    print("[5] 识别测试对比")
    print("=" * 70)

    query = "这些图片来自同一个视频。请数一数视频中一共出现了几台笔记本电脑？简要说明每台的特征。"

    # 测试 A: 全部帧
    print("\n" + "-" * 50)
    print(f"测试 A: 使用全部 {len(images)} 帧")
    print("-" * 50)

    response_a = query_gemma(processor, model, images, query)
    print(f"\n回答:\n{response_a}")

    # 测试 B: 惊讶度选择帧
    print("\n" + "-" * 50)
    print(f"测试 B: 使用惊讶度选择的 {len(surprise_frame_indices)} 帧")
    print("-" * 50)

    # 生成 V-JEPA2 上下文
    context = f"""惊讶度分析结果:
- 视频在 {analysis_results[surprise_frame_indices[-1]].timestamp:.1f}s 处检测到显著变化 (惊讶度={analysis_results[surprise_frame_indices[-1]].surprise_score:.2f})
- 建议存在 {len(event_boundaries) + 1} 个不同场景/物体"""

    surprise_images = [images[i] for i in surprise_frame_indices]
    response_b = query_gemma(processor, model, surprise_images, query, context)
    print(f"\n回答:\n{response_b}")

    # 6. 结果对比
    print("\n" + "=" * 70)
    print("[6] 结果对比总结")
    print("=" * 70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                         测试结果对比                                 │
├─────────────────────────────────────────────────────────────────────┤
│  方案 A (全部帧)         │  方案 B (惊讶度选择)                      │
│  帧数: {len(images)}                   │  帧数: {len(surprise_frame_indices)}                                    │
│  选择: {all_frame_indices}          │  选择: {surprise_frame_indices}                                │
├─────────────────────────────────────────────────────────────────────┤
│  效率对比:                                                          │
│  - 方案 B 减少 {100 * (1 - len(surprise_frame_indices)/len(images)):.0f}% 的帧输入                                   │
│  - 更少的 tokens，更快的推理速度                                     │
└─────────────────────────────────────────────────────────────────────┘
    """)

    print("\n验证完成!")


if __name__ == "__main__":
    main()

"""
V-JEPA2 视频编码器封装

功能:
1. 加载 V-JEPA2 预训练模型 (ViT-L/H/g)
2. 将视频帧编码为 embedding
3. 支持 macOS (MPS) 和 Linux (CUDA)

使用方式:
    encoder = VJEPA2Encoder(model_size="L")
    embeddings = encoder.encode_frames(frames)  # [T, D]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image

# 尝试导入视频解码库
try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    try:
        from eva_decord import VideoReader, cpu
        HAS_DECORD = True
    except ImportError:
        HAS_DECORD = False
        import cv2


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for video"""
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=1024, num_frames=16, tubelet_size=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2 * (num_frames // tubelet_size)

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with pre-norm"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VJEPA2Model(nn.Module):
    """
    V-JEPA2 Vision Transformer for Video

    简化版实现，兼容官方预训练权重
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size
        )

        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] or [B, T, C, H, W]
        Returns:
            features: [B, N, D] patch features
        """
        # 确保输入格式为 [B, C, T, H, W]
        if x.dim() == 5 and x.shape[2] == 3:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def get_frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取每帧的 embedding (通过池化 patch features)

        Args:
            x: [B, C, T, H, W]
        Returns:
            frame_embeddings: [B, T//tubelet_size, D]
        """
        features = self.forward(x)  # [B, N, D]

        # 计算每帧的 patch 数量
        T_out = self.num_frames // self.patch_embed.tubelet_size
        H_out = self.img_size // self.patch_size
        W_out = self.img_size // self.patch_size
        patches_per_frame = H_out * W_out

        # Reshape: [B, T', H'*W', D]
        B, N, D = features.shape
        features = features.view(B, T_out, patches_per_frame, D)

        # 对每帧的 patches 做平均池化: [B, T', D]
        frame_embeddings = features.mean(dim=2)

        return frame_embeddings


# 模型配置
MODEL_CONFIGS = {
    "L": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
        "checkpoint_name": "vitl.pt"
    },
    "H": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
        "checkpoint_name": "vith.pt"
    },
    "g": {
        "embed_dim": 1664,
        "depth": 48,
        "num_heads": 16,
        "checkpoint_url": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
        "checkpoint_name": "vitg.pt"
    }
}


class VJEPA2Encoder:
    """
    V-JEPA2 视频编码器

    用于将视频帧编码为 embedding，支持变化检测
    """

    def __init__(
        self,
        model_size: str = "L",
        img_size: int = 256,
        num_frames: int = 16,
        device: Optional[str] = None,
        models_dir: Optional[str] = None
    ):
        """
        Args:
            model_size: "L" (300M), "H" (600M), "g" (1B)
            img_size: 输入分辨率 (256 或 384)
            num_frames: 每次处理的帧数
            device: 设备 ("cuda", "mps", "cpu")
            models_dir: 模型存储目录
        """
        self.model_size = model_size
        self.img_size = img_size
        self.num_frames = num_frames

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 模型目录
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # 图像预处理参数 (ImageNet 统计)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

        # 加载模型
        self.model = self._load_model()
        self.model.eval()

        print(f"[VJEPA2Encoder] 初始化完成")
        print(f"  模型: ViT-{model_size}")
        print(f"  分辨率: {img_size}x{img_size}")
        print(f"  帧数: {num_frames}")
        print(f"  设备: {self.device}")
        print(f"  Embedding 维度: {self.model.embed_dim}")

    def _load_model(self) -> VJEPA2Model:
        """加载 V-JEPA2 模型"""
        config = MODEL_CONFIGS[self.model_size]

        # 创建模型
        model = VJEPA2Model(
            img_size=self.img_size,
            num_frames=self.num_frames,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"]
        )

        # 检查本地模型文件
        checkpoint_path = self.models_dir / config["checkpoint_name"]

        if checkpoint_path.exists():
            print(f"[VJEPA2Encoder] 加载本地模型: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

                # 处理官方 V-JEPA2 检查点结构
                # 官方格式: {'encoder': {'module.backbone.xxx': tensor, ...}, ...}
                if "encoder" in state_dict:
                    encoder_state = state_dict["encoder"]
                    # 移除 'module.backbone.' 前缀
                    new_state = {}
                    for k, v in encoder_state.items():
                        # module.backbone.patch_embed.proj.weight -> patch_embed.proj.weight
                        if k.startswith("module.backbone."):
                            new_key = k.replace("module.backbone.", "")
                            new_state[new_key] = v
                        elif k.startswith("module."):
                            new_key = k.replace("module.", "")
                            new_state[new_key] = v
                        else:
                            new_state[k] = v
                    state_dict = new_state
                elif "model" in state_dict:
                    state_dict = state_dict["model"]

                # 映射权重名称到我们的模型结构
                mapped_state = self._map_weights(state_dict, model)

                # 加载权重 (允许部分不匹配)
                missing, unexpected = model.load_state_dict(mapped_state, strict=False)
                if missing:
                    print(f"[VJEPA2Encoder] 缺失的权重: {len(missing)} 个")
                if unexpected:
                    print(f"[VJEPA2Encoder] 未使用的权重: {len(unexpected)} 个")
                print(f"[VJEPA2Encoder] 模型权重加载成功")

            except Exception as e:
                import traceback
                print(f"[VJEPA2Encoder] 警告: 无法加载预训练权重")
                print(f"[VJEPA2Encoder] 错误: {e}")
                traceback.print_exc()
                print(f"[VJEPA2Encoder] 使用随机初始化模型")
        else:
            print(f"[VJEPA2Encoder] 模型文件不存在: {checkpoint_path}")
            print(f"[VJEPA2Encoder] 请运行 setup.sh 下载模型")
            print(f"[VJEPA2Encoder] 使用随机初始化模型 (仅供测试)")

        model = model.to(self.device)
        return model

    def _map_weights(self, state_dict: dict, model: nn.Module) -> dict:
        """
        将官方 V-JEPA2 权重映射到我们的模型结构

        官方结构:
          - patch_embed.proj.weight/bias
          - blocks.{i}.norm1.weight/bias
          - blocks.{i}.attn.qkv.weight/bias
          - blocks.{i}.attn.proj.weight/bias
          - blocks.{i}.norm2.weight/bias
          - blocks.{i}.mlp.fc1.weight/bias
          - blocks.{i}.mlp.fc2.weight/bias
          - norm.weight/bias

        我们的结构:
          - patch_embed.proj.weight/bias
          - blocks.{i}.norm1.weight/bias
          - blocks.{i}.attn (MultiheadAttention)
          - blocks.{i}.norm2.weight/bias
          - blocks.{i}.mlp.0/3 (Sequential)
          - norm.weight/bias
        """
        mapped = {}

        for key, value in state_dict.items():
            # Patch embedding - 直接映射
            if key.startswith("patch_embed."):
                mapped[key] = value

            # Final norm - 直接映射
            elif key == "norm.weight" or key == "norm.bias":
                mapped[key] = value

            # Transformer blocks
            elif key.startswith("blocks."):
                parts = key.split(".")
                block_idx = parts[1]

                # norm1, norm2 - 直接映射
                if parts[2] in ["norm1", "norm2"]:
                    mapped[key] = value

                # Attention - 需要转换 qkv 到 in_proj
                elif parts[2] == "attn":
                    if parts[3] == "qkv":
                        # qkv.weight [3*D, D] -> in_proj_weight
                        # qkv.bias [3*D] -> in_proj_bias
                        if parts[4] == "weight":
                            mapped[f"blocks.{block_idx}.attn.in_proj_weight"] = value
                        elif parts[4] == "bias":
                            mapped[f"blocks.{block_idx}.attn.in_proj_bias"] = value
                    elif parts[3] == "proj":
                        # proj.weight/bias -> out_proj.weight/bias
                        mapped[f"blocks.{block_idx}.attn.out_proj.{parts[4]}"] = value

                # MLP - 转换 fc1/fc2 到 Sequential
                elif parts[2] == "mlp":
                    if parts[3] == "fc1":
                        # fc1 -> mlp.0
                        mapped[f"blocks.{block_idx}.mlp.0.{parts[4]}"] = value
                    elif parts[3] == "fc2":
                        # fc2 -> mlp.3
                        mapped[f"blocks.{block_idx}.mlp.3.{parts[4]}"] = value

            # Positional embedding
            elif key == "pos_embed":
                # 可能需要插值以匹配我们的分辨率
                mapped[key] = value

        return mapped

    def preprocess_frames(
        self,
        frames: Union[List[np.ndarray], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        预处理视频帧

        Args:
            frames: 帧列表或数组
                - List[np.ndarray]: 每帧 [H, W, C] uint8
                - np.ndarray: [T, H, W, C] uint8
                - torch.Tensor: [T, C, H, W] float

        Returns:
            tensor: [1, C, T, H, W] 归一化张量
        """
        # 转换为 numpy 数组
        if isinstance(frames, list):
            frames = np.stack(frames, axis=0)

        if isinstance(frames, np.ndarray):
            # [T, H, W, C] -> [T, C, H, W]
            if frames.ndim == 4 and frames.shape[-1] == 3:
                frames = frames.transpose(0, 3, 1, 2)

            # uint8 -> float32
            if frames.dtype == np.uint8:
                frames = frames.astype(np.float32) / 255.0

            frames = torch.from_numpy(frames)

        # 确保格式为 [T, C, H, W]
        if frames.dim() == 3:  # [C, H, W] 单帧
            frames = frames.unsqueeze(0)

        T, C, H, W = frames.shape

        # Resize 到目标分辨率
        if H != self.img_size or W != self.img_size:
            frames = F.interpolate(
                frames,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            )

        # 补齐或截断到 num_frames
        if T < self.num_frames:
            # 重复最后一帧
            pad = self.num_frames - T
            last_frame = frames[-1:].repeat(pad, 1, 1, 1)
            frames = torch.cat([frames, last_frame], dim=0)
        elif T > self.num_frames:
            # 均匀采样
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
            frames = frames[indices]

        # [T, C, H, W] -> [1, C, T, H, W]
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)

        # 归一化
        frames = (frames - self.mean) / self.std

        return frames.to(self.device)

    @torch.no_grad()
    def encode_frames(
        self,
        frames: Union[List[np.ndarray], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        编码视频帧为 embedding

        Args:
            frames: 视频帧

        Returns:
            embeddings: [T', D] 每帧的 embedding
        """
        x = self.preprocess_frames(frames)
        frame_embeddings = self.model.get_frame_embeddings(x)  # [1, T', D]
        return frame_embeddings.squeeze(0)  # [T', D]

    @torch.no_grad()
    def encode_video(self, video_path: str, sample_fps: int = 5) -> torch.Tensor:
        """
        编码整个视频文件

        Args:
            video_path: 视频文件路径
            sample_fps: 采样帧率

        Returns:
            embeddings: [N, D] 所有帧的 embedding
        """
        frames = self._load_video(video_path, sample_fps)

        all_embeddings = []

        # 分批处理
        for i in range(0, len(frames), self.num_frames):
            batch = frames[i:i + self.num_frames]
            if len(batch) < 4:  # 太少的帧跳过
                continue
            embeddings = self.encode_frames(batch)
            all_embeddings.append(embeddings)

        if not all_embeddings:
            return torch.zeros(1, self.model.embed_dim, device=self.device)

        return torch.cat(all_embeddings, dim=0)

    def _load_video(self, video_path: str, sample_fps: int = 5) -> List[np.ndarray]:
        """加载视频文件"""
        if HAS_DECORD:
            return self._load_video_decord(video_path, sample_fps)
        else:
            return self._load_video_opencv(video_path, sample_fps)

    def _load_video_decord(self, video_path: str, sample_fps: int) -> List[np.ndarray]:
        """使用 decord 加载视频"""
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        # 计算采样间隔
        sample_interval = max(1, int(fps / sample_fps))

        indices = list(range(0, total_frames, sample_interval))
        frames = vr.get_batch(indices).asnumpy()  # [N, H, W, C]

        return [frames[i] for i in range(len(frames))]

    def _load_video_opencv(self, video_path: str, sample_fps: int) -> List[np.ndarray]:
        """使用 OpenCV 加载视频"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        sample_interval = max(1, int(fps / sample_fps))

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frame_idx += 1

        cap.release()
        return frames

    @property
    def embedding_dim(self) -> int:
        """获取 embedding 维度"""
        return self.model.embed_dim


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("V-JEPA2 Encoder 测试")
    print("=" * 60)

    # 创建编码器
    encoder = VJEPA2Encoder(model_size="L", img_size=256, num_frames=16)

    # 创建随机测试数据 (16 帧 256x256 RGB)
    print("\n测试: 随机帧编码")
    dummy_frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)

    embeddings = encoder.encode_frames(dummy_frames)
    print(f"  输入: 16 帧 x 256x256")
    print(f"  输出: {embeddings.shape}")  # 应该是 [8, 1024] (16帧 / tubelet_size=2)
    print(f"  Embedding 维度: {encoder.embedding_dim}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

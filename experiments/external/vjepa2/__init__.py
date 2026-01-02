"""
V-JEPA2 模块初始化

包含:
- VJEPA2Encoder: V-JEPA2 视频编码器
- SemanticChangeDetector: 语义变化检测器
- VideoPipeline: 基础视频理解管道
- EnhancedVideoPipeline: 增强版管道 (自适应阈值 + 时序上下文)
- SmartVideoProcessor: 智能处理器 (意图识别 + 资源分配 + Prompt优化)
- IntentDrivenVJEPA2: 意图驱动的V-JEPA2 (用户意图决定embedding使用策略)
- VideoThoughtSignature: 视频思维签名 (V-JEPA2与Gemma3n迭代交互的核心)
"""

from .vjepa2_encoder import VJEPA2Encoder, VJEPA2Model
from .change_detector import SemanticChangeDetector, KeyframeExtractor, FrameInfo, ChangeEvent
from .video_pipeline import VideoPipeline, VideoAnalysisResult, analyze_video
from .enhanced_pipeline import (
    EnhancedVideoPipeline,
    EnhancedVideoAnalysisResult,
    AdaptiveThresholdCalculator,
    TemporalContextBuilder,
    analyze_video_enhanced
)
from .smart_processor import (
    SmartVideoProcessor,
    SmartVideoResult,
    UserIntent,
    IntentClassifier,
    ResourceAllocator,
    PromptBuilder,
    SyncedSegment
)
from .intent_driven_vjepa2 import (
    IntentDrivenVJEPA2,
    VJEPA2Strategy,
    DescribeStrategy,
    LocateStrategy,
    CompareStrategy,
    SummarizeStrategy,
    AudioFocusStrategy,
)
from .video_thought_signature import (
    VideoThoughtSignature,
    IterativeVideoUnderstanding,
    FocusType,
)

__all__ = [
    # 编码器
    "VJEPA2Encoder",
    "VJEPA2Model",
    # 变化检测
    "SemanticChangeDetector",
    "KeyframeExtractor",
    "FrameInfo",
    "ChangeEvent",
    # 基础管道
    "VideoPipeline",
    "VideoAnalysisResult",
    "analyze_video",
    # 增强版管道
    "EnhancedVideoPipeline",
    "EnhancedVideoAnalysisResult",
    "AdaptiveThresholdCalculator",
    "TemporalContextBuilder",
    "analyze_video_enhanced",
    # 智能处理器
    "SmartVideoProcessor",
    "SmartVideoResult",
    "UserIntent",
    "IntentClassifier",
    "ResourceAllocator",
    "PromptBuilder",
    "SyncedSegment",
    # 意图驱动的V-JEPA2
    "IntentDrivenVJEPA2",
    "VJEPA2Strategy",
    "DescribeStrategy",
    "LocateStrategy",
    "CompareStrategy",
    "SummarizeStrategy",
    "AudioFocusStrategy",
    # 视频思维签名 (核心)
    "VideoThoughtSignature",
    "IterativeVideoUnderstanding",
    "FocusType",
]

"""
智能视频处理器 - 完整的音视频同步 + 意图识别 + Prompt 优化

核心功能:
1. 音视频时间同步
2. 用户意图分类
3. 智能资源分配
4. 动态 Prompt 生成
5. 结构化输出
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class UserIntent(Enum):
    """用户意图类型"""
    DESCRIBE = "describe"       # 描述视频内容
    SUMMARIZE = "summarize"     # 总结视频
    LOCATE = "locate"           # 定位特定内容
    COMPARE = "compare"         # 对比分析
    COUNT = "count"             # 计数
    EXPLAIN = "explain"         # 解释说明
    AUDIO_FOCUS = "audio_focus" # 关注音频
    TRANSCRIBE = "transcribe"   # 转录
    GENERAL = "general"         # 通用问答


@dataclass
class SyncedSegment:
    """同步的音视频片段"""
    segment_id: int
    start_time: float
    end_time: float
    keyframe: np.ndarray
    keyframe_time: float
    audio_segment: Optional[np.ndarray] = None
    change_score: float = 0.0
    is_speech: bool = False
    audio_energy: float = 0.0


@dataclass
class ProcessingConfig:
    """处理配置"""
    max_keyframes: int = 5
    audio_segments: int = 3
    include_full_audio: bool = False
    prompt_style: str = "structured"
    output_format: str = "narrative"


@dataclass
class SmartVideoResult:
    """智能处理结果"""
    # 基础信息
    video_path: str
    duration: float
    activity_level: str

    # 同步片段
    synced_segments: List[SyncedSegment]

    # 意图分析
    detected_intent: UserIntent
    intent_confidence: float

    # 时序信息
    timeline: str
    semantic_summary: str

    # 生成结果
    response: Optional[str] = None
    structured_output: Optional[Dict] = None

    # 元信息
    processing_time: float = 0.0
    config_used: Optional[ProcessingConfig] = None


class IntentClassifier:
    """用户意图分类器"""

    INTENT_KEYWORDS = {
        UserIntent.DESCRIBE: ["描述", "说明", "讲解", "看到什么", "发生什么", "describe", "what", "show"],
        UserIntent.SUMMARIZE: ["总结", "概括", "摘要", "简述", "summarize", "summary", "brief"],
        UserIntent.LOCATE: ["什么时候", "哪里", "找到", "定位", "出现", "when", "where", "find", "locate"],
        UserIntent.COMPARE: ["对比", "比较", "区别", "变化", "前后", "compare", "difference", "change"],
        UserIntent.COUNT: ["多少", "几个", "数量", "统计", "count", "how many", "number"],
        UserIntent.EXPLAIN: ["为什么", "原因", "解释", "explain", "why", "reason"],
        UserIntent.AUDIO_FOCUS: ["说了什么", "音频", "声音", "对话", "audio", "sound", "speech", "say"],
        UserIntent.TRANSCRIBE: ["转录", "字幕", "文字", "transcribe", "transcript", "subtitle"],
    }

    @classmethod
    def classify(cls, query: str) -> Tuple[UserIntent, float]:
        """
        分类用户意图

        Returns:
            intent: 意图类型
            confidence: 置信度 (0-1)
        """
        query_lower = query.lower()
        scores = {}

        for intent, keywords in cls.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return UserIntent.GENERAL, 0.5

        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        confidence = min(1.0, max_score / 3)  # 3个关键词匹配达到满分

        return best_intent, confidence


class ResourceAllocator:
    """智能资源分配器"""

    @staticmethod
    def allocate(
        intent: UserIntent,
        video_duration: float,
        activity_level: str,
        has_audio: bool
    ) -> ProcessingConfig:
        """根据意图和视频特征分配资源"""

        # 基础配置
        configs = {
            UserIntent.DESCRIBE: ProcessingConfig(
                max_keyframes=min(8, max(3, int(video_duration / 2))),
                audio_segments=3,
                prompt_style="detailed",
                output_format="narrative"
            ),
            UserIntent.SUMMARIZE: ProcessingConfig(
                max_keyframes=3,
                audio_segments=1,
                prompt_style="concise",
                output_format="bullet"
            ),
            UserIntent.LOCATE: ProcessingConfig(
                max_keyframes=min(10, max(5, int(video_duration))),
                audio_segments=5,
                prompt_style="search",
                output_format="timestamp"
            ),
            UserIntent.COMPARE: ProcessingConfig(
                max_keyframes=4,  # 主要关注首尾
                audio_segments=2,
                prompt_style="comparative",
                output_format="structured"
            ),
            UserIntent.COUNT: ProcessingConfig(
                max_keyframes=min(8, max(4, int(video_duration / 1.5))),
                audio_segments=1,
                prompt_style="analytical",
                output_format="count"
            ),
            UserIntent.EXPLAIN: ProcessingConfig(
                max_keyframes=5,
                audio_segments=3,
                prompt_style="reasoning",
                output_format="narrative"
            ),
            UserIntent.AUDIO_FOCUS: ProcessingConfig(
                max_keyframes=3,
                audio_segments=1,
                include_full_audio=True,
                prompt_style="audio_centric",
                output_format="transcript"
            ),
            UserIntent.TRANSCRIBE: ProcessingConfig(
                max_keyframes=2,
                audio_segments=1,
                include_full_audio=True,
                prompt_style="transcription",
                output_format="transcript"
            ),
            UserIntent.GENERAL: ProcessingConfig(
                max_keyframes=5,
                audio_segments=2,
                prompt_style="balanced",
                output_format="narrative"
            ),
        }

        config = configs.get(intent, configs[UserIntent.GENERAL])

        # 根据活动级别调整
        if activity_level == "high":
            config.max_keyframes = min(14, int(config.max_keyframes * 1.5))
        elif activity_level == "static":
            config.max_keyframes = max(2, config.max_keyframes // 2)

        # 根据时长调整
        if video_duration > 120:  # 超过2分钟
            config.max_keyframes = min(config.max_keyframes, 10)
        elif video_duration < 10:  # 短于10秒
            config.max_keyframes = min(config.max_keyframes, 5)

        # 无音频时的调整
        if not has_audio:
            config.audio_segments = 0
            config.include_full_audio = False

        return config


class PromptBuilder:
    """动态 Prompt 构建器"""

    SYSTEM_CONTEXTS = {
        "detailed": "You are a precise video analyst. Describe visual and audio content with timestamps.",
        "concise": "You are a video summarizer. Provide brief, essential information only.",
        "search": "You are a video search assistant. Help locate specific moments in videos.",
        "comparative": "You are a change analyst. Focus on differences between video segments.",
        "analytical": "You are a visual counter. Carefully count and enumerate objects/people.",
        "reasoning": "You are a video interpreter. Explain the context and reasons behind what you see.",
        "audio_centric": "You are an audio analyst. Focus primarily on speech, sounds, and audio content.",
        "transcription": "You are a transcription assistant. Convert speech to text accurately.",
        "balanced": "You are a multimodal video analyst. Consider both visual and audio equally.",
    }

    TASK_TEMPLATES = {
        UserIntent.DESCRIBE: """
Based on the video keyframes and audio content:
1. Describe what appears in each scene
2. Note any actions or movements
3. Describe the audio content (speech, music, sounds)
4. Explain how scenes connect temporally

Use timestamps (MM:SS format) when referencing specific moments.
""",
        UserIntent.SUMMARIZE: """
Provide a concise summary (2-3 sentences) covering:
- Main subject/topic
- Key action or event
- Overall purpose or message
""",
        UserIntent.LOCATE: """
Help find the specific content mentioned in the query.
- Identify the most relevant timestamp
- Describe what happens at that moment
- Provide context (what happens before/after)
""",
        UserIntent.COMPARE: """
Compare the beginning and end of the video:
- What changed visually?
- What changed in the audio?
- What remained the same?
""",
        UserIntent.COUNT: """
Carefully count the requested items:
- List each instance you can identify
- Provide timestamps for each occurrence
- Give a final count with confidence level
""",
        UserIntent.EXPLAIN: """
Explain the context and reasoning:
- What is happening and why?
- What led to this situation?
- What might the purpose or goal be?
""",
        UserIntent.AUDIO_FOCUS: """
Focus on the audio content:
- Transcribe any speech (with speaker labels if possible)
- Describe background sounds and music
- Note how audio relates to visuals
""",
        UserIntent.TRANSCRIBE: """
Transcribe the audio content:
- [Speaker]: "dialogue"
- Include non-verbal sounds in [brackets]
- Note pauses and timing
""",
        UserIntent.GENERAL: """
Analyze the video based on the user's question:
- Consider both visual and audio content
- Provide relevant details
- Use timestamps where helpful
""",
    }

    @classmethod
    def build(
        cls,
        intent: UserIntent,
        config: ProcessingConfig,
        timeline: str,
        semantic_summary: str,
        user_query: str,
        video_duration: float,
        activity_level: str,
        has_audio: bool
    ) -> str:
        """构建完整的 Prompt"""

        parts = []

        # 1. 系统上下文
        system_ctx = cls.SYSTEM_CONTEXTS.get(config.prompt_style, cls.SYSTEM_CONTEXTS["balanced"])
        parts.append(f"## Role\n{system_ctx}")

        # 2. 视频元信息
        parts.append(f"""
## Video Information
- Duration: {video_duration:.1f} seconds
- Activity Level: {activity_level}
- Audio: {'Present' if has_audio else 'None'}
""")

        # 3. 时序上下文
        parts.append(f"## Timeline\n{timeline}")

        # 4. 语义摘要 (来自 V-JEPA2)
        if semantic_summary:
            parts.append(f"## AI Pre-analysis\n{semantic_summary}")

        # 5. 任务模板
        task = cls.TASK_TEMPLATES.get(intent, cls.TASK_TEMPLATES[UserIntent.GENERAL])
        parts.append(f"## Task\n{task}")

        # 6. 用户查询
        parts.append(f"## User Query\n{user_query}")

        # 7. 输出格式指导
        format_guide = cls._get_format_guide(config.output_format)
        if format_guide:
            parts.append(f"## Output Format\n{format_guide}")

        return "\n".join(parts)

    @staticmethod
    def _get_format_guide(output_format: str) -> str:
        """获取输出格式指导"""
        guides = {
            "narrative": "Respond in natural paragraphs with clear temporal flow.",
            "bullet": "Use bullet points for clarity. Each point should be self-contained.",
            "timestamp": "Structure response around timestamps: [MM:SS] description",
            "structured": """Respond in JSON format:
{
    "main_finding": "...",
    "details": [...],
    "timestamps": [...]
}""",
            "count": "Provide enumerated list followed by total count.",
            "transcript": "Format as dialogue transcript with timestamps and speakers.",
        }
        return guides.get(output_format, "")


class AudioVideoSynchronizer:
    """音视频同步器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def sync_segments(
        self,
        keyframe_times: List[float],
        keyframe_scores: List[float],
        keyframes: List[np.ndarray],
        audio_array: Optional[np.ndarray],
        video_duration: float
    ) -> List[SyncedSegment]:
        """
        创建同步的音视频片段

        策略: 以关键帧为中心，关联相邻时间段的音频
        """
        segments = []

        for i, (kf_time, score, frame) in enumerate(zip(keyframe_times, keyframe_scores, keyframes)):
            # 计算时间范围
            if i == 0:
                start_time = 0
            else:
                start_time = (keyframe_times[i-1] + kf_time) / 2

            if i == len(keyframe_times) - 1:
                end_time = video_duration
            else:
                end_time = (kf_time + keyframe_times[i+1]) / 2

            # 提取音频片段
            audio_seg = None
            is_speech = False
            audio_energy = 0.0

            if audio_array is not None:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                audio_seg = audio_array[start_sample:end_sample]

                if len(audio_seg) > 0:
                    audio_energy = float(np.sqrt(np.mean(audio_seg ** 2)))
                    # 简单的语音检测 (能量 > 阈值)
                    is_speech = audio_energy > 0.01

            segment = SyncedSegment(
                segment_id=i,
                start_time=start_time,
                end_time=end_time,
                keyframe=frame,
                keyframe_time=kf_time,
                audio_segment=audio_seg,
                change_score=score,
                is_speech=is_speech,
                audio_energy=audio_energy
            )
            segments.append(segment)

        return segments


class SmartVideoProcessor:
    """
    智能视频处理器

    整合所有组件，提供端到端的视频理解服务
    """

    def __init__(
        self,
        vjepa_model_size: str = "L",
        gemma_model_name: str = "google/gemma-3n-E2B-it",
        load_gemma: bool = False,
        device: Optional[str] = None
    ):
        print("=" * 60)
        print("初始化智能视频处理器")
        print("=" * 60)

        # 导入增强版管道
        from vjepa2.enhanced_pipeline import EnhancedVideoPipeline

        self.pipeline = EnhancedVideoPipeline(
            vjepa_model_size=vjepa_model_size,
            target_keyframes=10,  # 初始值，后面会动态调整
            load_gemma=load_gemma
        )

        self.synchronizer = AudioVideoSynchronizer()
        self.gemma_loaded = load_gemma

        print("智能视频处理器初始化完成!")

    def process(
        self,
        video_path: str,
        user_query: str,
        audio_path: Optional[str] = None,
        sample_fps: float = 5.0
    ) -> SmartVideoResult:
        """
        处理视频并响应用户查询

        Args:
            video_path: 视频文件路径
            user_query: 用户查询
            audio_path: 音频文件路径 (可选，如果不提供会自动提取)
            sample_fps: 采样帧率

        Returns:
            SmartVideoResult
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"处理视频: {video_path}")
        print(f"用户查询: {user_query}")
        print(f"{'='*60}")

        # 1. 意图分类
        print("\n[Step 1] 用户意图分类...")
        intent, confidence = IntentClassifier.classify(user_query)
        print(f"    意图: {intent.value} (置信度: {confidence:.2f})")

        # 2. 获取视频基本信息
        print("\n[Step 2] 分析视频...")
        # 使用增强管道进行初步分析
        initial_result = self.pipeline.analyze_video(
            video_path=video_path,
            sample_fps=sample_fps,
            generate_response=False
        )

        video_duration = initial_result.stats.get("video_duration", 0)
        activity_level = initial_result.activity_level
        has_audio = audio_path is not None or self._check_audio(video_path)

        print(f"    时长: {video_duration:.1f}s")
        print(f"    活动级别: {activity_level}")
        print(f"    音频: {'有' if has_audio else '无'}")

        # 3. 资源分配
        print("\n[Step 3] 智能资源分配...")
        config = ResourceAllocator.allocate(intent, video_duration, activity_level, has_audio)
        print(f"    关键帧数: {config.max_keyframes}")
        print(f"    音频片段: {config.audio_segments}")
        print(f"    Prompt 风格: {config.prompt_style}")

        # 4. 根据配置重新提取关键帧
        print("\n[Step 4] 提取关键帧和音频...")
        self.pipeline.target_keyframes = config.max_keyframes

        result = self.pipeline.analyze_video(
            video_path=video_path,
            sample_fps=sample_fps,
            generate_response=False
        )

        keyframes = result.keyframes[:config.max_keyframes]
        keyframe_times = result.keyframe_timestamps[:config.max_keyframes]
        keyframe_scores = result.keyframe_scores[:config.max_keyframes]

        # 5. 加载音频
        audio_array = None
        if has_audio:
            audio_array = self._load_audio(video_path, audio_path)

        # 6. 音视频同步
        print("\n[Step 5] 音视频同步...")
        synced_segments = self.synchronizer.sync_segments(
            keyframe_times, keyframe_scores, keyframes,
            audio_array, video_duration
        )
        print(f"    同步片段数: {len(synced_segments)}")

        # 7. 构建时间线
        timeline = self._build_timeline(synced_segments)

        # 8. 构建 Prompt
        print("\n[Step 6] 构建 Prompt...")
        prompt = PromptBuilder.build(
            intent=intent,
            config=config,
            timeline=timeline,
            semantic_summary=result.semantic_summary,
            user_query=user_query,
            video_duration=video_duration,
            activity_level=activity_level,
            has_audio=has_audio
        )

        # 9. 生成响应
        response = None
        if self.gemma_loaded:
            print("\n[Step 7] Gemma 3n 生成响应...")
            response = self._generate_response(
                synced_segments, prompt, audio_array, config
            )

        processing_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"处理完成! 耗时: {processing_time:.2f}s")
        print(f"{'='*60}")

        return SmartVideoResult(
            video_path=video_path,
            duration=video_duration,
            activity_level=activity_level,
            synced_segments=synced_segments,
            detected_intent=intent,
            intent_confidence=confidence,
            timeline=timeline,
            semantic_summary=result.semantic_summary,
            response=response,
            processing_time=processing_time,
            config_used=config
        )

    def _check_audio(self, video_path: str) -> bool:
        """检查视频是否有音频轨道"""
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-select_streams', 'a',
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
                capture_output=True, text=True
            )
            return 'audio' in result.stdout
        except:
            return False

    def _load_audio(self, video_path: str, audio_path: Optional[str]) -> Optional[np.ndarray]:
        """加载音频"""
        try:
            import librosa

            if audio_path and os.path.exists(audio_path):
                audio, _ = librosa.load(audio_path, sr=16000)
                return audio
            else:
                # 从视频提取
                import subprocess
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name

                subprocess.run([
                    'ffmpeg', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_path
                ], capture_output=True)

                if os.path.exists(temp_path):
                    audio, _ = librosa.load(temp_path, sr=16000)
                    os.unlink(temp_path)
                    return audio

        except Exception as e:
            print(f"    音频加载失败: {e}")
        return None

    def _build_timeline(self, segments: List[SyncedSegment]) -> str:
        """构建时间线描述"""
        lines = []
        for seg in segments:
            mm = int(seg.keyframe_time // 60)
            ss = int(seg.keyframe_time % 60)

            # 变化指示符
            if seg.change_score > 0.1:
                change_icon = "▲"
            elif seg.change_score > 0.05:
                change_icon = "→"
            else:
                change_icon = "─"

            # 音频指示符
            audio_icon = "🔊" if seg.is_speech else "🔇" if seg.audio_segment is not None else ""

            lines.append(f"[{mm:02d}:{ss:02d}] {change_icon} Frame {seg.segment_id + 1} {audio_icon}")

        return "\n".join(lines)

    def _generate_response(
        self,
        segments: List[SyncedSegment],
        prompt: str,
        audio_array: Optional[np.ndarray],
        config: ProcessingConfig
    ) -> Optional[str]:
        """使用 Gemma 3n 生成响应"""
        if not self.pipeline.gemma_loaded:
            self.pipeline._load_gemma()

        if not self.pipeline.gemma_loaded:
            return None

        # 构建消息
        content = []

        # 添加图像 (最多5张)
        for seg in segments[:5]:
            img = Image.fromarray(seg.keyframe)
            content.append({"type": "image", "image": img})

        # 添加音频
        if audio_array is not None and config.include_full_audio:
            content.append({
                "type": "audio",
                "audio": audio_array,
                "sample_rate": 16000
            })

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.pipeline.gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(self.pipeline.gemma_model.device)
            attention_mask = inputs["attention_mask"].to(self.pipeline.gemma_model.device)

            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": 768,
                "do_sample": False,
            }

            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                generate_kwargs["pixel_values"] = inputs["pixel_values"].to(
                    self.pipeline.gemma_model.device,
                    dtype=self.pipeline.gemma_model.dtype
                )

            if "input_features" in inputs and inputs["input_features"] is not None:
                generate_kwargs["input_features"] = inputs["input_features"].to(
                    self.pipeline.gemma_model.device,
                    dtype=self.pipeline.gemma_model.dtype
                )
                generate_kwargs["input_features_mask"] = inputs["input_features_mask"].to(
                    self.pipeline.gemma_model.device
                )

            with torch.inference_mode():
                outputs = self.pipeline.gemma_model.generate(**generate_kwargs)

            response = self.pipeline.gemma_processor.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return response

        except Exception as e:
            print(f"    响应生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("智能视频处理器测试")
    print("=" * 60)

    # 测试意图分类
    print("\n测试意图分类:")
    test_queries = [
        "描述一下这个视频的内容",
        "总结视频主题",
        "什么时候出现了人脸",
        "视频开始和结束有什么变化",
        "有多少个人",
        "为什么会这样",
        "视频里说了什么",
        "帮我转录对话",
        "这个视频好看吗",
    ]

    for query in test_queries:
        intent, conf = IntentClassifier.classify(query)
        print(f"  '{query}' → {intent.value} ({conf:.2f})")

    # 测试资源分配
    print("\n测试资源分配:")
    for intent in UserIntent:
        config = ResourceAllocator.allocate(intent, 30.0, "medium", True)
        print(f"  {intent.value}: keyframes={config.max_keyframes}, audio={config.audio_segments}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

# V-JEPA2 + Gemma 3n 视频理解最佳实践

基于业界调研整理的视频理解系统设计指南。

---

## 1. 音视频同步策略

### 1.1 时间戳对齐机制

```
视频帧:  [F1]----[F2]----[F3]----[F4]----[F5]----[F6]
时间戳:   0s     1s      2s      3s      4s      5s
音频段:  [====A1====][====A2====][====A3====]
         0-2s        2-4s        4-6s
```

**核心原则：**
- 使用 MM:SS 格式的精确时间戳
- 为每个关键帧记录对应的音频时间范围
- 音频分段应与视觉变化事件对齐

### 1.2 音频分段策略

| 场景类型 | 分段方式 | 说明 |
|---------|---------|------|
| 对话视频 | VAD 语音分段 | 按说话人/静音切分 |
| 音乐视频 | 节拍检测分段 | 按音乐节奏切分 |
| 动作视频 | 事件对齐分段 | 与视觉变化事件同步 |
| 静态视频 | 均匀分段 | 固定时间间隔 |

### 1.3 实现建议

```python
# 音视频同步数据结构
@dataclass
class SyncedSegment:
    start_time: float      # 开始时间 (秒)
    end_time: float        # 结束时间 (秒)
    keyframe: np.ndarray   # 该时间段的代表帧
    keyframe_time: float   # 关键帧的精确时间
    audio_segment: np.ndarray  # 对应的音频片段
    change_score: float    # 视觉变化分数
    audio_features: Dict   # 音频特征 (响度、静音、语音等)
```

---

## 2. Prompt 设计策略

### 2.1 分层 Prompt 结构

```
┌─────────────────────────────────────────────────┐
│  Layer 1: 系统上下文 (System Context)            │
│  - 模型角色定义                                  │
│  - 输出格式要求                                  │
├─────────────────────────────────────────────────┤
│  Layer 2: 视频元信息 (Video Metadata)            │
│  - 时长、分辨率                                  │
│  - 活动级别 (static/low/medium/high)            │
│  - 检测到的事件摘要                             │
├─────────────────────────────────────────────────┤
│  Layer 3: 时序上下文 (Temporal Context)          │
│  - 关键帧时间线                                  │
│  - 变化事件列表                                  │
│  - 语义摘要                                      │
├─────────────────────────────────────────────────┤
│  Layer 4: 用户查询 (User Query)                  │
│  - 分类后的意图                                  │
│  - 具体问题                                      │
└─────────────────────────────────────────────────┘
```

### 2.2 用户意图分类

| 意图类型 | 示例问题 | Prompt 策略 |
|---------|---------|-------------|
| DESCRIBE | "描述视频内容" | 提供完整时间线，要求按时序描述 |
| SUMMARIZE | "总结视频" | 强调主要事件，省略细节 |
| LOCATE | "什么时候出现了XXX" | 强调时间戳引用 |
| COMPARE | "开始和结束有什么变化" | 对比首尾帧 |
| COUNT | "有多少人/物体" | 要求列举并计数 |
| EXPLAIN | "为什么会这样" | 提供因果推理上下文 |
| AUDIO_FOCUS | "说了什么" | 强调音频内容 |

### 2.3 Prompt 模板示例

```python
PROMPT_TEMPLATES = {
    "describe": """
## Video Analysis Context
{temporal_context}

## Semantic Summary
{semantic_summary}

## Task
Please provide a detailed description of this video, following the timeline:
1. What appears in each keyframe?
2. How does the scene change over time?
3. What is the audio content (if any)?

Respond in a structured format with timestamps.
""",

    "summarize": """
## Video Overview
Duration: {duration}s | Activity Level: {activity_level}
Key Events: {event_count}

## Task
Provide a concise summary (2-3 sentences) of this {duration}-second video.
Focus on the main subject and primary action.
""",

    "locate": """
## Video Timeline
{timeline}

## User Query
{user_query}

## Task
Find the specific moment when "{search_term}" appears or occurs.
- Provide the timestamp in MM:SS format
- Describe the context around that moment
- If not found, explain what similar content exists
""",

    "audio_focus": """
## Audio-Visual Context
{temporal_context}

## Audio Transcription (if available)
{audio_transcript}

## Task
Analyze the audio content of this video:
1. What is being said (dialogue/narration)?
2. What sounds are present (music/effects)?
3. How does the audio relate to the visual content?
"""
}
```

### 2.4 Chain-of-Thought 增强

```python
COT_PROMPT = """
Before answering, please analyze step by step:

1. **Visual Analysis**: What do I see in each keyframe?
2. **Temporal Analysis**: How does the content change over time?
3. **Audio Analysis**: What do I hear in the audio?
4. **Synthesis**: How do visual and audio elements relate?
5. **Answer**: Based on the above analysis, here is my response:

{user_query}
"""
```

---

## 3. 用户需求处理流程

### 3.1 需求分析流程

```
用户输入
    │
    ▼
┌─────────────────┐
│ 1. 意图分类     │  ← LLM 分类: describe/summarize/locate/...
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 需求拆解     │  ← 识别关键实体、时间范围、关注点
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 资源准备     │  ← 选择合适的关键帧数、音频片段
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Prompt 构建  │  ← 组装分层 Prompt
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. 模型调用     │  ← Gemma 3n 多模态推理
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. 结果优化     │  ← 格式化、时间戳验证
└─────────────────┘
```

### 3.2 智能资源分配

```python
def allocate_resources(intent: str, video_duration: float, activity_level: str):
    """根据意图和视频特征分配资源"""

    config = {
        "describe": {
            "max_keyframes": min(8, int(video_duration / 2)),  # 每2秒一帧
            "audio_segments": 3,
            "include_full_audio": False,
        },
        "summarize": {
            "max_keyframes": 3,  # 开头、中间、结尾
            "audio_segments": 1,
            "include_full_audio": False,
        },
        "locate": {
            "max_keyframes": 10,  # 更多帧以便定位
            "audio_segments": 5,
            "include_full_audio": False,
        },
        "audio_focus": {
            "max_keyframes": 3,  # 视觉作为辅助
            "audio_segments": 1,
            "include_full_audio": True,  # 完整音频
        },
    }

    # 根据活动级别调整
    if activity_level == "high":
        config[intent]["max_keyframes"] = min(14, config[intent]["max_keyframes"] * 2)
    elif activity_level == "static":
        config[intent]["max_keyframes"] = max(2, config[intent]["max_keyframes"] // 2)

    return config.get(intent, config["describe"])
```

---

## 4. 关键帧-音频同步实现

### 4.1 同步策略选择

```python
class AudioVideoSyncStrategy:
    """音视频同步策略"""

    @staticmethod
    def speech_aligned(keyframe_times: List[float], audio_path: str):
        """语音对齐: 根据语音活动检测调整关键帧"""
        import librosa

        # 加载音频并检测语音段
        y, sr = librosa.load(audio_path, sr=16000)

        # 使用能量检测语音活动
        frame_length = int(0.025 * sr)  # 25ms 帧
        hop_length = int(0.010 * sr)    # 10ms 步长
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # 检测语音段
        threshold = np.mean(energy) * 0.5
        speech_segments = []
        in_speech = False
        start = 0

        for i, e in enumerate(energy):
            time = i * hop_length / sr
            if e > threshold and not in_speech:
                start = time
                in_speech = True
            elif e <= threshold and in_speech:
                speech_segments.append((start, time))
                in_speech = False

        return speech_segments

    @staticmethod
    def event_aligned(keyframe_times: List[float], change_scores: List[float]):
        """事件对齐: 将音频与视觉变化事件对齐"""
        # 找到变化峰值
        peaks = []
        for i in range(1, len(change_scores) - 1):
            if change_scores[i] > change_scores[i-1] and change_scores[i] > change_scores[i+1]:
                if change_scores[i] > 0.05:  # 阈值
                    peaks.append(keyframe_times[i])
        return peaks

    @staticmethod
    def uniform_aligned(video_duration: float, num_segments: int):
        """均匀对齐: 等间隔切分"""
        segment_duration = video_duration / num_segments
        return [(i * segment_duration, (i + 1) * segment_duration)
                for i in range(num_segments)]
```

### 4.2 音频片段提取

```python
def extract_audio_segments(
    audio_path: str,
    keyframe_times: List[float],
    video_duration: float,
    strategy: str = "keyframe_centered"
) -> List[Tuple[np.ndarray, float, float]]:
    """
    根据关键帧提取对应的音频片段

    Returns:
        List of (audio_segment, start_time, end_time)
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=16000)
    segments = []

    if strategy == "keyframe_centered":
        # 以关键帧为中心，前后各取一定时长
        window = 2.0  # 前后各1秒
        for kf_time in keyframe_times:
            start = max(0, kf_time - window / 2)
            end = min(video_duration, kf_time + window / 2)
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segments.append((y[start_sample:end_sample], start, end))

    elif strategy == "between_keyframes":
        # 取相邻关键帧之间的音频
        for i in range(len(keyframe_times) - 1):
            start = keyframe_times[i]
            end = keyframe_times[i + 1]
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segments.append((y[start_sample:end_sample], start, end))

    elif strategy == "full_audio":
        # 完整音频
        segments.append((y, 0, video_duration))

    return segments
```

---

## 5. Prompt 优化技巧

### 5.1 时间戳格式化

```python
def format_timeline_for_prompt(
    keyframe_times: List[float],
    change_scores: List[float],
    events: List[ChangeEvent]
) -> str:
    """格式化时间线供 LLM 使用"""

    lines = ["## Video Timeline"]
    lines.append("")

    # 关键帧时间线
    lines.append("### Keyframes")
    for i, (time, score) in enumerate(zip(keyframe_times, change_scores)):
        mm = int(time // 60)
        ss = int(time % 60)
        change_level = "▲" if score > 0.1 else "→" if score > 0.03 else "─"
        lines.append(f"  [{mm:02d}:{ss:02d}] Frame {i+1} {change_level}")

    # 变化事件
    if events:
        lines.append("")
        lines.append("### Detected Events")
        for e in events:
            start_mm, start_ss = int(e.start_time // 60), int(e.start_time % 60)
            end_mm, end_ss = int(e.end_time // 60), int(e.end_time % 60)
            lines.append(f"  [{start_mm:02d}:{start_ss:02d}-{end_mm:02d}:{end_ss:02d}] {e.description}")

    return "\n".join(lines)
```

### 5.2 Few-Shot 示例

```python
FEW_SHOT_EXAMPLES = """
## Example 1: Cooking Video
Input: "What happens in this video?"
Timeline: [00:00] Kitchen setup → [00:15] Chopping vegetables → [00:30] Frying in pan
Output: "This video shows a cooking demonstration. At 00:00, we see a kitchen with ingredients laid out. By 00:15, the cook begins chopping vegetables. At 00:30, the prepared vegetables are added to a frying pan."

## Example 2: Meeting Recording
Input: "Summarize the discussion"
Timeline: [00:00] Introduction → [02:30] Main topic → [05:00] Q&A
Audio: Contains spoken dialogue
Output: "The meeting covers [topic]. The presenter discusses [main points] from 02:30-05:00, followed by audience questions."
"""
```

### 5.3 输出格式控制

```python
OUTPUT_FORMAT_PROMPTS = {
    "structured": """
Please respond in the following JSON format:
{
    "summary": "Brief summary",
    "scenes": [
        {"time": "MM:SS", "description": "What happens"},
        ...
    ],
    "audio_content": "Description of audio",
    "key_insights": ["insight1", "insight2"]
}
""",

    "narrative": """
Please describe the video as a story, using temporal language like:
"The video begins with...", "At around 30 seconds...", "Finally..."
""",

    "bullet_points": """
Provide your analysis as bullet points:
• Scene 1 (00:00-00:30): ...
• Scene 2 (00:30-01:00): ...
"""
}
```

---

## 6. 细节把控清单

### 6.1 视频预处理检查

- [ ] 视频格式是否支持 (mp4, mov, avi, webm)
- [ ] 分辨率是否在合理范围 (建议 ≤ 1080p)
- [ ] 时长是否超限 (建议 ≤ 5分钟)
- [ ] 是否包含音频轨道
- [ ] 帧率是否正常 (建议 24-60 FPS)

### 6.2 关键帧质量检查

- [ ] 避免模糊帧 (运动模糊检测)
- [ ] 避免过曝/过暗帧 (直方图分析)
- [ ] 避免重复内容帧 (相似度去重)
- [ ] 确保时间分布均匀

### 6.3 音频质量检查

- [ ] 采样率标准化 (16kHz)
- [ ] 音量归一化
- [ ] 静音段处理
- [ ] 噪声检测

### 6.4 Prompt 质量检查

- [ ] 时间戳格式一致 (MM:SS)
- [ ] 上下文信息完整
- [ ] 用户意图明确
- [ ] 输出格式指定

---

## 7. 参考资源

### 论文与研究
- [Audio-Visual LLM for Video Understanding (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/html/Shu_Audio-Visual_LLM_for_Video_Understanding_ICCVW_2025_paper.html)
- [Video-LLaMA (EMNLP 2023)](https://arxiv.org/abs/2306.02858)
- [Grounding-Prompter](https://arxiv.org/html/2312.17117v1)
- [VideoITG: Instructed Temporal Grounding](https://arxiv.org/html/2507.13353)
- [Watch and Listen: Audio-Visual-Speech Understanding](https://arxiv.org/abs/2505.18110)

### 官方文档
- [Gemini Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Prompt Engineering Guide](https://www.promptingguide.ai/models/gemini)

### 开源项目
- [Awesome-LLMs-for-Video-Understanding](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)

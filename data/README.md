# 测试数据

用于测试 Gemma 3n 多模态能力的示例数据。

## 图片 (data/images/)

| 文件 | 说明 | 来源 |
|------|------|------|
| `bee.jpg` | 蜜蜂在花上采蜜 | HuggingFace |
| `cat.jpg` | 猫咪照片 | Wikipedia |
| `dog.jpg` | 金毛犬照片 | Wikipedia |
| `food.jpg` | 食物展示 | Wikipedia |

## 音频 (data/audio/)

| 文件 | 说明 | 来源 |
|------|------|------|
| `mlk_speech.flac` | MLK "I Have a Dream" 演讲片段 (13秒) | HuggingFace |

## 使用方法

### 命令行测试

```bash
# 图像理解
cd examples
./run_2_vision.sh
# 输入: ../data/images/bee.jpg

# 音频转录
./run_3_audio.sh
# 输入: ../data/audio/mlk_speech.flac
```

### Web UI 测试

```bash
./webui/run.sh
# 浏览器打开 http://localhost:5000
# 点击 📷 上传图片，或点击 🎤 录制语音
```

## 添加更多测试数据

您可以将自己的图片和音频放到对应目录：

```bash
# 添加图片
cp your_image.jpg data/images/

# 添加音频 (支持 wav, mp3, flac, ogg)
cp your_audio.wav data/audio/
```

#!/bin/bash
# 运行所有示例 (按顺序)
cd "$(dirname "$0")"

echo "========================================"
echo "Gemma 3n 示例集 - 完整流程"
echo "========================================"

run_example() {
    echo ""
    echo "----------------------------------------"
    echo "运行: $1"
    echo "----------------------------------------"
    bash "$1"
    if [ $? -ne 0 ]; then
        echo "错误: $1 执行失败"
        exit 1
    fi
}

echo ""
echo "选择要运行的示例:"
echo "  1) 文本推理"
echo "  2) 图像理解"
echo "  3) 音频理解"
echo "  4) LoRA 微调"
echo "  5) 测试微调模型"
echo "  6) 合并并导出"
echo "  7) GGUF 推理"
echo "  a) 运行全部 (1-3)"
echo "  f) 完整微调流程 (4-6)"
echo ""
read -p "请输入选项: " choice

case $choice in
    1) run_example "run_text.sh" ;;
    2) run_example "run_vision.sh" ;;
    3) run_example "run_audio.sh" ;;
    4) run_example "run_finetune.sh" ;;
    5) run_example "run_test_finetuned.sh" ;;
    6) run_example "run_merge.sh" ;;
    7) run_example "run_gguf.sh" ;;
    a)
        run_example "run_text.sh"
        run_example "run_vision.sh"
        run_example "run_audio.sh"
        ;;
    f)
        run_example "run_finetune.sh"
        run_example "run_test_finetuned.sh"
        run_example "run_merge.sh"
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "完成!"
echo "========================================"

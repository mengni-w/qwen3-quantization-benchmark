#!/bin/bash
# Evaluation Runner Script / 评估运行脚本
# This script runs the complete evaluation workflow / 此脚本运行完整的评估工作流

set -e  # Exit on error / 出错时退出

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Qwen3 Quantization Benchmark Evaluation"
echo "Qwen3 量化基准测试评估"
echo "=========================================="
echo ""

# Check Python / 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found / 错误：未找到 python3"
    exit 1
fi

# Check if vLLM is installed / 检查是否安装了 vLLM
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Warning: vLLM not found. Please install it first:"
    echo "警告：未找到 vLLM。请先安装："
    echo "  pip install vllm[all]"
    echo ""
fi

# Run evaluation / 运行评估
echo "Starting evaluation / 开始评估..."
echo ""

python3 src/main.py --config configs/model_config.yaml

# Check if evaluation was successful / 检查评估是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "评估成功完成！"
    echo "=========================================="
    echo ""
    echo "Results are saved in the results/ directory"
    echo "结果已保存在 results/ 目录中"
    echo ""
    echo "To generate a report, run:"
    echo "要生成报告，请运行："
    echo "  python3 src/reports/report_generator.py results/<timestamp>/evaluation_results.json"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed. Check logs for details."
    echo "评估失败。请查看日志了解详情。"
    echo "=========================================="
    exit 1
fi


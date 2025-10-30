# Qwen3 Quantization Accuracy Benchmark

# Qwen3 量化精度基准测试

> **English | [中文](#中文)**

## English

### Overview

This project provides a comprehensive benchmarking framework for evaluating quantization methods on the Qwen3-30B-A3B-Instruct-2507 model. It compares the accuracy differences between Original Precision (FP16/BF16), FP8 quantization, and AWQ 4-bit quantization across multiple benchmark datasets to help determine the optimal model variant for AI workflows.

### Features

- **Multi-Quantization Comparison**: Evaluates three quantization approaches (Original, FP8, AWQ) side-by-side
- **Comprehensive Benchmarking**: Uses multiple datasets including GSM8K, HellaSwag, and MMLU
- **vLLM Integration**: Leverages vLLM's extensive benchmark tools and utilities
- **Detailed Reporting**: Generates professional, bilingual evaluation reports
- **Automated Workflow**: Fully automated evaluation pipeline with server management

### Project Structure

```
qwen3-quantization-benchmark/
├── configs/              # Configuration files / 配置文件
│   └── model_config.yaml
├── src/                  # Source code / 源代码
│   ├── evaluators/       # Evaluation modules / 评估模块
│   ├── utils/            # Utility functions / 工具函数
│   ├── reports/          # Report generation / 报告生成
│   └── main.py          # Main evaluation script / 主评估脚本
├── tests/                # Test files / 测试文件
├── results/              # Evaluation results / 评估结果
└── README.md
```

### Installation

#### Prerequisites

- Python 3.8+
- CUDA-capable GPU with sufficient VRAM
- vLLM installed (with quantization support)

#### Setup

1. Clone or download this project:
```bash
cd ~/Desktop/qwen3-quantization-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure vLLM is properly installed with quantization support:
```bash
pip install vllm[all]
```

### Usage

#### Basic Evaluation

Run the complete evaluation workflow:

```bash
python src/main.py --config configs/model_config.yaml
```

#### Evaluate Specific Variant

Evaluate only a specific quantization variant:

```bash
python src/main.py --config configs/model_config.yaml --variant fp8
```

#### Generate Report

After evaluation, generate a comprehensive report:

```bash
python src/reports/report_generator.py results/<timestamp>/evaluation_results.json --language both
```

### Configuration

Edit `configs/model_config.yaml` to customize:

- Model variants to evaluate
- Benchmark datasets and sample sizes
- Server configuration (GPU memory, parallelism)
- Generation parameters (temperature, max tokens)

### Evaluation Datasets

- **GSM8K**: Grade School Math 8K - Mathematical reasoning (1319 questions)
- **HellaSwag**: Common sense reasoning (10,000 samples)
- **MMLU**: Massive Multitask Language Understanding (15,908 samples)

### Results

Evaluation results are saved in `results/<timestamp>/` with:

- `evaluation_results.json`: Complete results in JSON format
- `reports/evaluation_report_*.md`: Comprehensive Markdown reports

### Methodology

1. **Server Launch**: For each variant, launch a vLLM server with appropriate quantization settings
2. **Dataset Evaluation**: Run evaluations on configured benchmark datasets
3. **Metrics Collection**: Collect accuracy, latency, and throughput metrics
4. **Comparison Analysis**: Compare results across quantization methods
5. **Report Generation**: Generate detailed bilingual reports

### Key Metrics

- **Accuracy**: Percentage of correct predictions on benchmark datasets
- **Latency**: Time taken per evaluation
- **Throughput**: Evaluations per second (QPS)
- **Invalid Rate**: Percentage of invalid/unparseable responses

### Interpreting Results

- **Original Precision**: Baseline accuracy, highest memory usage
- **FP8 Quantization**: Balanced accuracy and performance, requires FP8-capable hardware
- **AWQ 4-bit**: Maximum memory reduction, potentially highest performance gain

### Best Practices

1. **Hardware Requirements**: Ensure sufficient GPU memory for the 30B model
2. **Warm-up**: Allow server warm-up time before evaluation
3. **Reproducibility**: Use fixed random seeds for consistent results
4. **Monitoring**: Monitor GPU memory and temperature during evaluation

### Troubleshooting

**Server won't start:**
- Check GPU memory availability
- Verify vLLM installation
- Check model path and quantization compatibility

**Low accuracy:**
- Verify dataset loading
- Check generation parameters (temperature, max_tokens)
- Ensure proper model variant configuration

**Memory errors:**
- Reduce `gpu_memory_utilization` in config
- Use tensor parallelism if multiple GPUs available
- Consider evaluating variants sequentially

### Contributing

This project is designed for evaluating quantization methods. Contributions for additional datasets, evaluation metrics, or quantization methods are welcome.

### License

SPDX-License-Identifier: Apache-2.0

---

## 中文

### 概述

本项目提供了一个全面的基准测试框架，用于评估 Qwen3-30B-A3B-Instruct-2507 模型的量化方法。它比较了原精度（FP16/BF16）、FP8 量化和 AWQ 4位量化在多个基准测试数据集上的精度差异，以帮助确定 AI 工作流中的最佳模型变体。

### 特性

- **多量化比较**：并行评估三种量化方法（原精度、FP8、AWQ）
- **全面基准测试**：使用包括 GSM8K、HellaSwag 和 MMLU 在内的多个数据集
- **vLLM 集成**：利用 vLLM 广泛的基准测试工具和实用程序
- **详细报告**：生成专业的中英文双语评估报告
- **自动化工作流**：完全自动化的评估流程，包含服务器管理

### 项目结构

```
qwen3-quantization-benchmark/
├── configs/              # 配置文件
│   └── model_config.yaml
├── src/                  # 源代码
│   ├── evaluators/       # 评估模块
│   ├── utils/            # 工具函数
│   ├── reports/          # 报告生成
│   └── main.py          # 主评估脚本
├── tests/                # 测试文件
├── results/              # 评估结果
└── README.md
```

### 安装

#### 前置要求

- Python 3.8+
- 支持 CUDA 的 GPU，具有足够的显存
- 已安装 vLLM（支持量化）

#### 设置

1. 克隆或下载此项目：
```bash
cd ~/Desktop/qwen3-quantization-benchmark
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 确保 vLLM 已正确安装并支持量化：
```bash
pip install vllm[all]
```

### 使用方法

#### 基本评估

运行完整的评估工作流：

```bash
python src/main.py --config configs/model_config.yaml
```

#### 评估特定变体

仅评估特定的量化变体：

```bash
python src/main.py --config configs/model_config.yaml --variant fp8
```

#### 生成报告

评估后，生成综合报告：

```bash
python src/reports/report_generator.py results/<timestamp>/evaluation_results.json --language both
```

### 配置

编辑 `configs/model_config.yaml` 以自定义：

- 要评估的模型变体
- 基准测试数据集和样本大小
- 服务器配置（GPU 内存、并行度）
- 生成参数（温度、最大 token 数）

### 评估数据集

- **GSM8K**：小学数学 8K - 数学推理（1319 个问题）
- **HellaSwag**：常识推理（10,000 个样本）
- **MMLU**：大规模多任务语言理解（15,908 个样本）

### 结果

评估结果保存在 `results/<timestamp>/` 中，包含：

- `evaluation_results.json`：JSON 格式的完整结果
- `reports/evaluation_report_*.md`：详细的 Markdown 报告

### 方法论

1. **服务器启动**：对于每个变体，使用适当的量化设置启动 vLLM 服务器
2. **数据集评估**：在配置的基准测试数据集上运行评估
3. **指标收集**：收集精度、延迟和吞吐量指标
4. **比较分析**：比较不同量化方法的结果
5. **报告生成**：生成详细的中英文双语报告

### 关键指标

- **精度**：基准测试数据集上正确预测的百分比
- **延迟**：每次评估所需时间
- **吞吐量**：每秒评估次数（QPS）
- **无效率**：无效/无法解析响应的百分比

### 结果解读

- **原精度**：基线精度，最高内存使用
- **FP8 量化**：精度和性能平衡，需要支持 FP8 的硬件
- **AWQ 4位**：最大内存减少，潜在最高性能提升

### 最佳实践

1. **硬件要求**：确保 GPU 内存足以运行 30B 模型
2. **预热**：评估前允许服务器预热时间
3. **可重现性**：使用固定随机种子以获得一致结果
4. **监控**：评估期间监控 GPU 内存和温度

### 故障排除

**服务器无法启动：**
- 检查 GPU 内存可用性
- 验证 vLLM 安装
- 检查模型路径和量化兼容性

**精度低：**
- 验证数据集加载
- 检查生成参数（温度、最大 token 数）
- 确保正确的模型变体配置

**内存错误：**
- 在配置中减少 `gpu_memory_utilization`
- 如果有多个 GPU，使用张量并行
- 考虑按顺序评估变体

### 贡献

本项目旨在评估量化方法。欢迎为额外的数据集、评估指标或量化方法做出贡献。

### 许可证

SPDX-License-Identifier: Apache-2.0

---

## GitHub Description

Comprehensive benchmarking framework for evaluating quantization methods (Original Precision, FP8, AWQ) on Qwen3-30B-A3B-Instruct-2507 model. Compare accuracy across multiple datasets (GSM8K, HellaSwag, MMLU) using vLLM tools. Includes automated evaluation pipeline, detailed bilingual reports, and performance analysis.


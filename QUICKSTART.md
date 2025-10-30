# Quick Start Guide / 快速开始指南

## English

### Prerequisites / 前置要求

1. **Hardware**
   - NVIDIA GPU with CUDA support
   - At least 40GB VRAM (for 30B model)
   - FP8 support (H100/Ada Lovelace) for FP8 quantization

2. **Software**
   - Python 3.8 or higher
   - CUDA toolkit (version matching your GPU)
   - vLLM installed with quantization support

### Installation Steps / 安装步骤

#### Step 1: Navigate to Project / 步骤 1：导航到项目

```bash
cd ~/Desktop/qwen3-quantization-benchmark
```

#### Step 2: Create Virtual Environment (Recommended) / 步骤 2：创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies / 步骤 3：安装依赖

```bash
pip install -r requirements.txt
```

#### Step 4: Install vLLM / 步骤 4：安装 vLLM

```bash
# For CUDA 11.8
pip install vllm

# For CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# Or install with all features
pip install "vllm[all]"
```

### Running Evaluation / 运行评估

#### Option 1: Using the Script / 选项 1：使用脚本

```bash
./run_evaluation.sh
```

#### Option 2: Direct Python Execution / 选项 2：直接 Python 执行

```bash
python src/main.py --config configs/model_config.yaml
```

#### Option 3: Evaluate Single Variant / 选项 3：评估单个变体

```bash
python src/main.py --config configs/model_config.yaml --variant fp8
```

### Expected Runtime / 预期运行时间

- **Original Precision**: ~30-60 minutes per dataset
- **FP8 Quantization**: ~25-50 minutes per dataset
- **AWQ Quantization**: ~20-40 minutes per dataset

Total evaluation time: **3-6 hours** (depending on hardware)

### Generating Reports / 生成报告

After evaluation completes, generate the report:

```bash
# Find the latest results directory
LATEST_RESULTS=$(ls -td results/*/ | head -1)

# Generate bilingual report
python src/reports/report_generator.py ${LATEST_RESULTS}evaluation_results.json --language both
```

### Troubleshooting / 故障排除

#### Issue: vLLM server won't start / 问题：vLLM 服务器无法启动

**Solution / 解决方案**:
- Check GPU memory: `nvidia-smi`
- Reduce `gpu_memory_utilization` in config
- Ensure CUDA version matches vLLM installation

#### Issue: Out of memory errors / 问题：内存不足错误

**Solution / 解决方案**:
- Reduce `max_model_len` in config
- Use tensor parallelism (multiple GPUs)
- Evaluate variants sequentially

#### Issue: Import errors / 问题：导入错误

**Solution / 解决方案**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

### Configuration Customization / 配置自定义

Edit `configs/model_config.yaml` to:

- Change model name
- Adjust dataset sample sizes
- Modify server settings (GPU memory, parallelism)
- Update generation parameters

### Next Steps / 后续步骤

1. Review `README.md` for detailed documentation
2. Check `IMPLEMENTATION.md` for technical details
3. Read `EXAMPLE_RESULTS.md` for result format
4. Customize configuration for your needs

---

## 中文

### 前置要求

1. **硬件**
   - 支持 CUDA 的 NVIDIA GPU
   - 至少 40GB 显存（用于 30B 模型）
   - FP8 支持（H100/Ada Lovelace）用于 FP8 量化

2. **软件**
   - Python 3.8 或更高版本
   - CUDA 工具包（版本与 GPU 匹配）
   - 安装了支持量化的 vLLM

### 安装步骤

#### 步骤 1：导航到项目

```bash
cd ~/Desktop/qwen3-quantization-benchmark
```

#### 步骤 2：创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 步骤 3：安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤 4：安装 vLLM

```bash
# CUDA 11.8
pip install vllm

# CUDA 12.1
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# 或安装所有功能
pip install "vllm[all]"
```

### 运行评估

#### 选项 1：使用脚本

```bash
./run_evaluation.sh
```

#### 选项 2：直接 Python 执行

```bash
python src/main.py --config configs/model_config.yaml
```

#### 选项 3：评估单个变体

```bash
python src/main.py --config configs/model_config.yaml --variant fp8
```

### 预期运行时间

- **原精度**：每个数据集约 30-60 分钟
- **FP8 量化**：每个数据集约 25-50 分钟
- **AWQ 量化**：每个数据集约 20-40 分钟

总评估时间：**3-6 小时**（取决于硬件）

### 生成报告

评估完成后，生成报告：

```bash
# 找到最新的结果目录
LATEST_RESULTS=$(ls -td results/*/ | head -1)

# 生成双语报告
python src/reports/report_generator.py ${LATEST_RESULTS}evaluation_results.json --language both
```

### 故障排除

#### 问题：vLLM 服务器无法启动

**解决方案**:
- 检查 GPU 内存：`nvidia-smi`
- 在配置中减少 `gpu_memory_utilization`
- 确保 CUDA 版本与 vLLM 安装匹配

#### 问题：内存不足错误

**解决方案**:
- 在配置中减少 `max_model_len`
- 使用张量并行（多个 GPU）
- 按顺序评估变体

#### 问题：导入错误

**解决方案**:
- 确保虚拟环境已激活
- 重新安装依赖：`pip install -r requirements.txt`
- 检查 Python 版本：`python --version`

### 配置自定义

编辑 `configs/model_config.yaml` 以：

- 更改模型名称
- 调整数据集样本大小
- 修改服务器设置（GPU 内存、并行度）
- 更新生成参数

### 后续步骤

1. 查看 `README.md` 了解详细文档
2. 查看 `IMPLEMENTATION.md` 了解技术细节
3. 阅读 `EXAMPLE_RESULTS.md` 了解结果格式
4. 根据需要自定义配置


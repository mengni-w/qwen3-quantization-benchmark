#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
vLLM Benchmark Utilities / vLLM 基准测试工具

This module provides utilities for integrating vLLM benchmark tools
into the quantization evaluation workflow.

该模块提供将 vLLM 基准测试工具集成到量化评估工作流的实用程序。
"""

import json
import time
import subprocess
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
import asyncio
from collections import defaultdict

# Note: vLLM should be installed as a package / 注意：vLLM 应作为包安装
# The vLLM import will work when vLLM is installed in the environment
# 当 vLLM 安装在环境中时，vLLM 导入将正常工作

logger = logging.getLogger(__name__)


class VLLMBenchmarkClient:
    """
    Client for interacting with vLLM server and running benchmarks
    用于与 vLLM 服务器交互并运行基准测试的客户端
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initialize the benchmark client / 初始化基准测试客户端
        
        Args:
            host: Server host address / 服务器主机地址
            port: Server port number / 服务器端口号
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry / 异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit / 异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def check_server_ready(self, max_retries: int = 30) -> bool:
        """
        Check if vLLM server is ready / 检查 vLLM 服务器是否就绪
        
        Args:
            max_retries: Maximum number of retry attempts / 最大重试次数
            
        Returns:
            True if server is ready, False otherwise / 如果服务器就绪返回 True，否则返回 False
        """
        for i in range(max_retries):
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        logger.info("Server is ready / 服务器已就绪")
                        return True
            except Exception as e:
                if i < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                logger.error(f"Server not ready after {max_retries} attempts / "
                           f"服务器在 {max_retries} 次尝试后仍未就绪: {e}")
                return False
        return False
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using vLLM API / 使用 vLLM API 生成文本
        
        Args:
            prompt: Input prompt / 输入提示
            **kwargs: Additional generation parameters / 其他生成参数
            
        Returns:
            API response dictionary / API 响应字典
        """
        data = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        if "seed" in kwargs:
            data["seed"] = kwargs["seed"]
        
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts / 为多个提示生成文本
        
        Args:
            prompts: List of input prompts / 输入提示列表
            **kwargs: Additional generation parameters / 其他生成参数
            
        Returns:
            List of API responses / API 响应列表
        """
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)


class LatencyCollector:
    """
    Collect latency metrics during benchmark runs
    在基准测试运行期间收集延迟指标
    """
    
    def __init__(self):
        """Initialize latency collector / 初始化延迟收集器"""
        self.latencies: List[float] = []
        self.start_time: Optional[float] = None
    
    def start(self):
        """Start timing / 开始计时"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop timing and record latency / 停止计时并记录延迟
        
        Returns:
            Elapsed time in seconds / 经过的时间（秒）
        """
        if self.start_time is None:
            raise ValueError("Timer not started / 计时器未启动")
        elapsed = time.perf_counter() - self.start_time
        self.latencies.append(elapsed)
        self.start_time = None
        return elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get latency statistics / 获取延迟统计信息
        
        Returns:
            Dictionary with mean, median, p50, p95, p99 latencies
            包含平均值、中位数、p50、p95、p99 延迟的字典
        """
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "mean": sum(self.latencies) / n,
            "median": sorted_latencies[n // 2],
            "p50": sorted_latencies[int(n * 0.50)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "min": min(self.latencies),
            "max": max(self.latencies),
        }


def launch_vllm_server(
    model_name: str,
    quantization: Optional[str] = None,
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    port: int = 8000,
    **kwargs
) -> subprocess.Popen:
    """
    Launch vLLM server process / 启动 vLLM 服务器进程
    
    Args:
        model_name: Model identifier / 模型标识符
        quantization: Quantization method (None, 'fp8', 'awq') / 量化方法
        kv_cache_dtype: KV cache data type / KV 缓存数据类型
        tensor_parallel_size: Tensor parallelism size / 张量并行大小
        max_model_len: Maximum model length / 最大模型长度
        gpu_memory_utilization: GPU memory utilization / GPU 内存利用率
        port: Server port / 服务器端口
        **kwargs: Additional vLLM arguments / 其他 vLLM 参数
        
    Returns:
        Process object / 进程对象
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    
    if quantization:
        cmd.extend(["--quantization", quantization])
    
    if kv_cache_dtype != "auto":
        cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
    
    # Add additional arguments / 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    logger.info(f"Launching vLLM server / 启动 vLLM 服务器: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def load_benchmark_dataset(dataset_name: str, num_samples: Optional[int] = None):
    """
    Load benchmark dataset / 加载基准测试数据集
    
    Args:
        dataset_name: Name of the dataset / 数据集名称
        num_samples: Number of samples to load / 要加载的样本数
        
    Returns:
        Dataset object / 数据集对象
    """
    # This is a placeholder - actual implementation would load from HuggingFace
    # 这是一个占位符 - 实际实现将从 HuggingFace 加载
    logger.info(f"Loading dataset / 加载数据集: {dataset_name}")
    return []


def save_results(results: Dict[str, Any], output_path: Path):
    """
    Save benchmark results to JSON file / 将基准测试结果保存到 JSON 文件
    
    Args:
        results: Results dictionary / 结果字典
        output_path: Output file path / 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to / 结果已保存到: {output_path}")


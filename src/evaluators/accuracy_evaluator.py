#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Accuracy Evaluator / 精度评估器

This module implements accuracy evaluation using various benchmark datasets
for comparing quantization methods.

该模块实现了使用各种基准测试数据集进行精度评估，用于比较量化方法。
"""

import asyncio
import json
import logging
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import aiohttp
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

# Constants / 常量
INVALID_ANSWER = -9999999


class GSM8KEvaluator:
    """
    GSM8K (Grade School Math 8K) accuracy evaluator
    GSM8K（小学数学8K）精度评估器
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialize GSM8K evaluator / 初始化 GSM8K 评估器
        
        Args:
            base_url: Base URL of the vLLM server / vLLM 服务器的基础 URL
        """
        self.base_url = base_url
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
    
    def _download_and_cache_file(self, url: str, filename: Path) -> Path:
        """
        Download and cache a file from URL / 从 URL 下载并缓存文件
        
        Args:
            url: URL to download from / 下载源 URL
            filename: Local filename / 本地文件名
            
        Returns:
            Path to cached file / 缓存文件路径
        """
        if filename.exists():
            return filename
        
        logger.info(f"Downloading / 下载: {url} -> {filename}")
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        
        return filename
    
    def _load_gsm8k_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load GSM8K train and test datasets / 加载 GSM8K 训练和测试数据集
        
        Returns:
            Tuple of (train_data, test_data) / (训练数据, 测试数据) 元组
        """
        cache_dir = Path.home() / ".cache" / "qwen3_benchmark"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        train_url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "master/grade_school_math/data/train.jsonl"
        )
        test_url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/"
            "master/grade_school_math/data/test.jsonl"
        )
        
        train_file = self._download_and_cache_file(
            train_url, cache_dir / "gsm8k_train.jsonl"
        )
        test_file = self._download_and_cache_file(
            test_url, cache_dir / "gsm8k_test.jsonl"
        )
        
        train_data = []
        test_data = []
        
        with open(train_file) as f:
            for line in f:
                if not line.startswith("#"):
                    train_data.append(json.loads(line))
        
        with open(test_file) as f:
            for line in f:
                if not line.startswith("#"):
                    test_data.append(json.loads(line))
        
        return train_data, test_data
    
    def _extract_answer(self, answer_str: str) -> int:
        """
        Extract numerical answer from response text / 从响应文本中提取数字答案
        
        Args:
            answer_str: Response text / 响应文本
            
        Returns:
            Extracted answer value / 提取的答案值
        """
        answer_str = answer_str.replace(",", "")
        numbers = re.findall(r"\d+", answer_str)
        if len(numbers) < 1:
            return INVALID_ANSWER
        try:
            return ast.literal_eval(numbers[-1])
        except SyntaxError:
            return INVALID_ANSWER
    
    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = 42,
    ) -> str:
        """
        Call vLLM API for text generation / 调用 vLLM API 进行文本生成
        
        Args:
            prompt: Input prompt / 输入提示
            temperature: Sampling temperature / 采样温度
            max_tokens: Maximum tokens to generate / 最大生成 token 数
            seed: Random seed / 随机种子
            
        Returns:
            Generated text / 生成的文本
        """
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        }
        if seed is not None:
            data["seed"] = seed
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/completions", json=data
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["text"]
        except Exception as e:
            logger.error(f"API call error / API 调用错误: {e}")
            return ""
    
    async def evaluate(
        self,
        num_questions: int = 1319,
        num_shots: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Evaluate GSM8K accuracy / 评估 GSM8K 精度
        
        Args:
            num_questions: Number of questions to evaluate / 要评估的问题数
            num_shots: Number of few-shot examples / few-shot 示例数
            max_tokens: Maximum tokens per response / 每个响应的最大 token 数
            temperature: Sampling temperature / 采样温度
            seed: Random seed / 随机种子
            
        Returns:
            Evaluation results dictionary / 评估结果字典
        """
        import time
        
        # Load datasets / 加载数据集
        train_data, test_data = self._load_gsm8k_data()
        num_questions = min(num_questions, len(test_data))
        
        # Build few-shot examples / 构建 few-shot 示例
        few_shot_examples = ""
        for i in range(num_shots):
            few_shot_examples += (
                f"Question: {train_data[i]['question']}\n"
                f"Answer: {train_data[i]['answer']}\n\n"
            )
        
        # Prepare test questions and labels / 准备测试问题和标签
        questions = []
        labels = []
        for i in range(num_questions):
            questions.append(f"Question: {test_data[i]['question']}\nAnswer:")
            labels.append(self._extract_answer(test_data[i]["answer"]))
        
        assert all(label != INVALID_ANSWER for label in labels), (
            "Some labels are invalid / 某些标签无效"
        )
        
        # Run evaluation / 运行评估
        async def get_answer(i: int) -> str:
            prompt = few_shot_examples + questions[i]
            return await self._call_api(
                prompt, temperature=temperature, max_tokens=max_tokens, seed=seed
            )
        
        logger.info(
            f"Evaluating GSM8K / 评估 GSM8K: {num_questions} questions, "
            f"{num_shots}-shot"
        )
        
        start_time = time.perf_counter()
        tasks = [get_answer(i) for i in range(num_questions)]
        responses = await tqdm.gather(*tasks, desc="Evaluating / 评估中")
        elapsed_time = time.perf_counter() - start_time
        
        # Compute metrics / 计算指标
        predictions = [self._extract_answer(resp) for resp in responses]
        correct = np.array(predictions) == np.array(labels)
        accuracy = np.mean(correct)
        invalid_rate = np.mean(np.array(predictions) == INVALID_ANSWER)
        
        return {
            "accuracy": float(accuracy),
            "invalid_rate": float(invalid_rate),
            "num_correct": int(np.sum(correct)),
            "num_total": num_questions,
            "latency_seconds": elapsed_time,
            "questions_per_second": num_questions / elapsed_time,
            "num_shots": num_shots,
            "max_tokens": max_tokens,
        }


class HellaSwagEvaluator:
    """
    HellaSwag common sense reasoning evaluator
    HellaSwag 常识推理评估器
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialize HellaSwag evaluator / 初始化 HellaSwag 评估器
        
        Args:
            base_url: Base URL of the vLLM server / vLLM 服务器的基础 URL
        """
        self.base_url = base_url
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
    
    async def evaluate(
        self,
        num_samples: int = 10000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate HellaSwag accuracy / 评估 HellaSwag 精度
        
        Args:
            num_samples: Number of samples to evaluate / 要评估的样本数
            **kwargs: Additional evaluation parameters / 其他评估参数
            
        Returns:
            Evaluation results dictionary / 评估结果字典
        """
        # Placeholder implementation - would integrate with lm-eval-harness
        # 占位符实现 - 将与 lm-eval-harness 集成
        logger.info(f"Evaluating HellaSwag / 评估 HellaSwag: {num_samples} samples")
        return {
            "accuracy": 0.0,
            "num_samples": num_samples,
        }


class MMLUEvaluator:
    """
    MMLU (Massive Multitask Language Understanding) evaluator
    MMLU（大规模多任务语言理解）评估器
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialize MMLU evaluator / 初始化 MMLU 评估器
        
        Args:
            base_url: Base URL of the vLLM server / vLLM 服务器的基础 URL
        """
        self.base_url = base_url
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
    
    async def evaluate(
        self,
        num_samples: int = 15908,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate MMLU accuracy / 评估 MMLU 精度
        
        Args:
            num_samples: Number of samples to evaluate / 要评估的样本数
            **kwargs: Additional evaluation parameters / 其他评估参数
            
        Returns:
            Evaluation results dictionary / 评估结果字典
        """
        # Placeholder implementation - would integrate with lm-eval-harness
        # 占位符实现 - 将与 lm-eval-harness 集成
        logger.info(f"Evaluating MMLU / 评估 MMLU: {num_samples} samples")
        return {
            "accuracy": 0.0,
            "num_samples": num_samples,
        }


def get_evaluator(dataset_name: str, base_url: str = "http://127.0.0.1:8000"):
    """
    Get appropriate evaluator for dataset / 获取数据集的相应评估器
    
    Args:
        dataset_name: Name of the dataset / 数据集名称
        base_url: Base URL of the vLLM server / vLLM 服务器的基础 URL
        
    Returns:
        Evaluator instance / 评估器实例
    """
    evaluators = {
        "gsm8k": GSM8KEvaluator,
        "hellaswag": HellaSwagEvaluator,
        "mmlu": MMLUEvaluator,
    }
    
    evaluator_class = evaluators.get(dataset_name.lower())
    if evaluator_class is None:
        raise ValueError(
            f"Unknown dataset / 未知数据集: {dataset_name}. "
            f"Available: {list(evaluators.keys())}"
        )
    
    return evaluator_class(base_url)


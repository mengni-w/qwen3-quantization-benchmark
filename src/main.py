#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Main Evaluation Script / 主评估脚本

This script orchestrates the complete evaluation workflow for comparing
quantization methods on Qwen3-30B-A3B-Instruct-2507.

该脚本编排完整的评估工作流，用于比较 Qwen3-30B-A3B-Instruct-2507 的量化方法。
"""

import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Handle imports with fallback for different execution contexts
# 处理导入，为不同的执行上下文提供回退
# Add project root to path / 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.evaluators.accuracy_evaluator import get_evaluator
    from src.utils.vllm_benchmark_utils import (
        VLLMBenchmarkClient,
        launch_vllm_server,
        save_results,
    )
except ImportError:
    # Fallback for direct execution / 直接执行的回退
    from evaluators.accuracy_evaluator import get_evaluator
    from utils.vllm_benchmark_utils import (
        VLLMBenchmarkClient,
        launch_vllm_server,
        save_results,
    )

# Setup logging / 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class QuantizationEvaluator:
    """
    Main evaluator class for comparing quantization methods
    用于比较量化方法的主评估器类
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize evaluator with configuration / 使用配置初始化评估器
        
        Args:
            config_path: Path to configuration YAML file / 配置 YAML 文件路径
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config["model"]
        self.datasets_config = self.config["datasets"]
        self.server_config = self.config["server"]
        self.generation_config = self.config["generation"]
        self.benchmark_config = self.config["benchmark"]
        
        self.results_dir = Path("results") / time.strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.server_process: Optional[subprocess.Popen] = None
        self.results: Dict[str, Dict] = {}
    
    def _start_server(self, variant: Dict) -> subprocess.Popen:
        """
        Start vLLM server for given variant / 为给定变体启动 vLLM 服务器
        
        Args:
            variant: Model variant configuration / 模型变体配置
            
        Returns:
            Server process / 服务器进程
        """
        logger.info(
            f"Starting server for variant / 为变体启动服务器: {variant['name']} "
            f"({variant['description']})"
        )
        
        model_name = self.model_config["base_name"]
        if variant["quantization"] == "awq":
            # For AWQ, the model name might need to point to quantized checkpoint
            # 对于 AWQ，模型名称可能需要指向量化检查点
            model_name = model_name  # Could be modified if needed / 如果需要可以修改
        
        process = launch_vllm_server(
            model_name=model_name,
            quantization=variant.get("quantization"),
            kv_cache_dtype=variant.get("kv_cache_dtype", "auto"),
            tensor_parallel_size=self.server_config["tensor_parallel_size"],
            max_model_len=self.server_config["max_model_len"],
            gpu_memory_utilization=self.server_config["gpu_memory_utilization"],
            port=self.server_config["port"],
            trust_remote_code=self.server_config["trust_remote_code"],
        )
        
        return process
    
    def _stop_server(self):
        """Stop vLLM server / 停止 vLLM 服务器"""
        if self.server_process:
            logger.info("Stopping server / 停止服务器")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
    
    async def _wait_for_server(self, max_wait: int = 300):
        """
        Wait for server to be ready / 等待服务器就绪
        
        Args:
            max_wait: Maximum wait time in seconds / 最大等待时间（秒）
        """
        base_url = (
            f"http://{self.server_config['host']}:{self.server_config['port']}"
        )
        async with VLLMBenchmarkClient(
            self.server_config["host"], self.server_config["port"]
        ) as client:
            await client.check_server_ready(max_retries=max_wait // 2)
    
    async def _evaluate_dataset(
        self, dataset_name: str, dataset_config: Dict
    ) -> Dict[str, Any]:
        """
        Evaluate a single dataset / 评估单个数据集
        
        Args:
            dataset_name: Name of the dataset / 数据集名称
            dataset_config: Dataset configuration / 数据集配置
            
        Returns:
            Evaluation results / 评估结果
        """
        logger.info(f"Evaluating dataset / 评估数据集: {dataset_name}")
        
        base_url = (
            f"http://{self.server_config['host']}:{self.server_config['port']}"
        )
        
        evaluator_class = get_evaluator(dataset_name, base_url)
        
        async with evaluator_class as evaluator:
            results = await evaluator.evaluate(
                num_samples=dataset_config.get("num_samples"),
                **self.generation_config,
            )
        
        return results
    
    async def _evaluate_variant(self, variant: Dict) -> Dict[str, Any]:
        """
        Evaluate a single model variant / 评估单个模型变体
        
        Args:
            variant: Model variant configuration / 模型变体配置
            
        Returns:
            Evaluation results for this variant / 此变体的评估结果
        """
        variant_name = variant["name"]
        logger.info(
            f"\n{'='*60}\n"
            f"Evaluating variant / 评估变体: {variant_name}\n"
            f"Description / 描述: {variant['description']}\n"
            f"{'='*60}\n"
        )
        
        # Start server / 启动服务器
        self.server_process = self._start_server(variant)
        
        try:
            # Wait for server / 等待服务器
            await self._wait_for_server()
            
            # Evaluate each dataset / 评估每个数据集
            variant_results = {
                "variant": variant_name,
                "description": variant["description"],
                "quantization": variant.get("quantization"),
                "datasets": {},
                "timestamp": time.time(),
            }
            
            for dataset_config in self.datasets_config:
                dataset_name = dataset_config["name"]
                try:
                    results = await self._evaluate_dataset(
                        dataset_name, dataset_config
                    )
                    variant_results["datasets"][dataset_name] = results
                    logger.info(
                        f"{dataset_name} results / {dataset_name} 结果: "
                        f"Accuracy = {results.get('accuracy', 'N/A'):.4f}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error evaluating {dataset_name} / "
                        f"评估 {dataset_name} 时出错: {e}",
                        exc_info=True,
                    )
                    variant_results["datasets"][dataset_name] = {
                        "error": str(e),
                    }
            
            return variant_results
        
        finally:
            # Stop server / 停止服务器
            self._stop_server()
            # Wait a bit before starting next variant / 在启动下一个变体前等待一会儿
            await asyncio.sleep(5)
    
    async def run_evaluation(self):
        """
        Run complete evaluation for all variants / 运行所有变体的完整评估
        """
        logger.info("Starting evaluation / 开始评估")
        logger.info(f"Results will be saved to / 结果将保存到: {self.results_dir}")
        
        all_results = {
            "model": self.model_config["base_name"],
            "config": self.config,
            "variants": {},
            "summary": {},
        }
        
        # Evaluate each variant / 评估每个变体
        for variant in self.model_config["variants"]:
            try:
                variant_results = await self._evaluate_variant(variant)
                all_results["variants"][variant["name"]] = variant_results
            except Exception as e:
                logger.error(
                    f"Error evaluating variant {variant['name']} / "
                    f"评估变体 {variant['name']} 时出错: {e}",
                    exc_info=True,
                )
                all_results["variants"][variant["name"]] = {
                    "error": str(e),
                }
        
        # Compute summary / 计算摘要
        all_results["summary"] = self._compute_summary(all_results["variants"])
        
        # Save results / 保存结果
        results_file = self.results_dir / "evaluation_results.json"
        save_results(all_results, results_file)
        
        logger.info(f"Evaluation complete / 评估完成")
        logger.info(f"Results saved to / 结果已保存到: {results_file}")
        
        return all_results
    
    def _compute_summary(self, variants_results: Dict) -> Dict[str, Any]:
        """
        Compute summary statistics / 计算摘要统计信息
        
        Args:
            variants_results: Results for all variants / 所有变体的结果
            
        Returns:
            Summary dictionary / 摘要字典
        """
        summary = {}
        
        # Collect all dataset names / 收集所有数据集名称
        dataset_names = set()
        for variant_result in variants_results.values():
            if "datasets" in variant_result:
                dataset_names.update(variant_result["datasets"].keys())
        
        # Compare accuracy across variants for each dataset
        # 比较每个数据集在不同变体间的精度
        for dataset_name in dataset_names:
            summary[dataset_name] = {}
            accuracies = {}
            
            for variant_name, variant_result in variants_results.items():
                if (
                    "datasets" in variant_result
                    and dataset_name in variant_result["datasets"]
                ):
                    dataset_result = variant_result["datasets"][dataset_name]
                    if "accuracy" in dataset_result:
                        accuracies[variant_name] = dataset_result["accuracy"]
            
            if accuracies:
                summary[dataset_name]["accuracies"] = accuracies
                summary[dataset_name]["best"] = max(
                    accuracies.items(), key=lambda x: x[1]
                )[0]
                summary[dataset_name]["worst"] = min(
                    accuracies.items(), key=lambda x: x[1]
                )[0]
                
                # Compute relative degradation / 计算相对退化
                if "original" in accuracies:
                    original_acc = accuracies["original"]
                    summary[dataset_name]["degradation"] = {
                        variant: (original_acc - acc) / original_acc
                        for variant, acc in accuracies.items()
                        if variant != "original"
                    }
        
        return summary


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully / 优雅处理 Ctrl+C"""
    logger.info("\nInterrupted / 中断")
    sys.exit(0)


def main():
    """Main entry point / 主入口点"""
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Evaluate quantization methods for Qwen3-30B-A3B-Instruct-2507 / "
                    "评估 Qwen3-30B-A3B-Instruct-2507 的量化方法"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to configuration file / 配置文件路径",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Evaluate only specific variant / 仅评估特定变体",
    )
    
    args = parser.parse_args()
    
    evaluator = QuantizationEvaluator(args.config)
    
    # Filter variants if specified / 如果指定则过滤变体
    if args.variant:
        filtered_variants = [
            v for v in evaluator.model_config["variants"] if v["name"] == args.variant
        ]
        if not filtered_variants:
            logger.error(f"Variant not found / 未找到变体: {args.variant}")
            return
        evaluator.model_config["variants"] = filtered_variants
    
    # Run evaluation / 运行评估
    try:
        results = asyncio.run(evaluator.run_evaluation())
        logger.info("Evaluation completed successfully / 评估成功完成")
    except Exception as e:
        logger.error(f"Evaluation failed / 评估失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


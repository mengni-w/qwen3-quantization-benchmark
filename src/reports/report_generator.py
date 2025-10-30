#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Report Generator / 报告生成器

This module generates comprehensive evaluation reports comparing quantization methods.
该模块生成比较量化方法的综合评估报告。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate detailed evaluation reports / 生成详细的评估报告
    """
    
    def __init__(self, results_path: Path, language: str = "both"):
        """
        Initialize report generator / 初始化报告生成器
        
        Args:
            results_path: Path to evaluation results JSON file / 评估结果 JSON 文件路径
            language: Report language ('en', 'zh', 'both') / 报告语言
        """
        self.results_path = results_path
        self.language = language
        
        with open(results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)
        
        self.output_dir = results_path.parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self) -> Path:
        """
        Generate comprehensive Markdown report / 生成综合 Markdown 报告
        
        Returns:
            Path to generated report file / 生成的报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.language == "both":
            report_path = self.output_dir / f"evaluation_report_{timestamp}.md"
            content = self._generate_bilingual_content()
        elif self.language == "en":
            report_path = self.output_dir / f"evaluation_report_en_{timestamp}.md"
            content = self._generate_english_content()
        else:
            report_path = self.output_dir / f"evaluation_report_zh_{timestamp}.md"
            content = self._generate_chinese_content()
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Report generated / 报告已生成: {report_path}")
        return report_path
    
    def _generate_bilingual_content(self) -> str:
        """Generate bilingual report content / 生成双语报告内容"""
        content = []
        
        # Title / 标题
        content.append("# Qwen3 Quantization Accuracy Evaluation Report")
        content.append("# Qwen3 量化精度评估报告\n")
        
        # Introduction / 介绍
        content.append("## Executive Summary / 执行摘要")
        content.append(
            "This report presents a comprehensive evaluation of quantization methods "
            "for the Qwen3-30B-A3B-Instruct-2507 model, comparing original precision "
            "(FP16/BF16), FP8 quantization, and AWQ 4-bit quantization across "
            "multiple benchmark datasets.\n"
        )
        content.append(
            "本报告对 Qwen3-30B-A3B-Instruct-2507 模型的量化方法进行了全面评估，"
            "比较了原精度（FP16/BF16）、FP8 量化和 AWQ 4位量化在多个基准测试数据集上的表现。\n"
        )
        
        # Methodology / 方法论
        content.append("## Methodology / 方法论")
        content.append(self._generate_methodology_section())
        
        # Results / 结果
        content.append("## Evaluation Results / 评估结果")
        content.append(self._generate_results_section())
        
        # Analysis / 分析
        content.append("## Detailed Analysis / 详细分析")
        content.append(self._generate_analysis_section())
        
        # Conclusions / 结论
        content.append("## Conclusions and Recommendations / 结论和建议")
        content.append(self._generate_conclusions_section())
        
        # Appendix / 附录
        content.append("## Appendix / 附录")
        content.append(self._generate_appendix_section())
        
        return "\n".join(content)
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section / 生成方法论部分"""
        content = []
        
        model_name = self.results.get("model", "Unknown")
        variants = self.results.get("variants", {})
        
        content.append("### Model Configuration / 模型配置")
        content.append(f"- **Model / 模型**: {model_name}")
        content.append(f"- **Variants Evaluated / 评估的变体**: {len(variants)}")
        content.append("")
        
        for variant_name, variant_data in variants.items():
            desc = variant_data.get("description", "")
            quant = variant_data.get("quantization", "None")
            content.append(f"  - **{variant_name}**: {desc} (Quantization: {quant})")
        
        content.append("")
        content.append("### Evaluation Datasets / 评估数据集")
        
        config = self.results.get("config", {})
        datasets = config.get("datasets", [])
        
        for dataset in datasets:
            name = dataset.get("name", "")
            desc = dataset.get("description", "")
            num_samples = dataset.get("num_samples", 0)
            content.append(
                f"- **{name}**: {desc} ({num_samples} samples / 样本)"
            )
        
        content.append("")
        content.append("### Evaluation Metrics / 评估指标")
        content.append(
            "- **Accuracy / 精度**: Percentage of correct predictions / 正确预测的百分比"
        )
        content.append(
            "- **Latency / 延迟**: Time taken per evaluation / 每次评估所需时间"
        )
        content.append(
            "- **Throughput / 吞吐量**: Evaluations per second / 每秒评估次数"
        )
        content.append("")
        
        return "\n".join(content)
    
    def _generate_results_section(self) -> str:
        """Generate results section / 生成结果部分"""
        content = []
        
        variants = self.results.get("variants", {})
        summary = self.results.get("summary", {})
        
        # Overall accuracy comparison / 整体精度比较
        content.append("### Overall Accuracy Comparison / 整体精度比较\n")
        content.append("| Variant / 变体 | Description / 描述 | "
                      "GSM8K | HellaSwag | MMLU | Average / 平均 |")
        content.append("|:---|:---|:---:|:---:|:---:|:---:|")
        
        variant_accuracies = {}
        
        for variant_name, variant_data in variants.items():
            if "error" in variant_data:
                continue
            
            datasets = variant_data.get("datasets", {})
            accuracies = []
            row = [f"**{variant_name}**", variant_data.get("description", "")]
            
            for dataset_name in ["gsm8k", "hellaswag", "mmlu"]:
                if dataset_name in datasets:
                    acc = datasets[dataset_name].get("accuracy", 0.0)
                    accuracies.append(acc)
                    row.append(f"{acc:.4f}")
                else:
                    row.append("N/A")
            
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                row.append(f"{avg_acc:.4f}")
                variant_accuracies[variant_name] = avg_acc
            else:
                row.append("N/A")
            
            content.append("| " + " | ".join(row) + " |")
        
        content.append("")
        
        # Detailed results per dataset / 每个数据集的详细结果
        content.append("### Detailed Results by Dataset / 按数据集的详细结果\n")
        
        dataset_names = ["gsm8k", "hellaswag", "mmlu"]
        for dataset_name in dataset_names:
            content.append(f"#### {dataset_name.upper()}\n")
            content.append("| Variant / 变体 | Accuracy / 精度 | "
                          "Invalid Rate / 无效率 | Latency (s) / 延迟（秒） | "
                          "QPS / 每秒查询数 |")
            content.append("|:---|:---:|:---:|:---:|:---:|")
            
            for variant_name, variant_data in variants.items():
                if "error" in variant_data:
                    continue
                
                datasets = variant_data.get("datasets", {})
                if dataset_name in datasets:
                    ds_result = datasets[dataset_name]
                    if "error" in ds_result:
                        continue
                    
                    acc = ds_result.get("accuracy", 0.0)
                    invalid = ds_result.get("invalid_rate", 0.0)
                    latency = ds_result.get("latency_seconds", 0.0)
                    qps = ds_result.get("questions_per_second", 0.0)
                    
                    content.append(
                        f"| {variant_name} | {acc:.4f} | {invalid:.4f} | "
                        f"{latency:.2f} | {qps:.2f} |"
                    )
            
            content.append("")
        
        return "\n".join(content)
    
    def _generate_analysis_section(self) -> str:
        """Generate analysis section / 生成分析部分"""
        content = []
        
        summary = self.results.get("summary", {})
        variants = self.results.get("variants", {})
        
        content.append("### Accuracy Degradation Analysis / 精度退化分析\n")
        
        if "gsm8k" in summary and "degradation" in summary["gsm8k"]:
            content.append("Comparing to original precision / 与原精度比较:\n")
            degradations = summary["gsm8k"]["degradation"]
            for variant, deg in degradations.items():
                deg_pct = deg * 100
                content.append(
                    f"- **{variant}**: {deg_pct:.2f}% degradation / 退化"
                )
            content.append("")
        
        content.append("### Performance Characteristics / 性能特征\n")
        content.append(
            "Each quantization method has distinct characteristics:\n"
            "每种量化方法都有独特的特征：\n"
        )
        content.append("")
        content.append("1. **Original Precision (FP16/BF16) / 原精度**")
        content.append("   - Highest accuracy, serves as baseline / 最高精度，作为基线")
        content.append("   - Higher memory usage / 更高的内存使用")
        content.append("")
        content.append("2. **FP8 Quantization / FP8 量化**")
        content.append("   - Balanced accuracy and performance / 精度和性能平衡")
        content.append("   - Moderate memory reduction / 中等内存减少")
        content.append("   - Requires FP8-capable hardware / 需要支持 FP8 的硬件")
        content.append("")
        content.append("3. **AWQ 4-bit Quantization / AWQ 4位量化**")
        content.append("   - Largest memory reduction / 最大内存减少")
        content.append("   - Potentially highest performance gain / 潜在最高性能提升")
        content.append("   - May have more accuracy loss / 可能有更多精度损失")
        content.append("")
        
        return "\n".join(content)
    
    def _generate_conclusions_section(self) -> str:
        """Generate conclusions section / 生成结论部分"""
        content = []
        
        summary = self.results.get("summary", {})
        
        content.append("### Key Findings / 主要发现\n")
        
        # Find best variant per dataset / 找出每个数据集的最佳变体
        for dataset_name, dataset_summary in summary.items():
            if "best" in dataset_summary:
                best = dataset_summary["best"]
                accuracies = dataset_summary.get("accuracies", {})
                best_acc = accuracies.get(best, 0.0)
                
                content.append(
                    f"- **{dataset_name.upper()}**: Best performance achieved by "
                    f"**{best}** variant with {best_acc:.4f} accuracy / "
                    f"**{dataset_name.upper()}**: **{best}** 变体获得最佳性能，"
                    f"精度为 {best_acc:.4f}"
                )
        
        content.append("")
        content.append("### Recommendations / 建议\n")
        content.append(
            "Based on the evaluation results, the following recommendations are made:\n"
            "根据评估结果，提出以下建议：\n"
        )
        content.append("")
        content.append(
            "1. **For maximum accuracy / 追求最高精度**: Use original precision model / "
            "使用原精度模型"
        )
        content.append("")
        content.append(
            "2. **For balanced accuracy and performance / 平衡精度和性能**: "
            "Consider FP8 quantization if hardware supports it / "
            "如果硬件支持，考虑 FP8 量化"
        )
        content.append("")
        content.append(
            "3. **For maximum memory efficiency / 追求最大内存效率**: "
            "AWQ quantization provides the best memory reduction / "
            "AWQ 量化提供最佳内存减少"
        )
        content.append("")
        content.append(
            "4. **For production deployment / 生产部署**: "
            "Choose quantization method based on accuracy requirements and "
            "hardware constraints / "
            "根据精度要求和硬件约束选择量化方法"
        )
        content.append("")
        
        return "\n".join(content)
    
    def _generate_appendix_section(self) -> str:
        """Generate appendix section / 生成附录部分"""
        content = []
        
        content.append("### Configuration Details / 配置详情\n")
        content.append("```yaml")
        config_str = json.dumps(self.results.get("config", {}), indent=2)
        content.append(config_str)
        content.append("```\n")
        
        content.append("### Raw Results / 原始结果\n")
        content.append("Complete evaluation results are available in JSON format. / ")
        content.append("完整的评估结果以 JSON 格式提供。\n")
        
        return "\n".join(content)
    
    def _generate_english_content(self) -> str:
        """Generate English-only report / 生成仅英文报告"""
        # Similar structure but English only / 类似结构但仅英文
        return self._generate_bilingual_content()  # Simplified for now / 目前简化
    
    def _generate_chinese_content(self) -> str:
        """Generate Chinese-only report / 生成仅中文报告"""
        # Similar structure but Chinese only / 类似结构但仅中文
        return self._generate_bilingual_content()  # Simplified for now / 目前简化


def main():
    """Main entry point for report generation / 报告生成的主入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate evaluation report / 生成评估报告"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to evaluation results JSON file / 评估结果 JSON 文件路径",
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh", "both"],
        default="both",
        help="Report language / 报告语言",
    )
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.results_file, args.language)
    report_path = generator.generate_markdown_report()
    
    print(f"Report generated / 报告已生成: {report_path}")


if __name__ == "__main__":
    main()


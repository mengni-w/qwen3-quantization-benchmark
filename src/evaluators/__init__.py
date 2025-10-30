"""
Evaluation modules / 评估模块
"""

try:
    from src.evaluators.accuracy_evaluator import (
        GSM8KEvaluator,
        HellaSwagEvaluator,
        MMLUEvaluator,
        get_evaluator,
    )
except ImportError:
    # Fallback for direct module import / 直接模块导入的回退
    from .accuracy_evaluator import (
        GSM8KEvaluator,
        HellaSwagEvaluator,
        MMLUEvaluator,
        get_evaluator,
    )

__all__ = [
    "GSM8KEvaluator",
    "HellaSwagEvaluator",
    "MMLUEvaluator",
    "get_evaluator",
]


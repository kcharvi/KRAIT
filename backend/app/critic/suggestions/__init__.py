"""
Suggestion engine modules for kernel analysis
"""

from .suggestion_generator import SuggestionGenerator
from .code_patcher import CodePatcher
from .severity_classifier import SeverityClassifier
from .hardware_optimizer import HardwareOptimizer

__all__ = [
    "SuggestionGenerator",
    "CodePatcher", 
    "SeverityClassifier",
    "HardwareOptimizer"
]

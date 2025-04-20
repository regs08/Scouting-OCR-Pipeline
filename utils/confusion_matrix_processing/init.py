"""
Confusion Matrix Processing Package

This package provides tools for calculating, visualizing, analyzing, 
and storing confusion matrices for OCR and other text processing tasks.
"""

from .confusion_matrix_calculator import ConfusionMatrixCalculator
from .confusion_matrix_visualizer import ConfusionMatrixVisualizer
from .confusion_matrix_storage import ConfusionMatrixStorage
from .confusion_matrix_analyzer import ConfusionMatrixAnalyzer

__all__ = [
    'ConfusionMatrixCalculator',
    'ConfusionMatrixVisualizer',
    'ConfusionMatrixStorage',
    'ConfusionMatrixAnalyzer'
]
"""
Model module - classifier heads and complete models.
"""

from .pooling import MeanPooling, CLSPooling, MaxPooling
from .head import LinearClassifierHead, TransformerClassifierHead
from .classifier import TaxonomicClassifier

__all__ = [
    'MeanPooling',
    'CLSPooling',
    'MaxPooling',
    'LinearClassifierHead',
    'TransformerClassifierHead',
    'TaxonomicClassifier',
]


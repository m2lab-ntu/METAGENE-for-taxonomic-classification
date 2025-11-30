"""
MetaClassifier: Modular Taxonomic Classification Pipeline

A flexible, extensible pipeline for DNA sequence classification.
"""

__version__ = "2.0.0"

from . import tokenization
from . import embedding
from . import model

# Convenience imports
from .model import TaxonomicClassifier
from .tokenization import BPETokenizer, KmerTokenizer, Evo2Tokenizer
from .embedding import MetageneEncoder, DNABERTEncoder

__all__ = [
    'tokenization',
    'embedding',
    'model',
    'TaxonomicClassifier',
    'BPETokenizer',
    'KmerTokenizer',
    'Evo2Tokenizer',
    'MetageneEncoder',
    'DNABERTEncoder',
]


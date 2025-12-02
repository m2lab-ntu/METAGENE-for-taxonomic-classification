"""
Embedding module - foundation model encoders for DNA sequences.
"""

from .base import BaseEncoder
from .metagene_encoder import MetageneEncoder
from .evo2_encoder import Evo2Encoder
from .dnabert_encoder import DNABERTEncoder
from .generanno_encoder import GENERannoEncoder

__all__ = [
    'BaseEncoder',
    'MetageneEncoder',
    'Evo2Encoder',
    'DNABERTEncoder',
    'GENERannoEncoder'
]

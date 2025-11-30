"""
Base tokenizer interface for DNA/RNA sequences.
"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""
    
    def __init__(self, max_length: int = 512):
        """
        Initialize tokenizer.
        
        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = None  # Optional
        self.sep_token_id = None  # Optional
    
    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a DNA/RNA sequence into tokens.
        
        Args:
            sequence: DNA/RNA sequence string (e.g., "ATCG...")
        
        Returns:
            List of token strings
        """
        pass
    
    @abstractmethod
    def encode(self, sequence: str) -> List[int]:
        """
        Encode a DNA/RNA sequence into token IDs.
        
        Args:
            sequence: DNA/RNA sequence string
        
        Returns:
            List of token IDs
        """
        pass
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to sequence.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded sequence string
        """
        raise NotImplementedError("Decode not implemented for this tokenizer")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        raise NotImplementedError("get_vocab_size not implemented")
    
    def pad_and_truncate(self, token_ids: List[int]) -> List[int]:
        """
        Pad or truncate token IDs to max_length.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Padded/truncated list of token IDs
        """
        if len(token_ids) > self.max_length:
            return token_ids[:self.max_length]
        else:
            return token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
    
    def create_attention_mask(self, token_ids: List[int]) -> List[int]:
        """
        Create attention mask for token IDs.
        
        Args:
            token_ids: List of token IDs (after padding)
        
        Returns:
            Attention mask (1 for real tokens, 0 for padding)
        """
        return [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]


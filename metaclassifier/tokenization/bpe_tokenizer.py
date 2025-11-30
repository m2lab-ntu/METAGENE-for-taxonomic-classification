"""
BPE (Byte-Pair Encoding) tokenizer for DNA sequences.
Uses METAGENE-1 pretrained tokenizer or minbpe.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

from transformers import AutoTokenizer

from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """BPE tokenizer using HuggingFace or minbpe."""
    
    def __init__(
        self,
        tokenizer_path: str = "metagene-ai/METAGENE-1",
        max_length: int = 512,
        use_hf_tokenizer: bool = True
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer (HF model ID or local path)
            max_length: Maximum sequence length
            use_hf_tokenizer: Use HuggingFace AutoTokenizer if available
        """
        super().__init__(max_length)
        
        self.tokenizer_path = tokenizer_path
        self.use_hf_tokenizer = use_hf_tokenizer
        self.tokenizer = None
        
        # Try to load HuggingFace tokenizer
        if use_hf_tokenizer or tokenizer_path.startswith("metagene-ai"):
            try:
                print(f"Loading HuggingFace tokenizer from {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )
                self.pad_token_id = self.tokenizer.pad_token_id
                self.unk_token_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else 1
                print(f"✓ Loaded HuggingFace tokenizer (vocab size: {self.tokenizer.vocab_size})")
                return
            except Exception as e:
                print(f"Warning: Could not load HuggingFace tokenizer: {e}")
                if tokenizer_path.startswith("metagene-ai"):
                    raise  # If explicitly using metagene-ai, fail
        
        # Fallback to minbpe
        self._load_minbpe_tokenizer(tokenizer_path)
    
    def _load_minbpe_tokenizer(self, tokenizer_path: str):
        """Load minbpe tokenizer as fallback."""
        print(f"Loading minbpe tokenizer from {tokenizer_path}")
        
        # Import minbpe
        try:
            sys.path.append('/media/user/disk2/METAGENE/metagene-pretrain/train')
            from minbpe.minbpe import RegexTokenizer
        except ImportError:
            raise ImportError("minbpe not found. Install it or use HuggingFace tokenizer.")
        
        # Load tokenizer
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.tokenizer = RegexTokenizer()
        self.tokenizer.load(str(tokenizer_path))
        
        # Set special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        
        print(f"✓ Loaded minbpe tokenizer (vocab size: {self.tokenizer.vocab_size})")
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize DNA sequence into token strings.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            List of token strings
        """
        if isinstance(self.tokenizer, AutoTokenizer):
            # HuggingFace tokenizer
            tokens = self.tokenizer.tokenize(sequence)
            return tokens
        else:
            # minbpe tokenizer - convert IDs to tokens
            token_ids = self.tokenizer.encode(sequence)
            # minbpe doesn't have decode for individual tokens, return IDs as strings
            return [str(tid) for tid in token_ids]
    
    def encode(self, sequence: str) -> List[int]:
        """
        Encode DNA sequence into token IDs.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            List of token IDs
        """
        if isinstance(self.tokenizer, AutoTokenizer):
            # HuggingFace tokenizer
            return self.tokenizer(sequence, add_special_tokens=False)['input_ids']
        else:
            # minbpe tokenizer
            return self.tokenizer.encode(sequence)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to sequence.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded sequence
        """
        if isinstance(self.tokenizer, AutoTokenizer):
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # minbpe
            return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if isinstance(self.tokenizer, AutoTokenizer):
            return self.tokenizer.vocab_size
        else:
            return self.tokenizer.vocab_size


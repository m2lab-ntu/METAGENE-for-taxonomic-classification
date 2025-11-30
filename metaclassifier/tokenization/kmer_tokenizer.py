"""
K-mer tokenizer for DNA sequences.
Supports overlapping and non-overlapping k-mers.
"""

from typing import List, Dict

from .base import BaseTokenizer


class KmerTokenizer(BaseTokenizer):
    """K-mer tokenizer for DNA sequences."""
    
    def __init__(
        self,
        k: int = 6,
        max_length: int = 512,
        overlap: bool = True,
        stride: int = 1
    ):
        """
        Initialize k-mer tokenizer.
        
        Args:
            k: K-mer size (e.g., 3 for 3-mers, 6 for 6-mers)
            max_length: Maximum sequence length
            overlap: If True, use overlapping k-mers; if False, non-overlapping
            stride: Stride for overlapping k-mers (ignored if overlap=False)
        """
        super().__init__(max_length)
        
        self.k = k
        self.overlap = overlap
        self.stride = stride if overlap else k
        
        # Build vocabulary
        self._build_vocab()
        
        print(f"Initialized {k}-mer tokenizer")
        print(f"  Overlap: {overlap}, Stride: {self.stride}")
        print(f"  Vocab size: {self.get_vocab_size()}")
    
    def _build_vocab(self):
        """Build k-mer vocabulary."""
        # Generate all possible k-mers
        bases = ['A', 'C', 'G', 'T']
        
        # Special tokens
        self.token_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<N>': 2,  # For ambiguous bases
        }
        
        # Generate all k-mers
        kmers = ['']
        for _ in range(self.k):
            kmers = [kmer + base for kmer in kmers for base in bases]
        
        # Add to vocabulary
        for kmer in sorted(kmers):
            self.token_to_id[kmer] = len(self.token_to_id)
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Set special token IDs
        self.pad_token_id = self.token_to_id['<PAD>']
        self.unk_token_id = self.token_to_id['<UNK>']
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize DNA sequence into k-mers.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            List of k-mer tokens
        """
        sequence = sequence.upper()
        tokens = []
        
        # Extract k-mers with stride
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmer = sequence[i:i + self.k]
            
            # Handle ambiguous bases
            if 'N' in kmer or len(kmer) < self.k:
                tokens.append('<N>')
            elif kmer in self.token_to_id:
                tokens.append(kmer)
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def encode(self, sequence: str) -> List[int]:
        """
        Encode DNA sequence into k-mer token IDs.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(sequence)
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to sequence (best effort).
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded sequence
        """
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        
        if self.overlap:
            # For overlapping k-mers, reconstruct by taking first base of each k-mer
            sequence = tokens[0] if tokens else ''
            for token in tokens[1:]:
                if token not in ['<PAD>', '<UNK>', '<N>']:
                    sequence += token[-1]  # Add last base
        else:
            # For non-overlapping k-mers, concatenate
            sequence = ''.join([t for t in tokens if t not in ['<PAD>', '<UNK>', '<N>']])
        
        return sequence
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)


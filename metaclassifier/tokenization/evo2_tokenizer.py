"""
Evo2-style single-nucleotide tokenizer for DNA sequences.
Each nucleotide is treated as a separate token.
"""

from typing import List, Dict

from .base import BaseTokenizer


class Evo2Tokenizer(BaseTokenizer):
    """Evo2-style single-nucleotide tokenizer."""
    
    def __init__(self, max_length: int = 512):
        """
        Initialize Evo2 tokenizer.
        
        Args:
            max_length: Maximum sequence length
        """
        super().__init__(max_length)
        
        # Build vocabulary - single nucleotides
        self.token_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<CLS>': 2,
            '<SEP>': 3,
            'A': 4,
            'C': 5,
            'G': 6,
            'T': 7,
            'N': 8,  # Ambiguous base
        }
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Set special token IDs
        self.pad_token_id = self.token_to_id['<PAD>']
        self.unk_token_id = self.token_to_id['<UNK>']
        self.cls_token_id = self.token_to_id['<CLS>']
        self.sep_token_id = self.token_to_id['<SEP>']
        
        print(f"Initialized Evo2-style nucleotide tokenizer")
        print(f"  Vocab size: {self.get_vocab_size()}")
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize DNA sequence into individual nucleotides.
        
        Args:
            sequence: DNA sequence
        
        Returns:
            List of single-nucleotide tokens
        """
        sequence = sequence.upper()
        tokens = []
        
        for base in sequence:
            if base in ['A', 'C', 'G', 'T']:
                tokens.append(base)
            elif base == 'N':
                tokens.append('N')
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode DNA sequence into nucleotide token IDs.
        
        Args:
            sequence: DNA sequence
            add_special_tokens: Add <CLS> and <SEP> tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(sequence)
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to sequence.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded sequence
        """
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        
        if skip_special_tokens:
            special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']
            tokens = [t for t in tokens if t not in special_tokens]
        
        return ''.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)


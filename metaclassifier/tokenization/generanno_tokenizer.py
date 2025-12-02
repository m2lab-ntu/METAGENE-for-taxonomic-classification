"""
GENERanno tokenizer wrapper.
"""

from transformers import AutoTokenizer
from .base import BaseTokenizer

class GENERannoTokenizer(BaseTokenizer):
    """GENERanno tokenizer using HuggingFace AutoTokenizer."""
    
    def __init__(
        self,
        tokenizer_path: str = "GenerTeam/GENERanno-prokaryote-0.5b-base",
        max_length: int = 8192,
        use_hf_tokenizer: bool = True
    ):
        """
        Initialize GENERanno tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer (HF model ID or local path)
            max_length: Maximum sequence length (GENERanno supports up to 8k)
            use_hf_tokenizer: Ignored, always uses HF
        """
        super().__init__(max_length)
        
        self.tokenizer_path = tokenizer_path
        
        print(f"Loading GENERanno tokenizer from {tokenizer_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
            
            # Set special tokens if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            self.pad_token_id = self.tokenizer.pad_token_id
            self.unk_token_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else self.tokenizer.pad_token_id
            
            print(f"✓ Loaded GENERanno tokenizer (vocab size: {self.tokenizer.vocab_size})")
            
        except Exception as e:
            print(f"Error loading GENERanno tokenizer: {e}")
            raise

    def tokenize(self, sequence: str):
        return self.tokenizer.tokenize(sequence)
    
    def encode(self, sequence: str):
        return self.tokenizer(sequence, add_special_tokens=False)['input_ids']
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

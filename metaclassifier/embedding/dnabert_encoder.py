"""
DNABERT encoder wrapper.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .base import BaseEncoder


class DNABERTEncoder(BaseEncoder):
    """DNABERT/DNABERT-2 encoder wrapper."""
    
    def __init__(
        self,
        model_name_or_path: str = "zhihan1996/DNABERT-2-117M",
        freeze: bool = False,
        lora_config: Optional[Dict] = None
    ):
        """
        Initialize DNABERT encoder.
        
        Args:
            model_name_or_path: DNABERT model identifier
                Options:
                  - "zhihan1996/DNABERT-2-117M" (DNABERT-2)
                  - "zhihan1996/DNA_bert_6" (original DNABERT, 6-mer)
            freeze: Freeze encoder weights
            lora_config: LoRA configuration (not yet implemented)
        """
        print(f"Loading DNABERT encoder from {model_name_or_path}")
        
        try:
            encoder = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
            hidden_size = encoder.config.hidden_size
        except Exception as e:
            print(f"Error loading DNABERT: {e}")
            print("Make sure to install: pip install transformers")
            raise
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            hidden_size=hidden_size,
            freeze=freeze,
            lora_config=lora_config
        )
        
        self.encoder = encoder
        
        if freeze:
            self.freeze_encoder()
        
        if lora_config is not None:
            print("Warning: LoRA not yet implemented for DNABERT")
        
        print(f"✓ DNABERT encoder loaded (hidden_size={self.hidden_size})")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DNABERT encoder.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Hidden states (batch_size, seq_len, hidden_size)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.last_hidden_state
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.hidden_size


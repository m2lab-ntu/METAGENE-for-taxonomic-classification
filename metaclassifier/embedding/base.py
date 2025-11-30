"""
Base encoder interface for DNA/RNA foundation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all foundation model encoders."""
    
    def __init__(
        self,
        model_name_or_path: str,
        hidden_size: int,
        freeze: bool = False,
        lora_config: Optional[Dict] = None
    ):
        """
        Initialize encoder.
        
        Args:
            model_name_or_path: Model identifier or local path
            hidden_size: Hidden dimension size
            freeze: Freeze encoder weights (feature extraction mode)
            lora_config: LoRA configuration for efficient fine-tuning
        """
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.hidden_size = hidden_size
        self.freeze = freeze
        self.lora_config = lora_config
        self.encoder = None
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Hidden states (batch_size, seq_len, hidden_size)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension of the encoder."""
        pass
    
    def freeze_encoder(self):
        """Freeze encoder parameters for feature extraction."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"✓ Encoder frozen ({self.model_name_or_path})")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"✓ Encoder unfrozen ({self.model_name_or_path})")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


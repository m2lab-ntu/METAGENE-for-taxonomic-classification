"""
METAGENE-1 encoder wrapper.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel


from .base import BaseEncoder


class MetageneEncoder(BaseEncoder):
    """METAGENE-1 foundation model encoder."""
    
    def __init__(
        self,
        model_name_or_path: str = "metagene-ai/METAGENE-1",
        freeze: bool = False,
        lora_config: Optional[Dict] = None,
        gradient_checkpointing: bool = False
    ):
        """
        Initialize METAGENE-1 encoder.
        
        Args:
            model_name_or_path: METAGENE model path
            freeze: Freeze encoder weights
            lora_config: LoRA configuration
            gradient_checkpointing: Enable gradient checkpointing
        """
        # Load model to get hidden size
        print(f"Loading METAGENE-1 encoder from {model_name_or_path}")
        try:
            encoder = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Failed with device_map='auto', trying without: {e}")
            encoder = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        hidden_size = encoder.config.hidden_size
        
        # Initialize base class
        super().__init__(
            model_name_or_path=model_name_or_path,
            hidden_size=hidden_size,
            freeze=freeze,
            lora_config=lora_config
        )
        
        self.encoder = encoder
        self.gradient_checkpointing = gradient_checkpointing
        
        # Apply LoRA if configured
        if lora_config is not None and lora_config.get('enabled', False):
            self._setup_lora(lora_config)
        
        # Freeze if requested
        if freeze:
            self.freeze_encoder()
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        print(f"✓ METAGENE-1 encoder loaded (hidden_size={self.hidden_size})")
    
    def _setup_lora(self, lora_config: Dict):
        """Setup LoRA configuration."""
        print("Setting up LoRA for METAGENE-1...")
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
            bias=lora_config.get('bias', 'none')
        )
        
        self.encoder = get_peft_model(self.encoder, peft_config)
        print(f"✓ LoRA applied (rank={peft_config.r})")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        print("Enabling gradient checkpointing...")
        try:
            if hasattr(self.encoder, 'enable_input_require_grads'):
                self.encoder.enable_input_require_grads()
            
            if hasattr(self.encoder, 'base_model'):
                if hasattr(self.encoder.base_model, 'gradient_checkpointing_enable'):
                    self.encoder.base_model.gradient_checkpointing_enable()
            elif hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
            
            print("✓ Gradient checkpointing enabled")
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through METAGENE-1 encoder.
        
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


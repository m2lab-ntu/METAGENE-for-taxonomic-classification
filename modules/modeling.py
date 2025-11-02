"""
Model architecture for DNA/RNA sequence classification.
METAGENE-1 encoder + mean pooling + linear classifier with LoRA fine-tuning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    """Mean pooling layer with attention mask support."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling with attention mask.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Pooled embeddings: (batch_size, hidden_size)
        """
        # Expand attention mask to match hidden states dimensions
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum hidden states weighted by attention mask
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
        
        # Sum attention mask to get sequence lengths
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings


class MetaGeneClassifier(nn.Module):
    """
    METAGENE-1 encoder + mean pooling + linear classifier with LoRA fine-tuning.
    """
    
    def __init__(
        self,
        num_classes: int,
        encoder_path: str = "metagene-ai/METAGENE-1",
        pooling: str = "mean",
        dropout: float = 0.1,
        lora_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_path = encoder_path
        self.pooling = pooling
        self.gradient_checkpointing = gradient_checkpointing
        
        # Load encoder
        print(f"Loading encoder from {encoder_path}")
        try:
            self.encoder = AutoModel.from_pretrained(
                encoder_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"  # Auto device mapping as per HuggingFace docs
            )
        except Exception as e:
            print(f"Warning: Failed to load with device_map='auto', trying without: {e}")
            self.encoder = AutoModel.from_pretrained(
                encoder_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        
        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size
        
        # Setup pooling
        if pooling == "mean":
            self.pooler = MeanPooling()
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Apply LoRA if configured
        if lora_config is not None and lora_config.get('enabled', False):
            self._setup_lora(lora_config)
        
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _setup_lora(self, lora_config: Dict[str, Any]) -> None:
        """Setup LoRA configuration."""
        print("Setting up LoRA...")
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
            bias=lora_config.get('bias', 'none')
        )
        
        self.encoder = get_peft_model(self.encoder, peft_config)
        print(f"LoRA applied with rank {peft_config.r}")
    
    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        print("Enabling gradient checkpointing...")
        try:
            # For PEFT models
            if hasattr(self.encoder, 'enable_input_require_grads'):
                self.encoder.enable_input_require_grads()
            
            # Enable checkpointing on base model
            if hasattr(self.encoder, 'base_model'):
                if hasattr(self.encoder.base_model, 'gradient_checkpointing_enable'):
                    self.encoder.base_model.gradient_checkpointing_enable()
            elif hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
            
            print("✓ Gradient checkpointing enabled (saves ~50% activation memory)")
        except Exception as e:
            print(f"Warning: Could not enable gradient checkpointing: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) - optional for training
        
        Returns:
            Dictionary with logits and optionally loss
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state
        
        # Apply pooling
        pooled_output = self.pooler(last_hidden_states, attention_mask)
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss
        
        return outputs
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get pooled embeddings for inference."""
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            last_hidden_states = encoder_outputs.last_hidden_state
            pooled_output = self.pooler(last_hidden_states, attention_mask)
            
            return pooled_output
    
    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        """Save model and configuration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_state = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(model_state, output_dir / "pytorch_model.bin")
        
        # Save configuration
        config = {
            "num_classes": self.num_classes,
            "encoder_path": self.encoder_path,
            "pooling": self.pooling,
            "hidden_size": self.hidden_size,
            "model_type": "MetaGeneClassifier"
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "MetaGeneClassifier":
        """Load model from saved checkpoint."""
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            num_classes=config["num_classes"],
            encoder_path=config["encoder_path"],
            pooling=config["pooling"]
        )
        
        # Load weights
        if (model_path / "pytorch_model.bin").exists():
            state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        elif (model_path / "model.safetensors").exists():
            state_dict = load_file(model_path / "model.safetensors")
        else:
            raise FileNotFoundError("No model weights found")
        
        model.load_state_dict(state_dict)
        return model


def create_model(
    num_classes: int,
    config: Dict[str, Any],
    device: torch.device
) -> MetaGeneClassifier:
    """Create model from configuration."""
    
    # Extract model config
    model_config = config.get("model", {})
    lora_config = model_config.get("lora", {})
    
    # Check for gradient checkpointing
    gradient_checkpointing = model_config.get("gradient_checkpointing", False)
    
    # Create model
    model = MetaGeneClassifier(
        num_classes=num_classes,
        encoder_path=model_config.get("encoder_path", "metagene-ai/METAGENE-1"),
        pooling=model_config.get("pooling", "mean"),
        dropout=model_config.get("dropout", 0.1),
        lora_config=lora_config,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # Move to device
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    return model


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device
) -> Tuple[MetaGeneClassifier, Dict[str, Any]]:
    """Load model from training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model config
    with open(checkpoint_path.parent / "config.json", "r") as f:
        config = json.load(f)
    
    # Create model
    model = MetaGeneClassifier(
        num_classes=config["num_classes"],
        encoder_path=config["encoder_path"],
        pooling=config["pooling"],
        dropout=config.get("dropout", 0.1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    return model, config


def freeze_encoder(model: MetaGeneClassifier) -> None:
    """Freeze encoder parameters (only train classifier)."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    print("Encoder frozen, only classifier will be trained")


def unfreeze_encoder(model: MetaGeneClassifier) -> None:
    """Unfreeze encoder parameters."""
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    print("Encoder unfrozen for fine-tuning")


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

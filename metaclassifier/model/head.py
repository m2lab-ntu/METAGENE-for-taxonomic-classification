"""
Classifier head modules.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifierHead(nn.Module):
    """Simple linear classifier head."""
    
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        """
        Initialize linear classifier head.
        
        Args:
            hidden_size: Input hidden dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pooled_output: (batch_size, hidden_size)
        
        Returns:
            Logits: (batch_size, num_classes)
        """
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class TransformerClassifierHead(nn.Module):
    """
    Transformer-based classifier head (MetaTransformer-style).
    
    Applies additional transformer layers on top of pooled embeddings.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None
    ):
        """
        Initialize transformer classifier head.
        
        Args:
            hidden_size: Hidden dimension
            num_classes: Number of output classes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            feedforward_dim: Feedforward dimension (default: 4 * hidden_size)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        if feedforward_dim is None:
            feedforward_dim = 4 * hidden_size
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        print(f"Initialized TransformerClassifierHead:")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Hidden: {hidden_size}")
        print(f"  Classes: {num_classes}")
    
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pooled_output: (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)
        
        Returns:
            Logits: (batch_size, num_classes)
        """
        # If pooled output is 2D, add sequence dimension
        if pooled_output.dim() == 2:
            pooled_output = pooled_output.unsqueeze(1)  # (batch, 1, hidden)
        
        # Apply transformer
        transformed = self.transformer(pooled_output)  # (batch, seq_len, hidden)
        
        # Pool again (mean over sequence)
        pooled = transformed.mean(dim=1)  # (batch, hidden)
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits


class MultiHeadClassifierHead(nn.Module):
    """
    Multi-head classifier for hierarchical taxonomy.
    
    Predicts multiple taxonomic levels (e.g., genus + species).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_classes_dict: dict,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head classifier.
        
        Args:
            hidden_size: Hidden dimension
            num_classes_dict: Dictionary mapping level name to num_classes
                Example: {'genus': 100, 'species': 500}
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create a classifier head for each taxonomic level
        self.heads = nn.ModuleDict({
            level: nn.Linear(hidden_size, num_classes)
            for level, num_classes in num_classes_dict.items()
        })
        
        print(f"Initialized MultiHeadClassifierHead:")
        for level, num_classes in num_classes_dict.items():
            print(f"  {level}: {num_classes} classes")
    
    def forward(self, pooled_output: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            pooled_output: (batch_size, hidden_size)
        
        Returns:
            Dictionary of logits for each taxonomic level
        """
        pooled_output = self.dropout(pooled_output)
        
        logits_dict = {}
        for level, classifier in self.heads.items():
            logits_dict[level] = classifier(pooled_output)
        
        return logits_dict


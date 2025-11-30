"""
Main taxonomic classifier model.
Combines encoder + pooling + classifier head.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embedding.base import BaseEncoder
from .pooling import get_pooling_layer
from .head import LinearClassifierHead, TransformerClassifierHead


class TaxonomicClassifier(nn.Module):
    """
    Complete taxonomic classifier model.
    
    Architecture: Encoder → Pooling → Classifier Head
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        pooling_strategy: str = "mean",
        classifier_type: str = "linear",
        classifier_config: Optional[Dict] = None
    ):
        """
        Initialize taxonomic classifier.
        
        Args:
            encoder: Foundation model encoder
            num_classes: Number of output classes
            pooling_strategy: Pooling strategy ('mean', 'cls', 'max')
            classifier_type: Classifier type ('linear' or 'transformer')
            classifier_config: Additional config for classifier head
        """
        super().__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.classifier_type = classifier_type
        
        # Get embedding dimension from encoder
        self.hidden_size = encoder.get_embedding_dim()
        
        # Pooling layer
        self.pooler = get_pooling_layer(pooling_strategy)
        
        # Classifier head
        classifier_config = classifier_config or {}
        
        if classifier_type == "linear":
            self.classifier = LinearClassifierHead(
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                dropout=classifier_config.get('dropout', 0.1)
            )
        elif classifier_type == "transformer":
            self.classifier = TransformerClassifierHead(
                hidden_size=self.hidden_size,
                num_classes=num_classes,
                num_layers=classifier_config.get('num_layers', 2),
                num_heads=classifier_config.get('num_heads', 8),
                dropout=classifier_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        print(f"TaxonomicClassifier initialized:")
        print(f"  Encoder: {encoder.model_name_or_path}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Pooling: {pooling_strategy}")
        print(f"  Classifier: {classifier_type}")
        print(f"  Num classes: {num_classes}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels (batch_size,) - optional for training
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Encode
        hidden_states = self.encoder(input_ids, attention_mask)
        
        # Pool
        pooled_output = self.pooler(hidden_states, attention_mask)
        
        # Classify
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
        """
        Get pooled embeddings without classification.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        with torch.no_grad():
            hidden_states = self.encoder(input_ids, attention_mask)
            pooled_output = self.pooler(hidden_states, attention_mask)
            return pooled_output
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_probabilities: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Make predictions.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_probabilities: Return probabilities along with predictions
        
        Returns:
            Predictions (batch_size,) or (predictions, probabilities) tuple
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            
            predictions = torch.argmax(logits, dim=1)
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=1)
                return predictions, probabilities
            else:
                return predictions
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = self.encoder.get_num_parameters()
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'encoder_total': encoder_params['total'],
            'encoder_trainable': encoder_params['trainable'],
            'classifier_trainable': trainable - encoder_params['trainable']
        }
    
    def save_pretrained(self, output_dir: Union[str, Path]):
        """Save model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full state dict
        torch.save(self.state_dict(), output_dir / "model.pt")
        
        # Save config
        import json
        config = {
            "encoder_path": self.encoder.model_name_or_path,
            "num_classes": self.num_classes,
            "pooling_strategy": self.pooling_strategy,
            "classifier_type": self.classifier_type,
            "hidden_size": self.hidden_size
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved to {output_dir}")


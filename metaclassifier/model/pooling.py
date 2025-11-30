"""
Pooling strategies for sequence embeddings.
"""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Mean pooling with attention mask support."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Pooled output: (batch_size, hidden_size)
        """
        # Expand attention mask
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum with mask
        sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        
        # Compute mean
        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """CLS token pooling (first token)."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract CLS token (first token).
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) - not used
        
        Returns:
            CLS token: (batch_size, hidden_size)
        """
        return hidden_states[:, 0, :]


class MaxPooling(nn.Module):
    """Max pooling over sequence."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply max pooling.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Max pooled output: (batch_size, hidden_size)
        """
        # Mask out padding tokens
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states.clone()
        hidden_states[attention_mask == 0] = -1e9  # Large negative value
        
        # Max pool
        return torch.max(hidden_states, dim=1)[0]


def get_pooling_layer(pooling_strategy: str) -> nn.Module:
    """
    Get pooling layer by name.
    
    Args:
        pooling_strategy: One of 'mean', 'cls', 'max'
    
    Returns:
        Pooling layer
    """
    pooling_map = {
        'mean': MeanPooling,
        'cls': CLSPooling,
        'max': MaxPooling
    }
    
    if pooling_strategy not in pooling_map:
        raise ValueError(
            f"Unknown pooling strategy: {pooling_strategy}. "
            f"Choose from {list(pooling_map.keys())}"
        )
    
    return pooling_map[pooling_strategy]()


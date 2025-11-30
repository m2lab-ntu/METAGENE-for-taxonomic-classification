"""
Evo2 encoder wrapper.
Integrates with Arc Institute's Evo2 models.

Reference: https://github.com/ArcInstitute/evo2
Paper: https://www.biorxiv.org/content/10.1101/2025.02.18.638918
"""

from typing import Dict, Optional, List

import torch
import torch.nn as nn

from .base import BaseEncoder


class Evo2Encoder(BaseEncoder):
    """
    Evo2 foundation model encoder from Arc Institute.
    
    Evo2 is a state-of-the-art DNA language model using the StripedHyena 2 
    architecture, pretrained on 8.8T tokens from OpenGenome2.
    
    Features:
    - Single-nucleotide resolution
    - Up to 1M context length
    - Models: evo2_7b, evo2_40b, evo2_1b_base
    
    Reference: https://github.com/ArcInstitute/evo2
    """
    
    def __init__(
        self,
        model_name_or_path: str = "evo2_7b",
        freeze: bool = False,
        lora_config: Optional[Dict] = None,
        embedding_layer: str = "blocks.28.mlp.l3",
        use_cached_embeddings: bool = True
    ):
        """
        Initialize Evo2 encoder.
        
        Args:
            model_name_or_path: Evo2 model name
                Options: evo2_7b, evo2_40b, evo2_1b_base, evo2_7b_262k, etc.
            freeze: Freeze encoder weights (recommended for feature extraction)
            lora_config: LoRA configuration (not yet supported for Evo2)
            embedding_layer: Which layer to extract embeddings from
                Default: 'blocks.28.mlp.l3' (intermediate layer, works well)
                Options: See Evo2 model architecture
            use_cached_embeddings: Cache embeddings to avoid recomputation
        """
        print(f"Loading Evo2 encoder: {model_name_or_path}")
        print(f"  Repository: https://github.com/ArcInstitute/evo2")
        
        # Try to import Evo2
        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "Evo2 not installed. Install with:\n"
                "  git clone https://github.com/ArcInstitute/evo2.git\n"
                "  cd evo2\n"
                "  pip install -e .\n"
                "\nOr see: https://github.com/ArcInstitute/evo2#installation"
            )
        
        # Load Evo2 model
        print(f"Loading Evo2 model: {model_name_or_path}")
        self.evo2_model = Evo2(model_name_or_path)
        
        # Detect hidden size from model config
        # Evo2 7B has hidden_size = 4096, 40B has 5120, 1B has 2048
        if '40b' in model_name_or_path.lower():
            hidden_size = 5120
        elif '7b' in model_name_or_path.lower():
            hidden_size = 4096
        elif '1b' in model_name_or_path.lower():
            hidden_size = 2048
        else:
            # Default fallback
            hidden_size = 4096
            print(f"Warning: Unknown model size, assuming hidden_size={hidden_size}")
        
        # Initialize base class
        super().__init__(
            model_name_or_path=model_name_or_path,
            hidden_size=hidden_size,
            freeze=freeze,
            lora_config=lora_config
        )
        
        self.encoder = self.evo2_model
        self.embedding_layer = embedding_layer
        self.use_cached_embeddings = use_cached_embeddings
        self._embedding_cache = {}
        
        # Freeze if requested
        if freeze:
            self.freeze_encoder()
        
        # LoRA not yet supported for Evo2
        if lora_config is not None:
            print("Warning: LoRA not yet implemented for Evo2")
        
        print(f"✓ Evo2 encoder loaded:")
        print(f"    Model: {model_name_or_path}")
        print(f"    Hidden size: {hidden_size}")
        print(f"    Embedding layer: {embedding_layer}")
        print(f"    Context length: {'1M' if 'base' not in model_name_or_path else '8K'}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Evo2 encoder.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
                NOTE: Evo2 uses single-nucleotide tokens (A=4, C=5, G=6, T=7)
            attention_mask: Attention mask (batch_size, seq_len)
                NOTE: Evo2 may not use attention masks in the same way
        
        Returns:
            Hidden states (batch_size, seq_len, hidden_size)
        """
        # Check cache
        cache_key = None
        if self.use_cached_embeddings:
            cache_key = self._create_cache_key(input_ids)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        # Evo2 expects input on the correct device
        device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(device)
        
        # Get embeddings from specified layer
        _, embeddings = self.encoder(
            input_ids,
            return_embeddings=True,
            layer_names=[self.embedding_layer]
        )
        
        # Extract embeddings for the specified layer
        hidden_states = embeddings[self.embedding_layer]
        
        # Cache if enabled
        if self.use_cached_embeddings and cache_key is not None:
            self._embedding_cache[cache_key] = hidden_states.detach()
        
        return hidden_states
    
    def _create_cache_key(self, input_ids: torch.Tensor) -> str:
        """Create cache key for input sequence."""
        # Use hash of input_ids tensor
        return str(hash(input_ids.cpu().numpy().tobytes()))
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        print("✓ Embedding cache cleared")
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.hidden_size
    
    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 4
    ) -> List[str]:
        """
        Generate DNA sequences using Evo2.
        
        Args:
            prompt_seqs: List of DNA sequence prompts
            n_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            List of generated sequences
        """
        output = self.evo2_model.generate(
            prompt_seqs=prompt_seqs,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        return output.sequences
    
    def score_sequence(self, sequence: str) -> torch.Tensor:
        """
        Score a DNA sequence (compute log likelihoods).
        
        Args:
            sequence: DNA sequence string
        
        Returns:
            Logits (length, vocab_size)
        """
        # Tokenize
        input_ids = torch.tensor(
            self.evo2_model.tokenizer.tokenize(sequence),
            dtype=torch.int
        ).unsqueeze(0).to(next(self.encoder.parameters()).device)
        
        # Forward pass
        outputs, _ = self.evo2_model(input_ids)
        logits = outputs[0].squeeze(0)
        
        return logits


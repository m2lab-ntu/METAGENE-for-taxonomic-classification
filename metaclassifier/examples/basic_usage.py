"""
Basic usage example for MetaClassifier.

Demonstrates:
- Creating tokenizer, encoder, and model
- Making predictions on DNA sequences
- Extracting embeddings
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tokenization import BPETokenizer
from embedding import MetageneEncoder
from model import TaxonomicClassifier


def main():
    print("=" * 80)
    print("MetaClassifier - Basic Usage Example")
    print("=" * 80)
    
    # 1. Create tokenizer
    print("\n1. Creating BPE tokenizer...")
    tokenizer = BPETokenizer(
        tokenizer_path="metagene-ai/METAGENE-1",
        max_length=192,
        use_hf_tokenizer=True
    )
    print(f"✓ Tokenizer created (vocab size: {tokenizer.get_vocab_size()})")
    
    # 2. Create encoder
    print("\n2. Creating METAGENE-1 encoder...")
    encoder = MetageneEncoder(
        model_name_or_path="metagene-ai/METAGENE-1",
        freeze=False,
        lora_config={
            'enabled': True,
            'r': 4,
            'alpha': 8
        }
    )
    print(f"✓ Encoder created (hidden size: {encoder.get_embedding_dim()})")
    
    # 3. Create classifier
    print("\n3. Creating taxonomic classifier...")
    num_classes = 10  # Example: 10 species
    model = TaxonomicClassifier(
        encoder=encoder,
        num_classes=num_classes,
        pooling_strategy="mean",
        classifier_type="linear"
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model created on {device}")
    
    # Print model info
    params = model.get_num_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Ratio: {params['trainable']/params['total']:.2%}")
    
    # 4. Tokenize a sequence
    print("\n4. Tokenizing DNA sequence...")
    sequence = "ATCGATCGATCGATCGATCGATCG" * 5  # 120 bp
    tokens = tokenizer.encode(sequence)
    tokens = tokenizer.pad_and_truncate(tokens)
    attention_mask = tokenizer.create_attention_mask(tokens)
    
    print(f"✓ Sequence length: {len(sequence)} bp")
    print(f"✓ Token count: {len([t for t in tokens if t != tokenizer.pad_token_id])}")
    
    # 5. Make prediction
    print("\n5. Making prediction...")
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions, probabilities = model.predict(
            input_ids,
            attention_mask_tensor,
            return_probabilities=True
        )
    
    predicted_class = predictions[0].item()
    confidence = probabilities[0, predicted_class].item()
    
    print(f"✓ Predicted class: {predicted_class}")
    print(f"✓ Confidence: {confidence:.4f}")
    print(f"✓ Top 3 probabilities:")
    top3 = torch.topk(probabilities[0], 3)
    for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
        print(f"    {i+1}. Class {idx.item()}: {prob.item():.4f}")
    
    # 6. Extract embeddings
    print("\n6. Extracting embeddings...")
    embeddings = model.get_embeddings(input_ids, attention_mask_tensor)
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding norm: {torch.norm(embeddings).item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Basic usage example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


"""
Modular prediction script for taxonomic classification.

Usage:
    # Per-read prediction
    python predict.py --config configs/predict_config.yaml \
        --input reads.fasta --output predictions.csv
    
    # With aggregation to sample level
    python predict.py --config configs/predict_config.yaml \
        --input reads.fasta --output predictions.csv \
        --aggregate --abundance_output abundance.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml
from tqdm import tqdm

import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modular components
from metaclassifier.tokenization import BPETokenizer, KmerTokenizer, Evo2Tokenizer
from metaclassifier.embedding import MetageneEncoder, Evo2Encoder, DNABERTEncoder
from metaclassifier.model import TaxonomicClassifier
import metaclassifier.aggregate as agg_module


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_tokenizer(tokenizer_config: Dict):
    """Create tokenizer from configuration."""
    tokenizer_type = tokenizer_config['type']
    
    if tokenizer_type == 'bpe':
        return BPETokenizer(
            tokenizer_path=tokenizer_config['path'],
            max_length=tokenizer_config['max_length'],
            use_hf_tokenizer=tokenizer_config.get('use_hf', True)
        )
    elif tokenizer_type == 'kmer':
        return KmerTokenizer(
            k=tokenizer_config['k'],
            max_length=tokenizer_config['max_length'],
            overlap=tokenizer_config.get('overlap', True)
        )
    elif tokenizer_type == 'evo2':
        return Evo2Tokenizer(
            max_length=tokenizer_config['max_length']
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def create_encoder(encoder_config: Dict):
    """Create encoder from configuration."""
    encoder_type = encoder_config['type']
    
    if encoder_type == 'metagene':
        return MetageneEncoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False),
            lora_config=encoder_config.get('lora', None)
        )
    elif encoder_type == 'evo2':
        return Evo2Encoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False)
        )
    elif encoder_type == 'dnabert':
        return DNABERTEncoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: torch.device
):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load label mappings
    label_dir = checkpoint_path.parent
    with open(label_dir / "label2id.json", 'r') as f:
        label2id = json.load(f)
    
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    
    # Create tokenizer and encoder
    tokenizer = create_tokenizer(config['tokenizer'])
    encoder = create_encoder(config['encoder'])
    
    # Create model
    model = TaxonomicClassifier(
        encoder=encoder,
        num_classes=num_classes,
        pooling_strategy=config['model'].get('pooling', 'mean'),
        classifier_type=config['model'].get('classifier_type', 'linear'),
        classifier_config=config['model'].get('classifier_config', {})
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load id2taxid if available
    id2taxid = {}
    if (label_dir / "id2taxid.json").exists():
        with open(label_dir / "id2taxid.json", 'r') as f:
            id2taxid = {int(k): v for k, v in json.load(f).items()}
    
    return model, tokenizer, id2label, id2taxid


def predict_fasta(
    model: TaxonomicClassifier,
    tokenizer,
    fasta_path: str,
    id2label: Dict,
    device: torch.device,
    batch_size: int = 256
) -> pd.DataFrame:
    """
    Make predictions on FASTA file.
    
    Args:
        model: Trained classifier
        tokenizer: Tokenizer
        fasta_path: Path to FASTA file
        id2label: ID to label mapping
        device: Device to use
        batch_size: Batch size
    
    Returns:
        DataFrame with predictions
    """
    from Bio import SeqIO
    
    print(f"Loading sequences from {fasta_path}")
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(fasta_path, 'fasta'):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Predict in batches
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
            batch_seqs = sequences[i:i+batch_size]
            
            # Tokenize
            batch_tokens = []
            batch_masks = []
            
            for seq in batch_seqs:
                tokens = tokenizer.encode(seq)
                tokens = tokenizer.pad_and_truncate(tokens)
                mask = tokenizer.create_attention_mask(tokens)
                
                batch_tokens.append(tokens)
                batch_masks.append(mask)
            
            # Convert to tensors
            input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            attention_mask = torch.tensor(batch_masks, dtype=torch.long).to(device)
            
            # Predict
            predictions, probabilities = model.predict(
                input_ids, attention_mask, return_probabilities=True
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Create results DataFrame
    results = pd.DataFrame({
        'sequence_id': seq_ids,
        'predicted_class': [id2label[p] for p in all_predictions],
        'confidence': [prob.max() for prob in all_probabilities]
    })
    
    # Add per-class probabilities (optional, can be large)
    # for i, label in id2label.items():
    #     results[f'prob_{label}'] = [prob[i] for prob in all_probabilities]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Modular taxonomic classification")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output predictions CSV")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: from config or 256)")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate to sample level")
    parser.add_argument("--abundance_output", help="Output abundance table")
    parser.add_argument("--sample_id_pattern", help="Regex to extract sample ID from sequence ID")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Min confidence")
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, tokenizer, id2label, id2taxid = load_model_from_checkpoint(
        args.checkpoint, config, device
    )
    
    # Determine batch size
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = config.get('prediction', {}).get('batch_size', 256)
    
    # Make predictions
    print(f"Making predictions (batch_size={batch_size})...")
    predictions_df = predict_fasta(
        model, tokenizer, args.input, id2label, device, batch_size
    )
    
    # Add tax_id if available
    if id2taxid:
        # Map predicted_class (name) back to ID then to tax_id?
        # Or we can map directly if we have the predicted index.
        # predict_fasta currently returns predicted_class as string.
        # Let's modify predict_fasta to return index as well or map it here.
        
        # Invert id2label to get name -> id
        label2id = {v: k for k, v in id2label.items()}
        
        predictions_df['tax_id'] = predictions_df['predicted_class'].map(
            lambda x: id2taxid.get(label2id.get(x))
        )
    
    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")
    
    # Print summary
    print(f"\nPrediction summary:")
    print(f"  Total reads: {len(predictions_df)}")
    print(f"  Mean confidence: {predictions_df['confidence'].mean():.3f}")
    print(f"  Unique species: {predictions_df['predicted_class'].nunique()}")
    
    # Aggregate if requested
    if args.aggregate:
        if not args.abundance_output:
            args.abundance_output = output_path.parent / "abundance.csv"
        
        print("\nAggregating to sample level...")
        
        # Extract sample IDs
        if args.sample_id_pattern:
            import re
            pattern = re.compile(args.sample_id_pattern)
            predictions_df['sample_id'] = predictions_df['sequence_id'].apply(
                lambda x: pattern.search(x).group(1) if pattern.search(x) else 'unknown'
            )
        else:
            # Default: use first part of sequence ID before underscore/dash
            predictions_df['sample_id'] = predictions_df['sequence_id'].str.split('[_-]').str[0]
        
        # Aggregate
        abundance_df = agg_module.aggregate_predictions_to_sample(
            predictions_df,
            confidence_threshold=args.confidence_threshold,
            tax_id_column='tax_id' if 'tax_id' in predictions_df.columns else None
        )
        
        # Save
        agg_module.save_abundance_table(abundance_df, args.abundance_output)
        
        print(f"\nAbundance summary:")
        print(f"  Samples: {abundance_df['sample_id'].nunique()}")
        print(f"  Unique species: {abundance_df['Species_Name'].nunique()}")


if __name__ == "__main__":
    main()


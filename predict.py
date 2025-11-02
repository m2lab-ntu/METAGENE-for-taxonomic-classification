#!/usr/bin/env python3
"""
Prediction script for METAGENE DNA/RNA sequence classification.
Supports both per-read and per-sample aggregation modes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.utils import setup_logging, get_device
from modules.dataloading import (
    SequenceDataset, MetaGeneTokenizer, load_mapping_tsv, create_dataloader
)
from modules.modeling import create_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict with METAGENE classification model")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input FASTA/FASTQ file or CSV file for per-sample mode"
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--per_sample",
        action="store_true",
        help="Enable per-sample aggregation mode"
    )
    
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum"],
        help="Aggregation method for per-sample mode"
    )
    
    parser.add_argument(
        "--sample_column",
        type=str,
        default="sample_id",
        help="Column name for sample ID in CSV input"
    )
    
    parser.add_argument(
        "--sequence_column",
        type=str,
        default="sequence",
        help="Column name for sequence in CSV input"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint."""
    ckpt_path = Path(ckpt_path)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load config
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        config_path = ckpt_path.parent.parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found for checkpoint: {ckpt_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load label mappings
    label2id_path = ckpt_path.parent / "label2id.json"
    if not label2id_path.exists():
        label2id_path = ckpt_path.parent.parent / "label2id.json"
    
    if not label2id_path.exists():
        raise FileNotFoundError(f"Label mapping not found for checkpoint: {ckpt_path}")
    
    with open(label2id_path, 'r') as f:
        label2id = json.load(f)
    
    id2label = {int(k): v for k, v in label2id.items()}
    num_classes = len(label2id)
    
    # Create model
    model = create_model(num_classes, config, device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, config, label2id, id2label


def predict_per_read(
    model: nn.Module,
    input_file: str,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    class_names: List[str]
) -> pd.DataFrame:
    """Predict per-read classifications."""
    print("Setting up per-read prediction...")
    
    # Create a dummy mapping for inference (we don't need labels)
    dummy_mapping = pd.DataFrame({
        'class_id': [0],  # Dummy class
        'label_name': ['dummy']
    })
    
    # Initialize tokenizer
    tokenizer = MetaGeneTokenizer(
        tokenizer_path=config['tokenizer']['name_or_path'],
        max_length=config['tokenizer']['max_length'],
        use_hf_tokenizer=config['tokenizer'].get('use_hf_tokenizer', False)
    )
    
    # Create dataset
    dataset = SequenceDataset(
        fasta_path=input_file,
        mapping_df=dummy_mapping,
        tokenizer=tokenizer,
        header_regex=config['dataset']['header_regex'],
        max_length=config['tokenizer']['max_length'],
        class_column='class_id',
        label_column='label_name',
        strict_classes=False  # Don't fail on unknown classes
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Run inference
    print("Running inference...")
    all_predictions = []
    all_probabilities = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs['logits']
            
            # Get predictions and probabilities
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_metadata.append(batch['metadata'])
    
    # Flatten metadata
    all_sequence_ids = []
    all_lengths = []
    
    for metadata in all_metadata:
        all_sequence_ids.extend(metadata['sequence_ids'])
        all_lengths.extend(metadata['lengths'])
    
    # Create results DataFrame
    results = {
        'sequence_id': all_sequence_ids,
        'predicted_class': [class_names[p] for p in all_predictions],
        'confidence': [prob.max() for prob in all_probabilities],
        'sequence_length': all_lengths
    }
    
    # Add per-class probabilities
    for i, class_name in enumerate(class_names):
        results[f'prob_{class_name}'] = [prob[i] for prob in all_probabilities]
    
    return pd.DataFrame(results)


def predict_per_sample(
    model: nn.Module,
    input_file: str,
    config: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    class_names: List[str],
    aggregation: str,
    sample_column: str,
    sequence_column: str
) -> pd.DataFrame:
    """Predict per-sample classifications with aggregation."""
    print("Setting up per-sample prediction...")
    
    # Load CSV file
    df = pd.read_csv(input_file)
    
    if sample_column not in df.columns:
        raise ValueError(f"Sample column '{sample_column}' not found in CSV")
    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not found in CSV")
    
    print(f"Loaded {len(df)} sequences from {len(df[sample_column].unique())} samples")
    
    # Initialize tokenizer
    tokenizer = MetaGeneTokenizer(
        tokenizer_path=config['tokenizer']['name_or_path'],
        max_length=config['tokenizer']['max_length'],
        use_hf_tokenizer=config['tokenizer'].get('use_hf_tokenizer', False)
    )
    
    # Process each sample
    sample_results = []
    
    for sample_id in tqdm(df[sample_column].unique(), desc="Processing samples"):
        # Get sequences for this sample
        sample_sequences = df[df[sample_column] == sample_id][sequence_column].tolist()
        
        # Tokenize sequences
        tokenized_sequences = []
        attention_masks = []
        
        for seq in sample_sequences:
            tokens = tokenizer.encode(seq)
            
            # Truncate or pad
            if len(tokens) > config['tokenizer']['max_length']:
                tokens = tokens[:config['tokenizer']['max_length']]
            else:
                tokens = tokens + [tokenizer.pad_token_id] * (config['tokenizer']['max_length'] - len(tokens))
            
            attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in tokens]
            
            tokenized_sequences.append(tokens)
            attention_masks.append(attention_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(tokenized_sequences, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
        
        # Get embeddings for each sequence
        with torch.no_grad():
            embeddings = model.get_embeddings(input_ids, attention_mask)
        
        # Aggregate embeddings
        if aggregation == "mean":
            aggregated_embedding = embeddings.mean(dim=0, keepdim=True)
        elif aggregation == "max":
            aggregated_embedding = embeddings.max(dim=0, keepdim=True)[0]
        elif aggregation == "sum":
            aggregated_embedding = embeddings.sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Classify aggregated embedding
        with torch.no_grad():
            # Apply dropout and classifier
            pooled_output = model.dropout(aggregated_embedding)
            logits = model.classifier(pooled_output)
            
            # Get prediction
            prediction = torch.argmax(logits, dim=1).cpu().item()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Store result
        sample_results.append({
            'sample_id': sample_id,
            'num_sequences': len(sample_sequences),
            'predicted_class': class_names[prediction],
            'confidence': probabilities.max(),
            **{f'prob_{class_name}': prob for class_name, prob in zip(class_names, probabilities)}
        })
    
    return pd.DataFrame(sample_results)


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting METAGENE classification prediction")
    
    # Setup device
    device = get_device()
    
    # Load model
    logger.info(f"Loading model from {args.ckpt}")
    model, config, label2id, id2label = load_model_from_checkpoint(args.ckpt, device)
    
    # Get class names
    class_names = [id2label[i] for i in range(len(id2label))]
    print(f"Model has {len(class_names)} classes: {class_names}")
    
    # Run prediction
    if args.per_sample:
        # Per-sample mode
        results = predict_per_sample(
            model, args.input, config, device, args.batch_size,
            class_names, args.aggregation, args.sample_column, args.sequence_column
        )
    else:
        # Per-read mode
        results = predict_per_read(
            model, args.input, config, device, args.batch_size, class_names
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(results)}")
    
    # Print summary
    if args.per_sample:
        print(f"Unique samples: {results['sample_id'].nunique()}")
        print(f"Average sequences per sample: {results['num_sequences'].mean():.1f}")
    
    print(f"Prediction confidence: {results['confidence'].mean():.3f} ± {results['confidence'].std():.3f}")
    
    # Show class distribution
    class_counts = results['predicted_class'].value_counts()
    print("\nPredicted class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} ({count/len(results)*100:.1f}%)")
    
    logger.info(f"Prediction completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()

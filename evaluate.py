#!/usr/bin/env python3
"""
Evaluation script for METAGENE DNA/RNA sequence classification.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.utils import setup_logging, get_device, load_checkpoint
from modules.dataloading import (
    SequenceDataset, MetaGeneTokenizer, load_mapping_tsv, create_dataloader
)
from modules.modeling import create_model
from modules.metrics import (
    compute_metrics, plot_confusion_matrix, generate_classification_report,
    save_metrics_summary, print_metrics_summary, compute_per_class_metrics
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate METAGENE classification model")
    
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
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
        # Try parent directory
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


def setup_eval_data(config: Dict[str, Any], split: str) -> tuple:
    """Setup evaluation dataset."""
    print(f"Setting up {split} dataset...")
    
    # Determine data path
    if split == "train":
        data_path = config['dataset']['train_fasta']
    elif split == "val":
        data_path = config['dataset']['val_fasta']
    elif split == "test":
        data_path = config['dataset']['test_fasta']
        if data_path is None:
            raise ValueError("Test dataset not specified in config")
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Load mapping
    mapping_df = load_mapping_tsv(config['dataset']['mapping_tsv'])
    
    # Initialize tokenizer
    tokenizer = MetaGeneTokenizer(
        tokenizer_path=config['tokenizer']['name_or_path'],
        max_length=config['tokenizer']['max_length'],
        use_hf_tokenizer=config['tokenizer'].get('use_hf_tokenizer', False)
    )
    
    # Create dataset
    dataset = SequenceDataset(
        fasta_path=data_path,
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=config['dataset']['header_regex'],
        max_length=config['tokenizer']['max_length'],
        class_column=config['dataset']['class_column'],
        label_column=config['dataset']['label_column'],
        strict_classes=config['dataset']['strict_classes']
    )
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=config.get('eval_batch_size', 128),
        shuffle=False,
        num_workers=4
    )
    
    return dataset, dataloader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    class_names: List[str]
) -> Dict[str, Any]:
    """Evaluate model on dataset."""
    model.eval()
    all_logits = []
    all_labels = []
    all_metadata = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Store results
            all_logits.append(outputs['logits'].cpu())
            all_labels.append(batch['labels'].cpu())
            all_metadata.append(batch['metadata'])
    
    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Flatten metadata
    all_sequence_ids = []
    all_class_ids = []
    all_label_names = []
    all_lengths = []
    
    for metadata in all_metadata:
        all_sequence_ids.extend(metadata['sequence_ids'])
        all_class_ids.extend(metadata['class_ids'])
        all_label_names.extend(metadata['labels'])
        all_lengths.extend(metadata['lengths'])
    
    # Compute metrics
    metrics = compute_metrics(
        all_logits, all_labels,
        num_classes=all_logits.shape[1],
        class_names=class_names,
        compute_auroc=config['metrics']['compute_auroc']
    )
    
    # Add metadata
    metrics['sequence_ids'] = all_sequence_ids
    metrics['class_ids'] = all_class_ids
    metrics['label_names'] = all_label_names
    metrics['lengths'] = all_lengths
    
    return metrics, all_logits, all_labels


def save_results(
    metrics: Dict[str, Any],
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    class_names: List[str],
    output_dir: Path,
    split: str
) -> None:
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    save_metrics_summary(metrics, output_dir / f"{split}_metrics.json")
    
    # Generate classification report
    if metrics.get('confusion_matrix') is not None:
        report_path = output_dir / f"{split}_classification_report.json"
        generate_classification_report(
            all_logits, all_labels, class_names, report_path
        )
    
    # Plot confusion matrix
    if metrics.get('confusion_matrix') is not None:
        cm_path = output_dir / f"{split}_confusion_matrix.png"
        plot_confusion_matrix(
            torch.tensor(metrics['confusion_matrix']).numpy(),
            class_names,
            cm_path,
            title=f"{split.title()} Confusion Matrix"
        )
    
    # Save per-class metrics
    per_class_df = compute_per_class_metrics(all_logits, all_labels, class_names)
    per_class_df.to_csv(output_dir / f"{split}_per_class_metrics.csv", index=False)
    
    # Save predictions
    predictions = torch.argmax(all_logits, dim=1).numpy()
    probabilities = torch.softmax(all_logits, dim=1).numpy()
    
    results_df = {
        'sequence_id': metrics['sequence_ids'],
        'true_label': metrics['label_names'],
        'predicted_class': [class_names[p] for p in predictions],
        'confidence': probabilities.max(axis=1),
        'sequence_length': metrics['lengths']
    }
    
    # Add per-class probabilities
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probabilities[:, i]
    
    import pandas as pd
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_dir / f"{split}_predictions.csv", index=False)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting METAGENE classification evaluation")
    
    # Setup device
    device = get_device()
    
    # Load model
    logger.info(f"Loading model from {args.ckpt}")
    model, config, label2id, id2label = load_model_from_checkpoint(args.ckpt, device)
    
    # Setup data
    dataset, dataloader = setup_eval_data(config, args.split)
    
    # Get class names
    class_names = [id2label[i] for i in range(len(id2label))]
    
    # Evaluate model
    metrics, all_logits, all_labels = evaluate_model(
        model, dataloader, device, config, class_names
    )
    
    # Print results
    print_metrics_summary(metrics)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.ckpt).parent / f"eval_{args.split}"
    
    save_results(
        metrics, all_logits, all_labels, class_names, output_dir, args.split
    )
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

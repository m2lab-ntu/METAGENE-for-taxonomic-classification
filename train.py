#!/usr/bin/env python3
"""
Training script for METAGENE DNA/RNA sequence classification.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.utils import (
    set_seed, setup_logging, get_device, save_checkpoint, 
    EarlyStopping, log_gpu_memory, get_learning_rate, count_parameters
)
from modules.dataloading import (
    SequenceDataset, MetaGeneTokenizer, load_mapping_tsv, 
    create_dataloader, get_sequence_statistics, save_class_distribution
)
from modules.modeling import create_model
from modules.metrics import (
    compute_metrics, plot_confusion_matrix, plot_training_curves,
    save_metrics_summary, print_metrics_summary, generate_classification_report
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train METAGENE classification model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to config file"
    )
    
    # Allow CLI overrides for key parameters
    parser.add_argument("--train_fasta", type=str, help="Training FASTA file path")
    parser.add_argument("--val_fasta", type=str, help="Validation FASTA file path")
    parser.add_argument("--mapping_tsv", type=str, help="Mapping TSV file path")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--max_epochs", type=int, help="Maximum epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration from YAML file and apply CLI overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.train_fasta:
        config['dataset']['train_fasta'] = args.train_fasta
    if args.val_fasta:
        config['dataset']['val_fasta'] = args.val_fasta
    if args.mapping_tsv:
        config['dataset']['mapping_tsv'] = args.mapping_tsv
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    if args.lr:
        config['optimizer']['lr'] = args.lr
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_paths = [
        config['dataset']['train_fasta'],
        config['dataset']['val_fasta'],
        config['dataset']['mapping_tsv']
    ]
    
    for path in required_paths:
        if path is None:
            raise ValueError(f"Required dataset path not specified: {path}")
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")


def setup_data(config: Dict[str, Any]) -> tuple:
    """Setup datasets and dataloaders."""
    print("Setting up data...")
    
    # Load mapping
    mapping_df = load_mapping_tsv(config['dataset']['mapping_tsv'])
    print(f"Loaded mapping with {len(mapping_df)} entries")
    
    # Initialize tokenizer
    tokenizer = MetaGeneTokenizer(
        tokenizer_path=config['tokenizer']['name_or_path'],
        max_length=config['tokenizer']['max_length'],
        use_hf_tokenizer=config['tokenizer'].get('use_hf_tokenizer', False)
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create datasets
    train_dataset = SequenceDataset(
        fasta_path=config['dataset']['train_fasta'],
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=config['dataset']['header_regex'],
        max_length=config['tokenizer']['max_length'],
        class_column=config['dataset']['class_column'],
        label_column=config['dataset']['label_column'],
        strict_classes=config['dataset']['strict_classes']
    )
    
    val_dataset = SequenceDataset(
        fasta_path=config['dataset']['val_fasta'],
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=config['dataset']['header_regex'],
        max_length=config['tokenizer']['max_length'],
        class_column=config['dataset']['class_column'],
        label_column=config['dataset']['label_column'],
        strict_classes=config['dataset']['strict_classes']
    )
    
    # Get statistics
    train_stats = get_sequence_statistics(train_dataset)
    val_stats = get_sequence_statistics(val_dataset)
    
    print(f"Train dataset: {train_stats['num_sequences']} sequences, {train_stats['num_classes']} classes")
    print(f"Val dataset: {val_stats['num_sequences']} sequences, {val_stats['num_classes']} classes")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return (
        train_dataset, val_dataset, train_loader, val_loader,
        tokenizer, mapping_df, train_stats, val_stats
    )


def setup_model(config: Dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    """Setup model and optimizer."""
    print("Setting up model...")
    
    # Create model
    model = create_model(num_classes, config, device)
    
    # Setup mixed precision
    scaler = GradScaler() if config['training']['precision'] != '32' else None
    
    return model, scaler


def setup_optimizer(model: nn.Module, config: Dict[str, Any], num_training_steps: int) -> tuple:
    """Setup optimizer and scheduler."""
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=config['optimizer']['betas']
    )
    
    # Scheduler
    if config['scheduler']['name'] == 'linear':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['scheduler']['warmup_steps'],
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']['name']}")
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: Dict[str, Any],
    logger: Any
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Filter batch to only include model inputs
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(**model_inputs)
                loss = outputs['loss'] / config['training']['grad_accum_steps']
        else:
            outputs = model(**model_inputs)
            loss = outputs['loss'] / config['training']['grad_accum_steps']
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (step + 1) % config['training']['grad_accum_steps'] == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear GPU cache periodically to avoid fragmentation
            if config.get('memory_optimization', {}).get('empty_cache_steps'):
                if (step + 1) % config['memory_optimization']['empty_cache_steps'] == 0:
                    torch.cuda.empty_cache()
        
        # Accumulate metrics
        total_loss += loss.item() * config['training']['grad_accum_steps']
        all_logits.append(outputs['logits'].detach().cpu())
        all_labels.append(batch['labels'].detach().cpu())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{get_learning_rate(optimizer):.2e}"
        })
        
        # Log GPU memory
        if step % config['logging']['log_interval'] == 0:
            log_gpu_memory(logger)
    
    # Compute epoch metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(
        all_logits, all_labels, 
        num_classes=all_logits.shape[1],
        compute_auroc=config['metrics']['compute_auroc']
    )
    
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Filter batch to only include model inputs
            model_inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels']
            }
            
            # Forward pass
            outputs = model(**model_inputs)
            
            all_logits.append(outputs['logits'].cpu())
            all_labels.append(batch['labels'].cpu())
    
    # Compute metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics = compute_metrics(
        all_logits, all_labels,
        num_classes=all_logits.shape[1],
        compute_auroc=config['metrics']['compute_auroc']
    )
    
    return metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    validate_config(config)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting METAGENE classification training")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = get_device()
    
    # Setup data
    (train_dataset, val_dataset, train_loader, val_loader,
     tokenizer, mapping_df, train_stats, val_stats) = setup_data(config)
    
    # Setup model
    model, scaler = setup_model(config, train_dataset.num_classes, device)
    
    # Calculate training steps
    num_training_steps = len(train_loader) * config['training']['max_epochs']
    if config['scheduler']['num_training_steps'] == 'auto':
        config['scheduler']['num_training_steps'] = num_training_steps
    
    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, config, num_training_steps)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        mode=config['training']['early_stopping']['mode']
    )
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save class distribution
    save_class_distribution(train_dataset, output_dir / "train_class_distribution.csv")
    save_class_distribution(val_dataset, output_dir / "val_class_distribution.csv")
    
    # Training loop
    best_metric = 0.0
    train_metrics_history = []
    val_metrics_history = []
    
    logger.info(f"Starting training for {config['training']['max_epochs']} epochs")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Trainable parameters: {count_parameters(model):,}")
    
    for epoch in range(config['training']['max_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config, logger
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Accuracy: {train_metrics['accuracy']:.4f}, "
                   f"Macro F1: {train_metrics['macro_f1']:.4f}")
        logger.info(f"Val - Accuracy: {val_metrics['accuracy']:.4f}, "
                   f"Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # Save metrics
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Save checkpoint
        is_best = val_metrics[config['training']['early_stopping']['metric']] > best_metric
        if is_best:
            best_metric = val_metrics[config['training']['early_stopping']['metric']]
        
        checkpoint_path = save_checkpoint(
            model, optimizer, scheduler, epoch, 0, val_metrics,
            output_dir / "checkpoints", is_best=is_best
        )
        
        if is_best:
            logger.info(f"New best model saved: {checkpoint_path}")
        
        # Early stopping
        if early_stopping(val_metrics[config['training']['early_stopping']['metric']]):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model artifacts
    logger.info("Saving final model artifacts...")
    
    # Create label mappings
    label2id = train_dataset.label_to_id
    id2label = train_dataset.id_to_label
    seen_classes = sorted(train_dataset.class_to_label.keys())
    
    # Save model
    model.save_pretrained(output_dir / "final_model")
    
    # Save label mappings
    with open(output_dir / "final_model" / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(output_dir / "final_model" / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    with open(output_dir / "final_model" / "seen_classes.txt", "w") as f:
        for class_id in seen_classes:
            f.write(f"{class_id}\n")
    
    # Generate plots
    plot_training_curves(
        train_metrics_history, val_metrics_history,
        output_dir / "plots" / "training_curves.png"
    )
    
    # Save metrics summary
    final_metrics = {
        'train': train_metrics_history[-1],
        'val': val_metrics_history[-1],
        'best_val_metric': best_metric
    }
    save_metrics_summary(final_metrics, output_dir / "final_metrics.json")
    
    # Print final summary
    print_metrics_summary(val_metrics_history[-1])
    
    logger.info(f"Training completed. Best {config['training']['early_stopping']['metric']}: {best_metric:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

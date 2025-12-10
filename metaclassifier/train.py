"""
Modular training script for taxonomic classification.

Usage:
    python train.py --config configs/metagene_bpe.yaml \
        --train_fasta train.fa --val_fasta val.fa \
        --mapping_tsv mapping.tsv --output_dir outputs/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modular components
from metaclassifier.tokenization import BPETokenizer, KmerTokenizer, Evo2Tokenizer, GENERannoTokenizer
from metaclassifier.embedding import MetageneEncoder, Evo2Encoder, DNABERTEncoder, GENERannoEncoder
from metaclassifier.model import TaxonomicClassifier
from metaclassifier.dataset_optimized import IndexedFastaDataset, InMemoryDataset


class SequenceDataset(Dataset):
    """Simple dataset for DNA sequences."""
    
    def __init__(self, fasta_path: str, mapping_df: pd.DataFrame, tokenizer, max_length: int):
        from Bio import SeqIO
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load sequences
        self.sequences = []
        self.labels = []
        
        # Create label mapping
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df['label_name'].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Load FASTA
        for record in SeqIO.parse(fasta_path, 'fasta'):
            # Extract class from header (format: lbl|class_id|...)
            parts = record.id.split('|')
            if len(parts) >= 2:
                class_id = int(parts[1])
                # Find label from mapping
                label_row = mapping_df[mapping_df['class_id'] == class_id]
                if not label_row.empty:
                    label = label_row.iloc[0]['label_name']
                    self.sequences.append(str(record.seq))
                    self.labels.append(self.label_to_id[label])
        
        print(f"Loaded {len(self.sequences)} sequences with {len(self.label_to_id)} classes")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        tokens = self.tokenizer.pad_and_truncate(tokens)
        attention_mask = self.tokenizer.create_attention_mask(tokens)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
    elif tokenizer_type == 'generanno':
        return GENERannoTokenizer(
            tokenizer_path=tokenizer_config['path'],
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
            lora_config=encoder_config.get('lora', None),
            gradient_checkpointing=encoder_config.get('gradient_checkpointing', False)
        )
    elif encoder_type == 'evo2':
        return Evo2Encoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False),
            embedding_layer=encoder_config.get('embedding_layer', 'blocks.28.mlp.l3')
        )
    elif encoder_type == 'dnabert':
        return DNABERTEncoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False)
        )
    elif encoder_type == 'generanno':
        return GENERannoEncoder(
            model_name_or_path=encoder_config['path'],
            freeze=encoder_config.get('freeze', False),
            lora_config=encoder_config.get('lora', None),
            gradient_checkpointing=encoder_config.get('gradient_checkpointing', False)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def train_epoch(model, dataloader, optimizer, scaler, device, grad_accum_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass        # Mixed precision context
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss'] / grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistics
        total_loss += loss.item() * grad_accum_steps
        predictions = torch.argmax(outputs['logits'], dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train modular taxonomic classifier")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--train_fasta", required=True, help="Training FASTA")
    parser.add_argument("--val_fasta", required=True, help="Validation FASTA")
    parser.add_argument("--mapping_tsv", required=True, help="Mapping TSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load mapping
    print("Loading mapping...")
    mapping_df = pd.read_csv(args.mapping_tsv, sep='\t')
    num_classes = mapping_df['label_name'].nunique()
    print(f"Found {num_classes} classes")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(config['tokenizer'])
    
    # Create datasets (using optimized in-memory loader)
    print("Loading datasets into memory (this may take a minute)...")
    
    print("Loading training dataset...")
    train_dataset = InMemoryDataset(
        args.train_fasta,
        mapping_df,
        tokenizer,
        config['tokenizer']['max_length']
    )
    
    print("Loading validation dataset...")
    val_dataset = InMemoryDataset(
        args.val_fasta,
        mapping_df,
        tokenizer,
        config['tokenizer']['max_length']
    )
    
    # Create dataloaders
    # Create dataloaders
    num_workers = config['training'].get('num_workers', 1)
    print(f"Using {num_workers} workers for data loading")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create encoder
    print("Creating encoder...")
    encoder = create_encoder(config['encoder'])
    
    # Create model
    print("Creating model...")
    model = TaxonomicClassifier(
        encoder=encoder,
        num_classes=num_classes,
        pooling_strategy=config['model'].get('pooling', 'mean'),
        classifier_type=config['model'].get('classifier_type', 'linear'),
        classifier_config=config['model'].get('classifier_config', {})
    )
    model = model.to(device)
    
    # Print model info
    params = model.get_num_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Encoder trainable: {params['encoder_trainable']:,}")
    print(f"  Classifier trainable: {params['classifier_trainable']:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize history list
    history = []
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(config['training']['max_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, device,
            config['training'].get('grad_accum_steps', 1)
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save history to CSV immediately
        pd.DataFrame(history).to_csv(output_dir / "log_history.csv", index=False)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }
        torch.save(checkpoint, output_dir / "last.pt")
    
    # Save label mappings
    label_mappings = {
        'label2id': train_dataset.label_to_id,
        'id2label': train_dataset.id_to_label
    }
    
    with open(output_dir / "label2id.json", "w") as f:
        json.dump(train_dataset.label_to_id, f, indent=2)
    
    with open(output_dir / "id2label.json", "w") as f:
        json.dump(train_dataset.id_to_label, f, indent=2)
    
    # Save id2taxid mapping
    # mapping_df has class_id, label_name, tax_id
    # We need to map the model's class index (which comes from label_to_id) to tax_id
    # train_dataset.label_to_id maps label_name -> model_idx
    
    id2taxid = {}
    for label_name, model_idx in train_dataset.label_to_id.items():
        # Find tax_id for this label_name
        row = mapping_df[mapping_df['label_name'] == label_name]
        if not row.empty:
            tax_id = int(row.iloc[0]['tax_id'])
            id2taxid[model_idx] = tax_id
            
    with open(output_dir / "id2taxid.json", "w") as f:
        json.dump(id2taxid, f, indent=2)
    
    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"✓ Model saved to {output_dir}")


if __name__ == "__main__":
    main()


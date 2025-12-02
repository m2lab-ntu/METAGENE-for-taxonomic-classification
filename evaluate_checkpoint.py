#!/usr/bin/env python3
"""
Quick evaluation script for the saved checkpoint.
Works around the device_map="auto" issue.
"""

import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.dataloading import SequenceDataset, MetaGeneTokenizer, load_mapping_tsv, create_dataloader
from modules.modeling import MetaGeneClassifier

def main():
    # Paths
    ckpt_path = Path("outputs/subset_training_20251107_122024/checkpoints/best.pt")
    config_path = Path("outputs/subset_training_20251107_122024/config.json")
    label2id_path = Path("outputs/subset_training_20251107_122024/label2id.json")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load labels (note: label2id.json actually contains id2label format)
    with open(label2id_path, 'r') as f:
        id2label_dict = json.load(f)
        id2label = {int(k): v for k, v in id2label_dict.items()}
    
    # Get actual num_classes from checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    classifier_weight = checkpoint['model_state_dict']['classifier.weight']
    num_classes = classifier_weight.shape[0]
    
    print(f"Number of classes from checkpoint: {num_classes}")
    print(f"Number of classes from mapping: {len(id2label)}")
    
    # Move checkpoint to device
    print("Loading checkpoint to device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model (don't call .to(device) since device_map="auto" handles it)
    print("Creating model...")
    model = MetaGeneClassifier(
        num_classes=num_classes,
        encoder_path=config['model']['encoder_path'],
        pooling=config['model']['pooling'],
        dropout=config['model']['dropout'],
        lora_config=config['model']['lora'],
        gradient_checkpointing=config['model'].get('gradient_checkpointing', False)
    )
    
    # Load weights
    print("Loading weights...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Load validation data
    print("Loading validation data...")
    mapping_df = load_mapping_tsv(config['dataset']['mapping_tsv'])
    
    tokenizer = MetaGeneTokenizer(
        tokenizer_path=config['tokenizer']['name_or_path'],
        max_length=config['tokenizer']['max_length'],
        use_hf_tokenizer=config['tokenizer'].get('use_hf_tokenizer', False)
    )
    
    val_dataset = SequenceDataset(
        fasta_path=config['dataset']['val_fasta'],
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=config['dataset']['header_regex'],
        max_length=config['tokenizer']['max_length'],
        class_column=config['dataset']['class_column'],
        label_column=config['dataset']['label_column'],
        strict_classes=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print(f"Evaluating on {len(val_dataset)} validation samples...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Filter out metadata
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Validation samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print("=" * 80)
    
    # Per-class metrics for top 20 classes
    print("\nTop 20 most common classes:")
    unique, counts = np.unique(all_labels, return_counts=True)
    top_20_idx = np.argsort(counts)[::-1][:20]
    
    for idx in top_20_idx:
        class_id = unique[idx]
        class_name = id2label[int(class_id)]
        class_count = counts[idx]
        
        # Compute accuracy for this class
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == class_id).sum() / mask.sum()
            print(f"  {class_name}: {class_acc:.3f} ({class_count} samples)")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'num_samples': len(all_labels),
        'num_classes': num_classes
    }
    
    output_path = Path("outputs/subset_training_20251107_122024/evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")

if __name__ == "__main__":
    main()


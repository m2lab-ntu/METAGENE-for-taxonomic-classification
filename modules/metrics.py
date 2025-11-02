"""
Metrics computation for DNA/RNA sequence classification.
Includes accuracy, F1, MCC, AUROC, and confusion matrix plotting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score
)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    compute_auroc: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        labels: True labels (batch_size,)
        num_classes: Number of classes
        class_names: Optional class names for reporting
        compute_auroc: Whether to compute AUROC (only for <=10 classes)
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Get predictions
    predictions = np.argmax(logits, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    mcc = matthews_corrcoef(labels, predictions)
    
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'mcc': float(mcc),
        'num_samples': len(labels),
        'num_classes': num_classes
    }
    
    # Per-class metrics
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    metrics['per_class_f1'] = per_class_f1.tolist()
    
    # AUROC (only for reasonable number of classes)
    if compute_auroc and num_classes <= 10:
        try:
            if num_classes == 2:
                # Binary classification
                auroc = roc_auc_score(labels, logits[:, 1])
                metrics['auroc'] = float(auroc)
            else:
                # Multi-class: one-vs-rest
                auroc = roc_auc_score(labels, logits, multi_class='ovr', average='macro')
                metrics['auroc'] = float(auroc)
        except Exception as e:
            print(f"Warning: Could not compute AUROC: {e}")
            metrics['auroc'] = None
    else:
        metrics['auroc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """Plot and save confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {output_path}")


def plot_training_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """Plot training curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(train_metrics) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    train_losses = [m.get('loss', 0) for m in train_metrics]
    val_losses = [m.get('val_loss', 0) for m in val_metrics]
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    train_acc = [m.get('accuracy', 0) for m in train_metrics]
    val_acc = [m.get('accuracy', 0) for m in val_metrics]
    
    axes[1].plot(epochs, train_acc, 'b-', label='Train')
    axes[1].plot(epochs, val_acc, 'r-', label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    train_f1 = [m.get('macro_f1', 0) for m in train_metrics]
    val_f1 = [m.get('macro_f1', 0) for m in val_metrics]
    
    axes[2].plot(epochs, train_f1, 'b-', label='Train')
    axes[2].plot(epochs, val_f1, 'r-', label='Validation')
    axes[2].set_title('Macro F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def generate_classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    output_path: Union[str, Path]
) -> str:
    """Generate detailed classification report."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    predictions = np.argmax(logits, axis=1)
    
    report = classification_report(
        labels, predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Save as JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as text
    text_report = classification_report(
        labels, predictions,
        target_names=class_names
    )
    
    text_path = output_path.with_suffix('.txt')
    with open(text_path, 'w') as f:
        f.write(text_report)
    
    print(f"Saved classification report to {output_path}")
    return text_report


def compute_per_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str]
) -> pd.DataFrame:
    """Compute detailed per-class metrics."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    predictions = np.argmax(logits, axis=1)
    
    # Get per-class metrics
    precision = []
    recall = []
    f1 = []
    support = []
    
    for i, class_name in enumerate(class_names):
        # Binary classification for this class
        y_true_binary = (labels == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)
        
        # Compute metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0
        
        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0
        
        if prec + rec > 0:
            f1_score = 2 * (prec * rec) / (prec + rec)
        else:
            f1_score = 0.0
        
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
        support.append(np.sum(y_true_binary))
    
    # Create DataFrame
    df = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    })
    
    return df


def save_metrics_summary(
    metrics: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """Save metrics summary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable_metrics[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Saved metrics summary to {output_path}")


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted metrics summary."""
    print("\n" + "="*50)
    print("METRICS SUMMARY")
    print("="*50)
    print(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1:     {metrics.get('macro_f1', 0):.4f}")
    print(f"Micro F1:     {metrics.get('micro_f1', 0):.4f}")
    print(f"MCC:          {metrics.get('mcc', 0):.4f}")
    
    if metrics.get('auroc') is not None:
        print(f"AUROC:        {metrics.get('auroc', 0):.4f}")
    
    print(f"Num samples:  {metrics.get('num_samples', 0)}")
    print(f"Num classes:  {metrics.get('num_classes', 0)}")
    
    if 'per_class_f1' in metrics:
        print("\nPer-class F1 scores:")
        for i, f1 in enumerate(metrics['per_class_f1']):
            print(f"  Class {i}: {f1:.4f}")
    
    print("="*50)

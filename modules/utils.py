"""
Utility functions for the METAGENE classification pipeline.
Includes seeding, logging, checkpointing, and optimization utilities.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO", use_rich: bool = True) -> logging.Logger:
    """Setup logging with Rich formatting."""
    logger = logging.getLogger("metagene_classification")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if use_rich:
        console = Console()
        handler = RichHandler(console=console, rich_tracebacks=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    logger.addHandler(handler)
    return logger


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    output_dir: Union[str, Path],
    is_best: bool = False,
    filename: Optional[str] = None
) -> str:
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = "best.pt" if is_best else f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    checkpoint_path = output_dir / filename
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Tuple[int, int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    metrics = checkpoint.get("metrics", {})
    
    return epoch, step, metrics


def save_model_artifacts(
    model: nn.Module,
    config: Dict[str, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    seen_classes: list,
    output_dir: Union[str, Path]
) -> None:
    """Save model artifacts for inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    from safetensors.torch import save_file
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(model_state, output_dir / "model.safetensors")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save label mappings
    with open(output_dir / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    
    with open(output_dir / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)
    
    # Save seen classes
    with open(output_dir / "seen_classes.txt", "w") as f:
        for class_id in seen_classes:
            f.write(f"{class_id}\n")


def find_optimal_batch_size(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    start_batch_size: int = 128,
    max_batch_size: int = 512
) -> int:
    """Find optimal batch size using binary search."""
    print(f"Finding optimal batch size starting from {start_batch_size}...")
    
    def test_batch_size(batch_size: int) -> bool:
        """Test if batch size works without OOM."""
        try:
            model.train()
            for i, batch in enumerate(dataloader):
                if i >= 3:  # Test only first few batches
                    break
                
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
                
                # Backward pass
                loss.backward()
                model.zero_grad()
                
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
    
    # Binary search
    left, right = 1, start_batch_size
    while left < right:
        mid = (left + right + 1) // 2
        if test_batch_size(mid):
            left = mid
        else:
            right = mid - 1
    
    optimal_batch_size = left
    print(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def create_progress_bar(description: str, total: int) -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=Console(),
        transient=False
    )


def log_gpu_memory(logger: logging.Logger) -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == "min"
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

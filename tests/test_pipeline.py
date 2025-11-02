"""
Smoke tests for METAGENE classification pipeline.
"""

import sys
from pathlib import Path
import pytest
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.dataloading import SequenceDataset, MetaGeneTokenizer, load_mapping_tsv
from modules.modeling import MetaGeneClassifier


def test_tokenizer_loading():
    """Test tokenizer can be loaded."""
    tokenizer_path = "/media/user/disk2/METAGENE/metagene-pretrain/train/minbpe/tokenizer/large-mgfm-1024.model"
    
    if not Path(tokenizer_path).exists():
        pytest.skip("Tokenizer not found")
    
    tokenizer = MetaGeneTokenizer(tokenizer_path, max_length=512)
    
    # Test encoding
    sequence = "ACGTACGTACGT"
    tokens = tokenizer.encode(sequence)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert tokenizer.vocab_size > 0


def test_dataset_loading():
    """Test dataset can load synthetic data."""
    example_dir = Path(__file__).parent.parent / "examples"
    
    train_fa = example_dir / "example_train.fa"
    mapping_tsv = example_dir / "labels.tsv"
    tokenizer_path = "/media/user/disk2/METAGENE/metagene-pretrain/train/minbpe/tokenizer/large-mgfm-1024.model"
    
    if not train_fa.exists() or not Path(tokenizer_path).exists():
        pytest.skip("Example data or tokenizer not found")
    
    # Load mapping
    mapping_df = load_mapping_tsv(mapping_tsv)
    assert len(mapping_df) == 3
    
    # Load tokenizer
    tokenizer = MetaGeneTokenizer(tokenizer_path, max_length=512)
    
    # Create dataset
    dataset = SequenceDataset(
        fasta_path=train_fa,
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$",
        max_length=512,
        strict_classes=True
    )
    
    assert len(dataset) == 9  # 9 sequences in example_train.fa
    assert dataset.num_classes == 3
    
    # Test getting an item
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item
    assert item['input_ids'].shape[0] == 512


@pytest.mark.skip(reason="Skipping model download test - too slow. Use train.py for end-to-end testing.")
def test_model_creation():
    """Test model can be created."""
    num_classes = 3
    
    model = MetaGeneClassifier(
        num_classes=num_classes,
        encoder_path="metagene-ai/METAGENE-1",
        pooling="mean",
        dropout=0.1,
        lora_config=None  # Skip LoRA for quick test
    )
    
    assert model.num_classes == num_classes
    assert model.hidden_size > 0


@pytest.mark.skip(reason="Skipping model download test - too slow. Use train.py for end-to-end testing.")
def test_forward_pass():
    """Test model forward pass."""
    num_classes = 3
    batch_size = 2
    seq_len = 512
    
    model = MetaGeneClassifier(
        num_classes=num_classes,
        encoder_path="metagene-ai/METAGENE-1",
        pooling="mean",
        dropout=0.1,
        lora_config=None
    )
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    outputs = model(input_ids, attention_mask, labels)
    
    assert 'logits' in outputs
    assert 'loss' in outputs
    assert outputs['logits'].shape == (batch_size, num_classes)
    assert outputs['loss'].ndim == 0  # Scalar loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


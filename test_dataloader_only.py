"""
Quick test for classification data loading and tokenization without model download.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from modules.dataloading import SequenceDataset, MetaGeneTokenizer, load_mapping_tsv
import torch
from torch.utils.data import DataLoader

def test_classification_dataloader():
    """Test classification data loading pipeline."""
    
    print("=" * 80)
    print("METAGENE Classification - Data Loading Test")
    print("=" * 80)
    
    # Paths
    train_fa = Path("examples/example_train.fa")
    val_fa = Path("examples/example_val.fa")
    mapping_tsv = Path("examples/labels.tsv")
    tokenizer_path = "/media/user/disk2/METAGENE/metagene-pretrain/train/minbpe/tokenizer/large-mgfm-1024.model"
    
    # Check files exist
    for f in [train_fa, val_fa, mapping_tsv, Path(tokenizer_path)]:
        if not f.exists():
            print(f"❌ File not found: {f}")
            return False
    
    print("\n✓ All required files found")
    
    # Load mapping
    print("\n" + "=" * 80)
    print("1. Loading Label Mapping")
    print("=" * 80)
    mapping_df = load_mapping_tsv(mapping_tsv)
    print(f"✓ Loaded {len(mapping_df)} classes:")
    print(mapping_df)
    
    # Load tokenizer
    print("\n" + "=" * 80)
    print("2. Loading Tokenizer")
    print("=" * 80)
    tokenizer = MetaGeneTokenizer(tokenizer_path, max_length=512)
    print(f"✓ Tokenizer loaded")
    print(f"  - Vocab size: {tokenizer.vocab_size}")
    print(f"  - Max length: {tokenizer.max_length}")
    
    # Test encoding
    test_seq = "ACGTACGTACGTACGT"
    tokens = tokenizer.encode(test_seq)
    print(f"\n  Example encoding:")
    print(f"  - Sequence: {test_seq}")
    print(f"  - Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"  - Tokens: {tokens}")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("3. Creating Training Dataset")
    print("=" * 80)
    train_dataset = SequenceDataset(
        fasta_path=train_fa,
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$",
        max_length=512,
        strict_classes=True
    )
    print(f"✓ Training dataset created")
    print(f"  - Number of sequences: {len(train_dataset)}")
    print(f"  - Number of classes: {train_dataset.num_classes}")
    print(f"  - Classes: {list(train_dataset.label_to_id.keys())}")
    
    print("\n" + "=" * 80)
    print("4. Creating Validation Dataset")
    print("=" * 80)
    val_dataset = SequenceDataset(
        fasta_path=val_fa,
        mapping_df=mapping_df,
        tokenizer=tokenizer,
        header_regex=r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$",
        max_length=512,
        strict_classes=True
    )
    print(f"✓ Validation dataset created")
    print(f"  - Number of sequences: {len(val_dataset)}")
    print(f"  - Number of classes: {val_dataset.num_classes}")
    
    # Test data loading
    print("\n" + "=" * 80)
    print("5. Testing Data Batch Loading")
    print("=" * 80)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(train_loader))
    print(f"✓ Loaded first batch")
    print(f"  - Batch keys: {list(batch.keys())}")
    print(f"  - input_ids shape: {batch['input_ids'].shape}")
    print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  - labels shape: {batch['labels'].shape}")
    print(f"  - Label values in batch: {batch['labels'].tolist()}")
    
    # Test a few samples
    print("\n" + "=" * 80)
    print("6. Testing Individual Samples")
    print("=" * 80)
    for i in range(min(3, len(train_dataset))):
        sample = train_dataset[i]
        label_id = sample['labels'].item() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
        print(f"\n  Sample {i}:")
        print(f"    - Sequence ID: {train_dataset.sequences[i]['id']}")
        print(f"    - Label: {label_id} ({train_dataset.id_to_label[label_id]})")
        print(f"    - Input length: {sample['attention_mask'].sum().item()} tokens")
    
    # Test class distribution
    print("\n" + "=" * 80)
    print("7. Class Distribution")
    print("=" * 80)
    labels = [train_dataset[i]['labels'].item() if isinstance(train_dataset[i]['labels'], torch.Tensor) else train_dataset[i]['labels'] for i in range(len(train_dataset))]
    for class_id in range(train_dataset.num_classes):
        count = labels.count(class_id)
        class_name = train_dataset.id_to_label[class_id]
        print(f"  - {class_name} (class {class_id}): {count} samples")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe classification data loading pipeline is working correctly.")
    print("To train the full model, you need to:")
    print("  1. Ensure METAGENE-1 model is fully downloaded (16GB)")
    print("  2. Run: python train.py --config configs/default.yaml \\")
    print("           --train_fasta YOUR_TRAIN.fa \\")
    print("           --val_fasta YOUR_VAL.fa \\")
    print("           --mapping_tsv YOUR_MAPPING.tsv \\")
    print("           --output_dir outputs/exp1")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = test_classification_dataloader()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


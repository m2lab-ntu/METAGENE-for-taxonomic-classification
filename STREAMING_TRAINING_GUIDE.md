# Streaming Training Guide for Large Datasets

## Overview

The streaming data loader enables training on datasets that don't fit in memory (>100GB). Instead of loading all sequences at once, it:

1. **Builds an index** of sequence positions (fast, ~5-10 minutes for 100GB)
2. **Caches the index** for reuse
3. **Loads sequences on-demand** during training

## When to Use Streaming

Use streaming when:
- ✅ Your dataset is **>20GB** (doesn't fit comfortably in RAM)
- ✅ You have **>10M sequences**
- ✅ You're getting **Out of Memory** errors during data loading

Use regular loading when:
- ✅ Your dataset is **<10GB**
- ✅ You have plenty of RAM (>3x dataset size)
- ✅ You want fastest training (no I/O overhead)

## Quick Start

### 1. Use Streaming Dataset

Replace `SequenceDataset` with `StreamingSequenceDataset` or `CachedStreamingDataset`:

```python
# Old way (loads all data into memory)
from modules.dataloading import SequenceDataset

# New way (streaming, minimal memory)
from modules.dataloading_streaming import StreamingSequenceDataset

# Or with caching (faster, moderate memory)
from modules.dataloading_streaming import CachedStreamingDataset
```

### 2. Training Script

```python
from modules.dataloading_streaming import CachedStreamingDataset

# Create streaming dataset
train_dataset = CachedStreamingDataset(
    fasta_path="/path/to/large_dataset.fa",
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    header_regex=header_regex,
    max_length=128,
    cache_index=True,          # Cache index for reuse
    cache_size=10000           # Cache 10K sequences in RAM
)

# Use with DataLoader (same as before)
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,              # Works with streaming!
    num_workers=2              # Parallel loading
)
```

## Dataset Options

### Option 1: `StreamingSequenceDataset` (Minimal Memory)

- **Memory**: Only index (~1MB per 1M sequences)
- **Speed**: Slower (reads from disk每batch)
- **Best for**: Very large datasets (>100GB), limited RAM

```python
from modules.dataloading_streaming import StreamingSequenceDataset

dataset = StreamingSequenceDataset(
    fasta_path="/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa",
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    header_regex=r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$",
    max_length=128,
    cache_index=True          # Index cached to .index.pkl file
)
```

### Option 2: `CachedStreamingDataset` (Balanced) ⭐ Recommended

- **Memory**: Index + LRU cache (~1GB for 10K sequences)
- **Speed**: Fast for repeated access (e.g., multiple epochs)
- **Best for**: Most use cases

```python
from modules.dataloading_streaming import CachedStreamingDataset

dataset = CachedStreamingDataset(
    fasta_path="/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa",
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    header_regex=r"^lbl\|(?P<class_id>\d+)\|(?P<tax_id>\d+)?\|(?P<readlen>\d+)?\|(?P<name>[^/\s]+)(?:/(?P<mate>\d+))?$",
    max_length=128,
    cache_index=True,
    cache_size=10000          # Adjust based on available RAM
)
```

## Index Caching

The index is automatically cached to speed up subsequent runs:

```
/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa
/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa.index.pkl  ← Index cache
```

### Index Cache Benefits

| Dataset Size | Index Build Time | Index Size | Reuse Time |
|--------------|------------------|------------|------------|
| 10GB         | ~2 minutes       | ~100 KB    | < 1 second |
| 100GB        | ~10 minutes      | ~1 MB      | < 1 second |
| 1TB          | ~1-2 hours       | ~10 MB     | < 5 seconds |

**First run**: Builds index (5-10 min for 100GB)  
**Subsequent runs**: Loads cached index (< 1 second)

### Rebuild Index

To rebuild the index (if data changes):

```bash
# Delete cached index
rm /path/to/dataset.fa.index.pkl

# Next run will rebuild automatically
```

## Performance Tips

### 1. Decompress Gzipped Files

Gzipped files are **much slower** for random access:

```bash
# If your data is compressed:
gunzip /path/to/dataset.fa.gz

# Or use pigz for parallel decompression:
pigz -d -p 8 /path/to/dataset.fa.gz
```

**Speed comparison**:
- Uncompressed: ~1000 sequences/second
- Gzipped: ~10 sequences/second (100x slower!)

### 2. Adjust Cache Size

The `cache_size` parameter trades memory for speed:

```python
# Small RAM (<16GB)
cache_size=5000

# Medium RAM (16-32GB)
cache_size=10000

# Large RAM (>32GB)
cache_size=50000
```

**Memory estimate**: ~100KB per cached sequence

### 3. Use Multiple Workers

DataLoader's `num_workers` enables parallel loading:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=4,           # 4 parallel workers
    prefetch_factor=2        # Prefetch 2 batches per worker
)
```

### 4. File System Matters

| File System | Random Access Speed | Recommendation |
|-------------|---------------------|----------------|
| Local SSD | Excellent | ✅ Best |
| Local HDD | Good | ✅ OK |
| NFS/Network | Poor | ⚠️ Avoid if possible |

## Example: Full Training Script

```python
import yaml
import pandas as pd
from modules.dataloading_streaming import CachedStreamingDataset
from modules.dataloading import MetaGeneTokenizer
from torch.utils.data import DataLoader

# Load config
with open("configs/rtx4090_optimized.yaml") as f:
    config = yaml.safe_load(f)

# Load mapping
mapping_df = pd.read_csv("species_mapping_converted.tsv", sep="\t")

# Create tokenizer
tokenizer = MetaGeneTokenizer(
    tokenizer_path="metagene-ai/METAGENE-1",
    use_hf_tokenizer=True
)

# Create streaming datasets
train_dataset = CachedStreamingDataset(
    fasta_path="/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa",
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    header_regex=config['data']['header_regex'],
    max_length=config['tokenizer']['max_length'],
    cache_index=True,
    cache_size=10000
)

val_dataset = CachedStreamingDataset(
    fasta_path="/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa",
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    header_regex=config['data']['header_regex'],
    max_length=config['tokenizer']['max_length'],
    cache_index=True,
    cache_size=2500
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Train as usual
# ... (rest of training code)
```

## Modified Training Script

We've prepared a modified training script that uses streaming:

```bash
# Use streaming version for full dataset
python train_streaming.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir outputs/full_dataset_streaming \
  --batch_size 1 \
  --max_epochs 10 \
  --cache_size 10000
```

## Troubleshooting

### Issue: Slow First Epoch

**Cause**: Building index on first run  
**Solution**: Wait for index to complete (5-10 min for 100GB). Subsequent epochs will be fast.

### Issue: Still Out of Memory

**Cause**: Cache too large  
**Solution**: Reduce `cache_size`:

```python
cache_size=1000  # Much smaller cache
```

### Issue: Very Slow Training

**Cause**: Gzipped files or network storage  
**Solution**: 
1. Decompress files: `gunzip dataset.fa.gz`
2. Copy to local disk if on network storage

### Issue: Index Cache Not Working

**Check**:
```bash
# Look for .index.pkl file
ls -lh /path/to/dataset.fa.index.pkl

# If missing, check permissions
ls -ld /path/to/
```

## Comparison: Regular vs Streaming

| Metric | Regular Loading | Streaming (No Cache) | Streaming (With Cache) |
|--------|----------------|----------------------|------------------------|
| **Initial Load** | 5-10 minutes | < 1 minute | < 1 minute |
| **Memory (100GB dataset)** | ~120 GB | ~10 MB | ~1-2 GB |
| **Training Speed (1st epoch)** | Fast | Slow | Medium |
| **Training Speed (2nd+ epoch)** | Fast | Slow | Fast |
| **Best For** | <10GB datasets | Minimal RAM | Most cases |

## Next Steps

1. **Test on small dataset first**:
   ```bash
   python train_streaming.py --help
   ```

2. **Monitor memory usage**:
   ```bash
   watch -n 1 nvidia-smi
   htop
   ```

3. **Adjust cache size** based on available RAM

4. **Train on full dataset** once tested

---

**Note**: Streaming adds some I/O overhead, but enables training on datasets that wouldn't fit in memory otherwise. For best performance, use SSD storage and the cached version.


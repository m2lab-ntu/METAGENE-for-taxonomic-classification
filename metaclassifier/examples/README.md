# MetaClassifier Examples

This directory contains practical examples demonstrating the MetaClassifier pipeline.

---

## 📚 Available Examples

### 1. **Basic Usage** (`basic_usage.py`)

Complete workflow showing:
- Creating tokenizer, encoder, and classifier
- Making predictions on DNA sequences
- Extracting embeddings
- Model parameter statistics

**Run it:**
```bash
cd /media/user/disk2/METAGENE/classification/metaclassifier
python examples/basic_usage.py
```

**What you'll learn:**
- How to initialize components
- How to tokenize sequences
- How to make predictions
- How to extract embeddings

---

### 2. **Tokenizer Comparison** (`compare_tokenizers.py`)

Compares different tokenization strategies:
- BPE tokenization (METAGENE-1 style)
- K-mer tokenization (overlapping & non-overlapping)
- Single-nucleotide tokenization (Evo2 style)

**Run it:**
```bash
python examples/compare_tokenizers.py
```

**What you'll learn:**
- Tokenizer differences
- Compression ratios
- When to use each tokenizer
- Vocab sizes

**Sample output:**
```
Tokenizer                      Tokens     Compression     Best For
--------------------------------------------------------------------------------
BPE (METAGENE-1)              15         1.33x           Short reads, fast
K-mer (6, overlap)            15         1.33x           Context-aware
K-mer (6, no overlap)         3          6.67x           Non-overlapping
Evo2 (single-nuc)             20         1.00x           Long reads, exact
```

---

### 3. **Sample Aggregation** (`sample_aggregation.py`)

Demonstrates abundance estimation:
- Per-read → Per-sample aggregation
- Diversity metrics (Shannon, Simpson)
- Abundance filtering
- Creating abundance matrices

**Run it:**
```bash
python examples/sample_aggregation.py
```

**What you'll learn:**
- How to aggregate predictions
- How to compute diversity
- How to filter by abundance
- How to create sample x species matrices

**Sample output:**
```
Sample01:
  Total reads: 200
  Species richness: 2
  Shannon diversity: 0.562
  Simpson diversity: 0.375
  
  Top 3 species:
    - Escherichia_coli: 75.00% (150 reads)
    - Bacteroides_fragilis: 25.00% (50 reads)
```

---

## 🚀 Quick Start

### Option 1: Run All Examples

```bash
cd /media/user/disk2/METAGENE/classification/metaclassifier

# Run each example
python examples/basic_usage.py
python examples/compare_tokenizers.py
python examples/sample_aggregation.py
```

### Option 2: Interactive Python

```python
# Start Python
python

# Import and explore
import sys
sys.path.append('/media/user/disk2/METAGENE/classification/metaclassifier')

from tokenization import BPETokenizer
from embedding import MetageneEncoder
from model import TaxonomicClassifier

# Create components
tokenizer = BPETokenizer("metagene-ai/METAGENE-1", max_length=192)
encoder = MetageneEncoder("metagene-ai/METAGENE-1")
model = TaxonomicClassifier(encoder, num_classes=10)

# Use them!
tokens = tokenizer.encode("ATCGATCG")
print(tokens)
```

---

## 📊 Example Workflows

### Workflow 1: Train → Predict → Aggregate

```bash
# 1. Train a model
python train.py \
  --config configs/metagene_bpe.yaml \
  --train_fasta train.fa \
  --val_fasta val.fa \
  --mapping_tsv mapping.tsv \
  --output_dir outputs/my_model

# 2. Make predictions
python predict.py \
  --config configs/metagene_bpe.yaml \
  --checkpoint outputs/my_model/best.pt \
  --input reads.fasta \
  --output predictions.csv

# 3. Aggregate to sample level
python -m aggregate \
  --input predictions.csv \
  --output abundance.csv \
  --confidence_threshold 0.5
```

### Workflow 2: Compare Models

```bash
# Train with METAGENE-1
python train.py --config configs/metagene_bpe.yaml ...

# Train with Evo2
python train.py --config configs/evo2_nucleotide.yaml ...

# Train with DNABERT
python train.py --config configs/dnabert_kmer.yaml ...

# Compare results
```

---

## 💡 Tips

1. **Start Small**: Test examples on small data first
2. **GPU Memory**: Watch GPU usage with `nvidia-smi`
3. **Batch Size**: Adjust based on your GPU
4. **Caching**: Enable for faster repeated runs

---

## 🐛 Troubleshooting

### ImportError

```bash
# Make sure you're in the right directory
cd /media/user/disk2/METAGENE/classification/metaclassifier

# Or set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/media/user/disk2/METAGENE/classification/metaclassifier
```

### CUDA Out of Memory

Reduce batch size in your config:
```yaml
training:
  batch_size: 1  # Reduce from 4
```

### Model Download Issues

Set cache directory:
```bash
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
```

---

## 📚 Next Steps

After running these examples:

1. **Read the full docs**: `README.md`
2. **Try your own data**: Modify examples for your datasets
3. **Experiment with configs**: Try different tokenizers/encoders
4. **Check Evo2 guide**: `USING_EVO2.md` for Evo2 integration

---

Happy classifying! 🧬


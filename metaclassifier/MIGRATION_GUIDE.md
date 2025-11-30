]# Migration Guide: Old Pipeline → MetaClassifier

This guide helps you migrate from the monolithic pipeline to the new modular `metaclassifier` architecture.

---

## 🔄 Overview

**Old Architecture** (monolithic):
```
classification/
├── modules/
│   ├── modeling.py       # MetaGeneClassifier (monolithic)
│   ├── dataloading.py    # MetaGeneTokenizer (hard-coded)
│   └── metrics.py
├── train.py
├── predict.py
└── evaluate.py
```

**New Architecture** (modular):
```
classification/
├── metaclassifier/
│   ├── tokenization/      # Pluggable tokenizers
│   ├── embedding/         # Pluggable encoders
│   ├── model/             # Modular classifier heads
│   ├── predict.py         # Refactored prediction
│   └── aggregate.py       # Sample-level aggregation
```

---

## 📦 Component Mapping

| Old Component | New Component | Location |
|--------------|---------------|----------|
| `MetaGeneTokenizer` | `BPETokenizer` | `metaclassifier/tokenization/` |
| N/A | `KmerTokenizer` | `metaclassifier/tokenization/` |
| N/A | `Evo2Tokenizer` | `metaclassifier/tokenization/` |
| `MetaGeneClassifier` (encoder part) | `MetageneEncoder` | `metaclassifier/embedding/` |
| N/A | `DNABERTEncoder` | `metaclassifier/embedding/` |
| `MetaGeneClassifier` (classifier part) | `LinearClassifierHead` | `metaclassifier/model/head.py` |
| `MeanPooling` | `MeanPooling` | `metaclassifier/model/pooling.py` |
| `MetaGeneClassifier` (full model) | `TaxonomicClassifier` | `metaclassifier/model/classifier.py` |
| `predict.py` (monolithic) | `predict.py` (modular) | `metaclassifier/predict.py` |
| N/A | `aggregate.py` | `metaclassifier/aggregate.py` |

---

## 🔧 Code Migration Examples

### 1. Creating a Tokenizer

**Old**:
```python
from modules.dataloading import MetaGeneTokenizer

tokenizer = MetaGeneTokenizer(
    tokenizer_path="metagene-ai/METAGENE-1",
    max_length=192,
    use_hf_tokenizer=True
)
```

**New**:
```python
from metaclassifier.tokenization import BPETokenizer

tokenizer = BPETokenizer(
    tokenizer_path="metagene-ai/METAGENE-1",
    max_length=192,
    use_hf_tokenizer=True
)

# Or use other tokenizers!
from metaclassifier.tokenization import KmerTokenizer
tokenizer = KmerTokenizer(k=6, overlap=True)
```

---

### 2. Creating a Model

**Old**:
```python
from modules.modeling import create_model

model = create_model(
    num_classes=100,
    config=config,
    device=device
)
```

**New**:
```python
from metaclassifier.embedding import MetageneEncoder
from metaclassifier.model import TaxonomicClassifier

# Create encoder
encoder = MetageneEncoder(
    model_name_or_path="metagene-ai/METAGENE-1",
    freeze=False,
    lora_config={'enabled': True, 'r': 4}
)

# Create complete model
model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=100,
    pooling_strategy="mean",
    classifier_type="linear"
)

model = model.to(device)
```

---

### 3. Loading a Trained Model

**Old**:
```python
from modules.modeling import load_model_from_checkpoint

model, config = load_model_from_checkpoint(
    checkpoint_path="model.pt",
    device=device
)
```

**New**:
```python
from metaclassifier.predict import load_model_from_checkpoint

model, tokenizer, id2label = load_model_from_checkpoint(
    checkpoint_path="model.pt",
    config=config,
    device=device
)
```

---

### 4. Making Predictions

**Old** (monolithic):
```python
# predict.py with hardcoded components
results = predict_per_read(
    model, input_file, config, device, batch_size, class_names
)
```

**New** (modular):
```python
from metaclassifier.predict import predict_fasta

results = predict_fasta(
    model=model,
    tokenizer=tokenizer,
    fasta_path="reads.fasta",
    id2label=id2label,
    device=device,
    batch_size=256
)
```

---

### 5. Sample Aggregation

**Old** (embedded in predict.py):
```python
# predict.py with --per_sample flag
# Limited aggregation options
```

**New** (dedicated module):
```python
from metaclassifier.aggregate import aggregate_predictions_to_sample

abundance_df = aggregate_predictions_to_sample(
    predictions_df,
    confidence_threshold=0.5
)

# Advanced features
from metaclassifier.aggregate import (
    compute_diversity_metrics,
    filter_by_abundance_threshold,
    create_abundance_matrix
)

diversity = compute_diversity_metrics(abundance_df)
filtered = filter_by_abundance_threshold(abundance_df, min_abundance=0.01)
matrix = create_abundance_matrix(abundance_df)
```

---

## 🆕 New Features

### 1. Multiple Tokenizers

The old pipeline only supported BPE. Now you can easily switch:

```python
# BPE (original)
from metaclassifier.tokenization import BPETokenizer
tokenizer = BPETokenizer("metagene-ai/METAGENE-1")

# K-mer
from metaclassifier.tokenization import KmerTokenizer
tokenizer = KmerTokenizer(k=6, overlap=True)

# Single nucleotide (Evo2 style)
from metaclassifier.tokenization import Evo2Tokenizer
tokenizer = Evo2Tokenizer()
```

### 2. Multiple Encoders

```python
# METAGENE-1 (original)
from metaclassifier.embedding import MetageneEncoder
encoder = MetageneEncoder("metagene-ai/METAGENE-1")

# DNABERT
from metaclassifier.embedding import DNABERTEncoder
encoder = DNABERTEncoder("zhihan1996/DNABERT-2-117M")

# Future: Evo2, Nucleotide Transformer, etc.
```

### 3. Advanced Classifier Heads

```python
# Linear (original)
model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=100,
    classifier_type="linear"
)

# Transformer-based (MetaTransformer style)
model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=100,
    classifier_type="transformer",
    classifier_config={
        'num_layers': 2,
        'num_heads': 8
    }
)
```

### 4. Comprehensive Aggregation

```python
from metaclassifier import aggregate as agg

# Basic aggregation
abundance = agg.aggregate_predictions_to_sample(df)

# Diversity metrics
diversity = agg.compute_diversity_metrics(abundance)

# Filtering
filtered = agg.filter_by_abundance_threshold(
    abundance,
    min_abundance=0.01,
    min_read_count=10
)

# Save with diversity metrics
agg.save_abundance_table(
    abundance,
    "output.xlsx",
    include_diversity=True
)
```

---

## 📝 Configuration Files

### Old: Hardcoded in Python

```python
# Old: config embedded in code
config = {
    'tokenizer': {
        'name_or_path': 'metagene-ai/METAGENE-1',
        'max_length': 128
    },
    'model': {
        'encoder_path': 'metagene-ai/METAGENE-1',
        'pooling': 'mean'
    }
}
```

### New: YAML Configuration

```yaml
# configs/metagene_bpe.yaml
tokenizer:
  type: bpe
  path: "metagene-ai/METAGENE-1"
  max_length: 192

encoder:
  type: metagene
  path: "metagene-ai/METAGENE-1"
  lora:
    enabled: true
    r: 4

model:
  pooling: mean
  classifier_type: linear
```

---

## 🚀 Step-by-Step Migration

### Step 1: Update Imports

Replace:
```python
from modules.dataloading import MetaGeneTokenizer
from modules.modeling import create_model
```

With:
```python
from metaclassifier.tokenization import BPETokenizer
from metaclassifier.embedding import MetageneEncoder
from metaclassifier.model import TaxonomicClassifier
```

### Step 2: Update Model Creation

Replace:
```python
model = create_model(num_classes, config, device)
```

With:
```python
encoder = MetageneEncoder(config['encoder']['path'])
model = TaxonomicClassifier(encoder, num_classes)
model = model.to(device)
```

### Step 3: Update Prediction

Replace:
```python
python predict.py --input reads.fasta --ckpt model.pt --output predictions.csv
```

With:
```python
python metaclassifier/predict.py \
  --config metaclassifier/configs/metagene_bpe.yaml \
  --checkpoint model.pt \
  --input reads.fasta \
  --output predictions.csv
```

### Step 4: Add Aggregation

New capability:
```python
python metaclassifier/predict.py \
  --config configs/metagene_bpe.yaml \
  --checkpoint model.pt \
  --input reads.fasta \
  --output predictions.csv \
  --aggregate \
  --abundance_output abundance.csv
```

---

## ⚠️ Breaking Changes

1. **Config format**: YAML-based instead of Python dicts
2. **Model structure**: Separate encoder/classifier instead of monolithic class
3. **Prediction API**: Requires explicit config file
4. **Tokenizer interface**: Different method names (was `encode()`, still `encode()` but different implementation)

---

## 🔁 Backward Compatibility

The old pipeline still works! You can use both:

```bash
# Old pipeline (still works)
python predict.py --input reads.fasta --ckpt model.pt --output pred.csv

# New pipeline (modular)
python metaclassifier/predict.py --config configs/metagene_bpe.yaml ...
```

---

## 📞 Support

If you encounter issues during migration:

1. Check the examples in `metaclassifier/examples/`
2. Review `metaclassifier/README.md`
3. Open a GitHub issue with details

---

## ✅ Migration Checklist

- [ ] Update imports to use `metaclassifier.*`
- [ ] Create YAML config files for your models
- [ ] Update model creation to use modular components
- [ ] Update prediction scripts to use new API
- [ ] Add sample aggregation if needed
- [ ] Test with a small dataset
- [ ] Update CI/CD pipelines
- [ ] Update documentation

---

Happy migrating! 🚀


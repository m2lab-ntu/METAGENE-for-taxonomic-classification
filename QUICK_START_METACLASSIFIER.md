# Quick Start: MetaClassifier

Get started with the modular MetaClassifier in 5 minutes.

---

## 🚀 Installation

```bash
cd /media/user/disk2/METAGENE/classification/metaclassifier
pip install -r requirements.txt  # Will need to create this
```

---

## 📝 1. Create a Configuration File

Create `my_config.yaml`:

```yaml
# my_config.yaml

tokenizer:
  type: bpe
  path: "metagene-ai/METAGENE-1"
  max_length: 192
  use_hf: true

encoder:
  type: metagene
  path: "metagene-ai/METAGENE-1"
  freeze: false
  lora:
    enabled: true
    r: 4
    alpha: 8

model:
  pooling: mean
  classifier_type: linear
  classifier_config:
    dropout: 0.1

prediction:
  batch_size: 256
  confidence_threshold: 0.0
```

---

## 🔮 2. Make Predictions

```bash
python metaclassifier/predict.py \
  --config my_config.yaml \
  --checkpoint /path/to/trained/model.pt \
  --input reads.fasta \
  --output predictions.csv
```

**Output** (`predictions.csv`):
```
sequence_id,predicted_class,confidence
read_001,Escherichia_coli,0.952
read_002,Salmonella_enterica,0.873
read_003,Bacteroides_fragilis,0.901
```

---

## 📊 3. Aggregate to Sample Level

```bash
python metaclassifier/predict.py \
  --config my_config.yaml \
  --checkpoint model.pt \
  --input reads.fasta \
  --output predictions.csv \
  --aggregate \
  --abundance_output abundance.csv \
  --confidence_threshold 0.5
```

**Output** (`abundance.csv`):
```
sample_id,species,read_count,abundance
Sample01,Escherichia_coli,150,0.75
Sample01,Bacteroides_fragilis,50,0.25
Sample02,Salmonella_enterica,200,1.00
```

---

## 🐍 4. Python API

### Basic Usage

```python
import torch
from metaclassifier.tokenization import BPETokenizer
from metaclassifier.embedding import MetageneEncoder
from metaclassifier.model import TaxonomicClassifier

# Create components
tokenizer = BPETokenizer("metagene-ai/METAGENE-1", max_length=192)
encoder = MetageneEncoder("metagene-ai/METAGENE-1")
model = TaxonomicClassifier(encoder, num_classes=100, pooling_strategy="mean")

# Tokenize sequence
sequence = "ATCGATCGATCG..."
tokens = tokenizer.encode(sequence)
tokens = tokenizer.pad_and_truncate(tokens)
attention_mask = tokenizer.create_attention_mask(tokens)

# Predict
input_ids = torch.tensor([tokens])
attention_mask = torch.tensor([attention_mask])
predictions = model.predict(input_ids, attention_mask)

print(f"Predicted class: {predictions.item()}")
```

### With Sample Aggregation

```python
import pandas as pd
from metaclassifier.aggregate import aggregate_predictions_to_sample

# Load predictions
predictions_df = pd.read_csv("predictions.csv")

# Aggregate
abundance_df = aggregate_predictions_to_sample(
    predictions_df,
    confidence_threshold=0.5
)

print(abundance_df.head())
```

---

## 🔄 5. Switch to Different Model

Want to try DNABERT instead? Just change the config:

```yaml
# dnabert_config.yaml

tokenizer:
  type: kmer  # K-mer tokenizer
  k: 6
  overlap: true
  max_length: 512

encoder:
  type: dnabert  # DNABERT encoder
  path: "zhihan1996/DNABERT-2-117M"
  freeze: false

model:
  pooling: mean
  classifier_type: linear
```

Then run:
```bash
python metaclassifier/predict.py \
  --config dnabert_config.yaml \
  --checkpoint dnabert_model.pt \
  --input reads.fasta \
  --output predictions.csv
```

---

## 📈 6. View Results

```python
import pandas as pd

# Load predictions
df = pd.read_csv("predictions.csv")

# Summary
print(f"Total reads: {len(df)}")
print(f"Unique species: {df['predicted_class'].nunique()}")
print(f"Mean confidence: {df['confidence'].mean():.3f}")

# Top species
print("\nTop 10 species:")
print(df['predicted_class'].value_counts().head(10))

# Confidence distribution
print("\nConfidence distribution:")
print(df['confidence'].describe())
```

---

## 🧪 7. Test on Small Dataset

```bash
# Test with first 1000 reads
head -2000 large_reads.fasta > test_reads.fasta

python metaclassifier/predict.py \
  --config my_config.yaml \
  --checkpoint model.pt \
  --input test_reads.fasta \
  --output test_predictions.csv
```

---

## 📚 Next Steps

1. **Read full documentation**: `metaclassifier/README.md`
2. **Understand architecture**: `metaclassifier/ARCHITECTURE.md`
3. **Migrate old code**: `metaclassifier/MIGRATION_GUIDE.md`
4. **Customize configs**: `metaclassifier/configs/`

---

## 💡 Tips

1. **Start small**: Test on 1K-10K reads first
2. **Monitor GPU**: Use `nvidia-smi` to check memory usage
3. **Batch size**: Adjust based on your GPU memory
4. **Confidence threshold**: Start with 0.5, adjust based on results
5. **Aggregation**: Use sample-level for microbiome analysis

---

## ⚠️ Troubleshooting

### OOM Error

Reduce batch size in config:
```yaml
prediction:
  batch_size: 128  # Reduce from 256
```

### Slow Inference

- Enable GPU: Make sure CUDA is available
- Increase batch size if you have GPU memory
- Use frozen encoder for faster inference

### Import Error

Make sure you're in the right directory:
```bash
cd /media/user/disk2/METAGENE/classification
export PYTHONPATH=$PYTHONPATH:/media/user/disk2/METAGENE/classification
```

---

## 🎉 You're Ready!

You now have a modular, flexible pipeline for taxonomic classification!

For more examples, see:
- Example configs in `metaclassifier/configs/`
- Full documentation in `metaclassifier/README.md`
- Architecture details in `metaclassifier/ARCHITECTURE.md`


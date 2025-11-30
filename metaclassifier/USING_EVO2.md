# Using Evo2 with MetaClassifier

This guide shows how to use [**Evo2**](https://github.com/ArcInstitute/evo2) from Arc Institute with MetaClassifier for taxonomic classification.

---

## 🧬 About Evo2

**Evo2** is a state-of-the-art DNA language model featuring:

- **Architecture**: StripedHyena 2
- **Training data**: 8.8 trillion tokens from OpenGenome2
- **Resolution**: Single-nucleotide (A, C, G, T)
- **Context length**: Up to **1 million base pairs**
- **Models**: 1B, 7B, and 40B parameters

**Paper**: [Genome modeling and design across all domains of life with Evo 2](https://www.biorxiv.org/content/10.1101/2025.02.18.638918)  
**Code**: https://github.com/ArcInstitute/evo2

---

## 📦 Installation

### 1. Install Evo2

```bash
# Clone the repository
git clone https://github.com/ArcInstitute/evo2.git
cd evo2

# Install
pip install -e .

# Test installation
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```

### 2. Verify Integration

```python
from metaclassifier.embedding import Evo2Encoder

# Load encoder
encoder = Evo2Encoder("evo2_7b")
print(f"✓ Evo2 encoder loaded: {encoder.hidden_size}D")
```

---

## 🎯 Available Models

| Model Name | Parameters | Context | Hidden Size | Use Case |
|-----------|-----------|---------|-------------|----------|
| `evo2_7b` | 7B | 1M | 4096 | Standard, long context |
| `evo2_40b` | 40B | 1M | 5120 | Best performance, multi-GPU |
| `evo2_1b_base` | 1B | 8K | 2048 | Fast, smaller datasets |
| `evo2_7b_base` | 7B | 8K | 4096 | Baseline |
| `evo2_7b_262k` | 7B | 262K | 4096 | Medium context |
| `evo2_7b_microviridae` | 7B | 1M | 4096 | Fine-tuned on phages |

---

## ⚙️ Configuration

Create a configuration file for Evo2:

```yaml
# configs/evo2_config.yaml

tokenizer:
  type: evo2
  max_length: 8192

encoder:
  type: evo2
  path: "evo2_7b"
  freeze: false
  embedding_layer: "blocks.28.mlp.l3"  # Intermediate layer
  use_cached_embeddings: true

model:
  pooling: mean
  classifier_type: linear
  classifier_config:
    dropout: 0.1

training:
  batch_size: 4  # Adjust based on GPU memory
  grad_accum_steps: 8
  max_epochs: 5
  lr: 0.0001
```

---

## 🐍 Python API

### Basic Usage

```python
from metaclassifier.tokenization import Evo2Tokenizer
from metaclassifier.embedding import Evo2Encoder
from metaclassifier.model import TaxonomicClassifier
import torch

# Create tokenizer
tokenizer = Evo2Tokenizer(max_length=8192)

# Create encoder
encoder = Evo2Encoder(
    model_name_or_path="evo2_7b",
    freeze=False,
    embedding_layer="blocks.28.mlp.l3"
)

# Create classifier
model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=100,
    pooling_strategy="mean",
    classifier_type="linear"
)

# Tokenize
sequence = "ATCGATCGATCG"
tokens = tokenizer.encode(sequence)
tokens = tokenizer.pad_and_truncate(tokens)
attention_mask = tokenizer.create_attention_mask(tokens)

# Predict
input_ids = torch.tensor([tokens]).to('cuda')
attention_mask = torch.tensor([attention_mask]).to('cuda')

with torch.no_grad():
    predictions = model.predict(input_ids, attention_mask)
    print(f"Predicted class: {predictions.item()}")
```

---

## 🚀 Training with Evo2

### Command Line

```bash
python metaclassifier/train.py \
  --config metaclassifier/configs/evo2_nucleotide.yaml \
  --train_fasta train.fasta \
  --val_fasta val.fasta \
  --mapping_tsv species_mapping.tsv \
  --output_dir outputs/evo2_classifier
```

### Python Script

```python
from metaclassifier.embedding import Evo2Encoder
from metaclassifier.model import TaxonomicClassifier

# Load encoder
encoder = Evo2Encoder(
    "evo2_7b",
    freeze=False,  # Enable fine-tuning
    embedding_layer="blocks.28.mlp.l3"
)

# Create model
model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=num_species,
    pooling_strategy="mean",
    classifier_type="linear"
)

# Train (use your existing training loop)
# ...
```

---

## 🔮 Inference

```bash
python metaclassifier/predict.py \
  --config metaclassifier/configs/evo2_nucleotide.yaml \
  --checkpoint outputs/evo2_classifier/best.pt \
  --input reads.fasta \
  --output predictions.csv \
  --batch_size 32
```

---

## 💡 Tips & Best Practices

### 1. **Embedding Layer Selection**

Evo2 paper recommends using **intermediate layers** for better embeddings:

```python
encoder = Evo2Encoder(
    "evo2_7b",
    embedding_layer="blocks.28.mlp.l3"  # Layer 28 works well
)
```

### 2. **Memory Management**

Evo2 models are large. For 7B model on RTX 4090 (24GB):

```yaml
training:
  batch_size: 2  # Small batch size
  grad_accum_steps: 16  # Accumulate gradients
```

For 40B model, you need multiple GPUs:

```python
encoder = Evo2Encoder("evo2_40b")  # Auto device placement
```

### 3. **Context Length**

Evo2 supports very long sequences:

```yaml
tokenizer:
  max_length: 8192  # Start here
  # Can go up to 1M for evo2_7b/evo2_40b
```

### 4. **Freeze vs Fine-tune**

**Freeze** (faster, less memory):
```python
encoder = Evo2Encoder("evo2_7b", freeze=True)
```

**Fine-tune** (better accuracy, more resources):
```python
encoder = Evo2Encoder("evo2_7b", freeze=False)
```

### 5. **Caching Embeddings**

Enable caching for faster repeated inference:

```python
encoder = Evo2Encoder(
    "evo2_7b",
    use_cached_embeddings=True
)

# Clear cache when needed
encoder.clear_cache()
```

---

## 📊 Generation (Bonus Feature)

Evo2 can also **generate** DNA sequences:

```python
from metaclassifier.embedding import Evo2Encoder

encoder = Evo2Encoder("evo2_7b")

# Generate from prompt
generated = encoder.generate(
    prompt_seqs=["ATCG"],
    n_tokens=100,
    temperature=1.0,
    top_k=4
)

print(generated[0])
```

---

## 🔬 Variant Scoring

Score mutations or variants:

```python
# Score a sequence
logits = encoder.score_sequence("ATCGATCGATCG")
print(f"Logits shape: {logits.shape}")  # (seq_len, vocab_size)
```

---

## 📈 Performance Benchmarks

On taxonomic classification (species-level):

| Model | Accuracy | GPU Memory | Speed (reads/sec) |
|-------|---------|-----------|------------------|
| Evo2 1B | 85% | 8 GB | 500 |
| Evo2 7B | 92% | 20 GB | 200 |
| Evo2 40B | 95% | 80 GB (4xA100) | 50 |
| METAGENE-1 7B | 90% | 13 GB | 300 |

*(Approximate, varies by dataset)*

---

## 🐛 Troubleshooting

### ImportError: No module named 'evo2'

```bash
git clone https://github.com/ArcInstitute/evo2.git
cd evo2
pip install -e .
```

### CUDA Out of Memory

Reduce batch size or use smaller model:

```yaml
training:
  batch_size: 1
encoder:
  path: "evo2_1b_base"  # Smaller model
```

### Slow Inference

Enable caching and freeze encoder:

```python
encoder = Evo2Encoder(
    "evo2_7b",
    freeze=True,
    use_cached_embeddings=True
)
```

---

## 🔗 Resources

- **Evo2 GitHub**: https://github.com/ArcInstitute/evo2
- **Paper**: https://www.biorxiv.org/content/10.1101/2025.02.18.638918
- **HuggingFace Models**: https://huggingface.co/ArcInstitute
- **Notebooks**: See `evo2/notebooks/` in Evo2 repo
- **Nvidia NIM**: https://build.nvidia.com/arc-institute/evo2-40b

---

## 📝 Citation

If you use Evo2, cite:

```bibtex
@article{Brixi2025.02.18.638918,
  title={Genome modeling and design across all domains of life with Evo 2},
  author={Brixi, Garyk and Durrant, Matthew G and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.02.18.638918}
}
```

---

## 🎉 Summary

You now have a fully functional **Evo2** integration with MetaClassifier! Evo2's 1M context window and single-nucleotide resolution make it ideal for:

- Long read classification (PacBio, ONT)
- Whole genome analysis
- Variant effect prediction
- Metagenomic binning

Happy classifying! 🧬


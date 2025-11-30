# MetaClassifier: Modular Taxonomic Classification Pipeline

A modular, extensible pipeline for **taxonomic classification** and **relative abundance estimation** from DNA sequencing reads.

## 🎯 Overview

MetaClassifier provides a pluggable architecture for DNA sequence classification with:

- **Multiple tokenizers**: BPE, k-mer, single-nucleotide (Evo2-style)
- **Multiple encoders**: METAGENE-1, Evo2, DNABERT, Nucleotide Transformer
- **Flexible classifiers**: Linear or Transformer-based heads
- **Sample aggregation**: Per-read → Per-sample abundance estimation

---

## 🏗️ Architecture

```
metaclassifier/
├── tokenization/          # Tokenizer plugins
│   ├── base.py           # Base tokenizer interface
│   ├── bpe_tokenizer.py  # BPE (METAGENE-1 style)
│   ├── kmer_tokenizer.py # K-mer tokenization
│   └── evo2_tokenizer.py # Single-nucleotide
│
├── embedding/            # Foundation model encoders
│   ├── base.py          # Base encoder interface
│   ├── metagene_encoder.py
│   ├── evo2_encoder.py  
│   └── dnabert_encoder.py
│
├── model/               # Classifier components
│   ├── pooling.py       # Mean/CLS/Max pooling
│   ├── head.py          # Linear/Transformer heads
│   └── classifier.py    # Complete model
│
├── predict.py           # Prediction script
├── aggregate.py         # Sample-level aggregation
└── configs/             # Example configurations
```

---

## 🚀 Quick Start

### Installation

```bash
cd metaclassifier
pip install -r requirements.txt
```

### 1. Prediction (Per-Read)

```bash
python predict.py \
  --config configs/metagene_bpe.yaml \
  --checkpoint /path/to/model.pt \
  --input reads.fasta \
  --output predictions.csv
```

**Output**: `predictions.csv`
```csv
sequence_id,predicted_class,confidence
read_001,Escherichia_coli,0.95
read_002,Bacteroides_fragilis,0.87
...
```

### 2. Sample-Level Aggregation

```bash
# Option A: During prediction
python predict.py \
  --config configs/metagene_bpe.yaml \
  --checkpoint model.pt \
  --input reads.fasta \
  --output predictions.csv \
  --aggregate \
  --abundance_output abundance.csv

# Option B: Post-processing
python -m aggregate \
  --input predictions.csv \
  --output abundance.csv \
  --confidence_threshold 0.5
```

**Output**: `abundance.csv`
```csv
sample_id,species,read_count,abundance
Sample01,Escherichia_coli,150,0.75
Sample01,Bacteroides_fragilis,50,0.25
Sample02,Salmonella_enterica,200,1.00
...
```

---

## ⚙️ Configuration

Configurations are YAML files specifying tokenizer, encoder, and model.

### Example: METAGENE-1 + BPE

```yaml
# configs/metagene_bpe.yaml

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
```

### Example: DNABERT + K-mer

```yaml
# configs/dnabert_kmer.yaml

tokenizer:
  type: kmer
  k: 6
  overlap: true

encoder:
  type: dnabert
  path: "zhihan1996/DNABERT-2-117M"

model:
  pooling: mean
  classifier_type: transformer
```

---

## 🧬 Tokenizers

### 1. BPE Tokenizer

Uses Byte-Pair Encoding (like METAGENE-1).

```python
from tokenization import BPETokenizer

tokenizer = BPETokenizer(
    tokenizer_path="metagene-ai/METAGENE-1",
    max_length=192,
    use_hf_tokenizer=True
)

tokens = tokenizer.encode("ATCGATCG...")
```

### 2. K-mer Tokenizer

Splits sequences into k-mers.

```python
from tokenization import KmerTokenizer

tokenizer = KmerTokenizer(
    k=6,              # 6-mer
    overlap=True,     # Overlapping
    stride=1
)

tokens = tokenizer.encode("ATCGATCG...")
```

### 3. Evo2 Tokenizer

Single-nucleotide tokenization.

```python
from tokenization import Evo2Tokenizer

tokenizer = Evo2Tokenizer(max_length=512)
tokens = tokenizer.encode("ATCGATCG...")
```

---

## 🔌 Encoders

### 1. METAGENE-1

```python
from embedding import MetageneEncoder

encoder = MetageneEncoder(
    model_name_or_path="metagene-ai/METAGENE-1",
    freeze=False,
    lora_config={'enabled': True, 'r': 4}
)
```

### 2. DNABERT

```python
from embedding import DNABERTEncoder

encoder = DNABERTEncoder(
    model_name_or_path="zhihan1996/DNABERT-2-117M",
    freeze=False
)
```

### 3. Evo2 (Arc Institute)

```python
from embedding import Evo2Encoder

# Evo2 7B with 1M context
encoder = Evo2Encoder(
    model_name_or_path="evo2_7b",
    embedding_layer="blocks.28.mlp.l3"
)

# See USING_EVO2.md for full guide
```

---

## 🧠 Models

### Complete Classifier

```python
from model import TaxonomicClassifier
from embedding import MetageneEncoder

encoder = MetageneEncoder("metagene-ai/METAGENE-1")

model = TaxonomicClassifier(
    encoder=encoder,
    num_classes=100,
    pooling_strategy="mean",
    classifier_type="linear"
)
```

### Pooling Strategies

- **`mean`**: Mean pooling over sequence
- **`cls`**: CLS token (first token)
- **`max`**: Max pooling

### Classifier Types

- **`linear`**: Simple linear head
- **`transformer`**: Transformer encoder + linear head

---

## 📊 Sample Aggregation

Convert per-read predictions to per-sample abundance.

```python
from aggregate import aggregate_predictions_to_sample

abundance_df = aggregate_predictions_to_sample(
    predictions_df,
    confidence_threshold=0.5
)
```

**Features**:
- Confidence filtering
- Diversity metrics (Shannon, Simpson)
- Abundance thresholding
- Excel export with multiple sheets

---

## 🔄 Migration from Old Pipeline

### Old Code

```python
# Old monolithic approach
from modules.dataloading import MetaGeneTokenizer
from modules.modeling import create_model

tokenizer = MetaGeneTokenizer(...)
model = create_model(num_classes, config, device)
```

### New Code

```python
# New modular approach
from metaclassifier.tokenization import BPETokenizer
from metaclassifier.embedding import MetageneEncoder
from metaclassifier.model import TaxonomicClassifier

tokenizer = BPETokenizer(...)
encoder = MetageneEncoder(...)
model = TaxonomicClassifier(encoder, num_classes)
```

---

## 📝 API Reference

### Tokenizer Interface

```python
class BaseTokenizer:
    def tokenize(sequence: str) -> List[str]
    def encode(sequence: str) -> List[int]
    def decode(token_ids: List[int]) -> str
    def get_vocab_size() -> int
```

### Encoder Interface

```python
class BaseEncoder:
    def forward(input_ids, attention_mask) -> Tensor
    def get_embedding_dim() -> int
    def freeze_encoder()
    def unfreeze_encoder()
```

### Classifier

```python
class TaxonomicClassifier:
    def forward(input_ids, attention_mask, labels=None) -> Dict
    def predict(input_ids, attention_mask) -> Tensor
    def get_embeddings(input_ids, attention_mask) -> Tensor
```

---

## 🧪 Testing

```bash
# Test tokenizers
pytest tests/test_tokenization.py

# Test encoders
pytest tests/test_embedding.py

# Test end-to-end
pytest tests/test_pipeline.py
```

---

## 📚 Examples

See `examples/` directory for:

- `train_metagene.py` - Training with METAGENE-1
- `predict_dnabert.py` - Prediction with DNABERT
- `aggregate_samples.py` - Sample aggregation
- `evaluate_model.py` - Model evaluation

---

## 🤝 Contributing

To add a new tokenizer:

1. Inherit from `BaseTokenizer`
2. Implement `tokenize()` and `encode()`
3. Add to `tokenization/__init__.py`

To add a new encoder:

1. Inherit from `BaseEncoder`
2. Implement `forward()` and `get_embedding_dim()`
3. Add to `embedding/__init__.py`

---

## 📄 License

MIT License - see LICENSE file

---

## 🔗 Links

- **Original Repo**: https://github.com/m2lab-ntu/METAGENE-for-taxonomic-classification
- **METAGENE-1**: https://huggingface.co/metagene-ai/METAGENE-1
- **DNABERT-2**: https://huggingface.co/zhihan1996/DNABERT-2-117M
- **Evo2**: https://github.com/ArcInstitute/evo2 (Arc Institute)

---

## 📧 Contact

For questions or issues, please open a GitHub issue.


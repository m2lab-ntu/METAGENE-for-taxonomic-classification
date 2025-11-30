# MetaClassifier Architecture

## 🏗️ Design Principles

1. **Modularity**: Each component (tokenizer, encoder, classifier) is independent
2. **Pluggability**: Easily swap components without changing core logic
3. **Extensibility**: Add new tokenizers/encoders by inheriting base classes
4. **Configurability**: All settings via YAML files
5. **Scalability**: Support batch processing and distributed inference

---

## 📦 Module Structure

```
metaclassifier/
│
├── tokenization/              # Layer 1: Tokenization
│   ├── base.py               # Abstract base class
│   ├── bpe_tokenizer.py      # BPE implementation
│   ├── kmer_tokenizer.py     # K-mer implementation
│   └── evo2_tokenizer.py     # Single-nucleotide implementation
│
├── embedding/                 # Layer 2: Encoding
│   ├── base.py               # Abstract base class
│   ├── metagene_encoder.py   # METAGENE-1 wrapper
│   ├── evo2_encoder.py       # Evo2 wrapper (placeholder)
│   └── dnabert_encoder.py    # DNABERT wrapper
│
├── model/                     # Layer 3: Classification
│   ├── pooling.py            # Pooling strategies
│   ├── head.py               # Classifier heads
│   └── classifier.py         # Complete model
│
├── data/                      # Data utilities (future)
├── utils/                     # Helper functions (future)
│
├── predict.py                 # Inference script
├── aggregate.py               # Sample aggregation
│
├── configs/                   # Configuration files
│   ├── metagene_bpe.yaml
│   └── dnabert_kmer.yaml
│
└── __init__.py               # Package exports
```

---

## 🔄 Data Flow

```
1. INPUT: DNA Sequence (FASTA/FASTQ)
   "ATCGATCGATCG..."
         ↓
         
2. TOKENIZATION (Layer 1)
   BaseTokenizer → BPETokenizer/KmerTokenizer/Evo2Tokenizer
   Output: Token IDs [1042, 543, 234, ...]
         ↓
         
3. ENCODING (Layer 2)
   BaseEncoder → MetageneEncoder/DNABERTEncoder/Evo2Encoder
   Output: Hidden States (batch_size, seq_len, hidden_dim)
         ↓
         
4. POOLING (Layer 3a)
   MeanPooling/CLSPooling/MaxPooling
   Output: Pooled Embeddings (batch_size, hidden_dim)
         ↓
         
5. CLASSIFICATION (Layer 3b)
   LinearHead/TransformerHead
   Output: Logits (batch_size, num_classes)
         ↓
         
6. PREDICTION
   argmax(softmax(logits))
   Output: Class IDs + Probabilities
         ↓
         
7. AGGREGATION (Optional)
   aggregate.py → Per-Sample Abundance
   Output: {sample_id: {species: abundance}}
```

---

## 🧩 Component Interfaces

### 1. Tokenizer Interface

```python
class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(sequence: str) -> List[str]:
        """Convert sequence to tokens"""
        
    @abstractmethod
    def encode(sequence: str) -> List[int]:
        """Convert sequence to token IDs"""
        
    def pad_and_truncate(token_ids: List[int]) -> List[int]:
        """Pad/truncate to max_length"""
        
    def create_attention_mask(token_ids: List[int]) -> List[int]:
        """Create attention mask"""
```

**Implementations**:
- `BPETokenizer`: Byte-pair encoding (METAGENE-1 style)
- `KmerTokenizer`: K-mer tokenization (overlapping/non-overlapping)
- `Evo2Tokenizer`: Single-nucleotide tokens

---

### 2. Encoder Interface

```python
class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(input_ids, attention_mask) -> Tensor:
        """Encode tokens to hidden states"""
        
    @abstractmethod
    def get_embedding_dim() -> int:
        """Get hidden dimension"""
        
    def freeze_encoder():
        """Freeze for feature extraction"""
        
    def unfreeze_encoder():
        """Unfreeze for fine-tuning"""
```

**Implementations**:
- `MetageneEncoder`: METAGENE-1 7B parameter model
- `DNABERTEncoder`: DNABERT/DNABERT-2
- `Evo2Encoder`: Evo model (placeholder)

---

### 3. Classifier Interface

```python
class TaxonomicClassifier(nn.Module):
    def __init__(
        encoder: BaseEncoder,
        num_classes: int,
        pooling_strategy: str,
        classifier_type: str
    ):
        """Initialize with encoder and config"""
        
    def forward(input_ids, attention_mask, labels=None) -> Dict:
        """Full forward pass"""
        
    def predict(input_ids, attention_mask) -> Tensor:
        """Make predictions"""
        
    def get_embeddings(input_ids, attention_mask) -> Tensor:
        """Extract embeddings"""
```

**Pooling Options**:
- `MeanPooling`: Average over sequence
- `CLSPooling`: First token (CLS)
- `MaxPooling`: Max over sequence

**Classifier Types**:
- `LinearClassifierHead`: Simple linear layer
- `TransformerClassifierHead`: Transformer + linear (MetaTransformer-style)

---

## 🔌 Extending the Pipeline

### Adding a New Tokenizer

```python
# tokenization/my_tokenizer.py

from .base import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    def __init__(self, my_param, max_length=512):
        super().__init__(max_length)
        self.my_param = my_param
        
    def tokenize(self, sequence: str) -> List[str]:
        # Your tokenization logic
        return tokens
        
    def encode(self, sequence: str) -> List[int]:
        tokens = self.tokenize(sequence)
        # Convert to IDs
        return token_ids
        
    def get_vocab_size(self) -> int:
        return len(self.vocab)
```

Then add to `tokenization/__init__.py`:
```python
from .my_tokenizer import MyTokenizer
__all__ = [..., 'MyTokenizer']
```

---

### Adding a New Encoder

```python
# embedding/my_encoder.py

from .base import BaseEncoder
import torch.nn as nn

class MyEncoder(BaseEncoder):
    def __init__(self, model_path, freeze=False):
        # Load your model
        model = load_my_model(model_path)
        hidden_size = model.config.hidden_size
        
        super().__init__(model_path, hidden_size, freeze)
        self.encoder = model
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        return outputs.last_hidden_state
        
    def get_embedding_dim(self) -> int:
        return self.hidden_size
```

---

### Adding a New Classifier Head

```python
# model/head.py

class MyClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        # Your classifier architecture
        
    def forward(self, pooled_output):
        # Classification logic
        return logits
```

---

## 📊 Configuration Schema

```yaml
# Full configuration schema

tokenizer:
  type: str              # bpe, kmer, evo2, my_tokenizer
  path: str              # Model path or identifier
  max_length: int        # Max sequence length
  # Type-specific params
  k: int                 # For kmer
  overlap: bool          # For kmer
  use_hf: bool           # For bpe

encoder:
  type: str              # metagene, dnabert, evo2, my_encoder
  path: str              # Model path
  freeze: bool           # Freeze weights
  lora:                  # LoRA config (optional)
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: list
    bias: str

model:
  pooling: str           # mean, cls, max
  classifier_type: str   # linear, transformer
  classifier_config:     # Classifier-specific config
    dropout: float
    num_layers: int      # For transformer
    num_heads: int       # For transformer

training:               # Training config (optional)
  batch_size: int
  max_epochs: int
  lr: float

prediction:             # Prediction config (optional)
  batch_size: int
  confidence_threshold: float
```

---

## 🎯 Design Patterns

### 1. Strategy Pattern

Different tokenizers/encoders implement the same interface:

```python
# Client code doesn't know which tokenizer is used
tokenizer = create_tokenizer(config['tokenizer'])
tokens = tokenizer.encode(sequence)
```

### 2. Factory Pattern

Creating components from config:

```python
def create_tokenizer(config):
    tokenizer_type = config['type']
    if tokenizer_type == 'bpe':
        return BPETokenizer(...)
    elif tokenizer_type == 'kmer':
        return KmerTokenizer(...)
    # etc.
```

### 3. Composition Pattern

Model is composed of pluggable components:

```python
model = TaxonomicClassifier(
    encoder=encoder,      # Pluggable
    pooling=pooling,      # Pluggable
    classifier=head       # Pluggable
)
```

---

## 🔬 Testing Strategy

```
tests/
├── test_tokenization/
│   ├── test_bpe.py
│   ├── test_kmer.py
│   └── test_evo2.py
│
├── test_embedding/
│   ├── test_metagene.py
│   ├── test_dnabert.py
│   └── test_base.py
│
├── test_model/
│   ├── test_pooling.py
│   ├── test_heads.py
│   └── test_classifier.py
│
└── test_integration/
    ├── test_end_to_end.py
    └── test_prediction.py
```

---

## 📈 Performance Considerations

1. **Memory**: Use gradient checkpointing for large models
2. **Speed**: Batch processing, mixed precision (bf16/fp16)
3. **Scalability**: DataLoader with multiple workers
4. **Inference**: Model quantization, ONNX export

---

## 🚀 Future Enhancements

1. **Multi-GPU**: DistributedDataParallel support
2. **Streaming**: Process large FASTQ files without loading all into memory
3. **Caching**: Cache encoder outputs for faster re-training
4. **Quantization**: INT8 quantization for faster inference
5. **Multi-task**: Predict multiple taxonomic levels simultaneously
6. **Hybrid models**: Combine multiple encoders

---

## 📚 References

- **METAGENE-1**: https://arxiv.org/abs/2410.03461
- **DNABERT-2**: https://arxiv.org/abs/2306.15006
- **Evo**: https://arxiv.org/abs/2403.11389
- **LoRA**: https://arxiv.org/abs/2106.09685

---

This architecture provides a solid foundation for taxonomic classification while remaining flexible and extensible for future research directions.


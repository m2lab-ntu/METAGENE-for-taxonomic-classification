# METAGENE DNA/RNA Sequence Classification Pipeline

Production-ready pipeline for DNA/RNA sequence classification using METAGENE-1 encoder with mean-pooled embeddings, linear classifier, and LoRA fine-tuning.

## 🆕 **NEW: MetaClassifier - Modular Pipeline**

**A new modular, extensible architecture is now available!**

👉 **See [`metaclassifier/`](metaclassifier/)** for:
- **Pluggable tokenizers**: BPE, K-mer, single-nucleotide
- **Multiple encoders**: METAGENE-1, **Evo2**, DNABERT
- **Flexible classifiers**: Linear, Transformer heads
- **Sample aggregation**: Per-read → abundance estimation

📚 **Documentation**:
- [MetaClassifier README](metaclassifier/README.md)
- [Using Evo2](metaclassifier/USING_EVO2.md)
- [Migration Guide](metaclassifier/MIGRATION_GUIDE.md)

---

## 📊 **Benchmarks (Zymo Mock Community)**

We extensively benchmarked foundation models on the standard Zymo dataset to evaluate classification accuracy:

| Model | Parameters | Training | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **METAGENE-1** | 6.8B | 10 Epochs | **86.50%** | ✅ Baseline (Best) |
| **METAGENE-1** | 6.8B | 100 Epochs | *Run in progress* | 🔄 Epoch 53: **~81.4%** (Plateaued/Overfitting) |
| **GENERanno** | 0.5B | 100 Epochs | **77.00%** | ✅ Efficient Alternative |
| **Evo2** | 1B | 10 Epochs | 4.55% | ❌ Failed to converge |

### Training Comparison
![Model Comparison](outputs/model_comparison.png)
*Figure: Training trajectories of METAGENE-1, GENERanno, and Evo2.*

### How to Run Benchmarks
```bash
# Run GENERanno + Evo2 (Baseline)
./run_zymo_benchmark.sh

# Run GENERanno (100 Epochs)
./run_zymo_generanno_100e.sh

# Run METAGENE-1 (100 Epochs, RTX 4090 Optimized)
./run_zymo_metagene1_100e.sh
```

---

## ✅ **VERIFIED: Works on RTX 4090 (24GB)!**

**🎉 Successfully trained METAGENE-1 (7B) on single RTX 4090 with 13GB peak memory usage.**

## Overview

This pipeline provides end-to-end training, evaluation, and inference for per-read DNA/RNA sequence classification with optional per-sample aggregation.

**Choose Your Pipeline:**
- **Original Pipeline** (below): Battle-tested, production-ready METAGENE-1 classifier
- **[MetaClassifier](metaclassifier/)**: Modular, supports multiple models (METAGENE-1, Evo2, DNABERT)

**Key Features:**
- ✅ **RTX 4090 optimized** - Successfully trains 7B model on 24GB GPU
- 🚀 METAGENE-1 7B parameter encoder with BPE tokenization
- 🔧 HuggingFace tokenizer integration (official + minbpe support)
- 💾 Gradient checkpointing (saves 50% activation memory)
- 🎯 LoRA fine-tuning with flexible configurations
- ⚡ Mixed precision training (bf16/fp16)
- 📊 Comprehensive metrics (accuracy, F1, MCC, AUROC, confusion matrix)
- 📁 Streaming FASTA/FASTQ support with gzip
- 🔄 Batch and per-sample inference modes
- 📈 Early stopping on macro-F1

## Installation

### Prerequisites
- Conda environment named `METAGENE`
- NVIDIA GPU with CUDA support (tested on RTX 4090 24GB)
- Python 3.10+

### Setup

1. **Activate the METAGENE conda environment:**
```bash
conda activate METAGENE
```

2. **Run the installation script:**
```bash
cd /media/user/disk2/METAGENE/classification
chmod +x install.sh
./install.sh
```

The script will:
- Verify you're in the `METAGENE` conda environment
- Detect your CUDA version and install PyTorch accordingly
- Install all required dependencies (transformers, PEFT, biopython, etc.)
- Install the minbpe tokenizer from the pretrain repo

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### **NEW: RTX 4090 Optimized Training** (Recommended)

For **RTX 4090 (24GB)** or similar GPUs, use the optimized configuration:

```bash
# Setup environment
source setup_env.sh

# Train with optimized config (works on 24GB GPU!)
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads/train_reads.fa \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads/val_reads.fa \
  --mapping_tsv /media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab \
  --output_dir outputs/species_classification \
  --max_epochs 10
```

**Memory usage**: ~13GB / 24GB ✅  
**Optimizations**: Gradient checkpointing, reduced sequence length (128), LoRA rank=4

👉 **See [QUICK_START_RTX4090.md](QUICK_START_RTX4090.md) for detailed guide**

### Standard Training (40GB+ GPU)

For larger GPUs (A100, A6000):

```bash
python train.py \
  --config configs/default.yaml \
  --train_fasta YOUR_TRAIN.fa \
  --val_fasta YOUR_VAL.fa \
  --mapping_tsv YOUR_MAPPING.tsv \
  --output_dir outputs/exp1 \
  --batch_size 16 \
  --max_epochs 10
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
  --ckpt outputs/exp1/checkpoints/best.pt \
  --split val \
  --output_dir outputs/exp1/eval_results
```

### Prediction

Per-read inference:

```bash
python predict.py \
  --input /path/to/test_reads.fa.gz \
  --ckpt outputs/exp1/checkpoints/best.pt \
  --output predictions.csv \
  --batch_size 256
```

Per-sample inference with aggregation:

```bash
python predict.py \
  --input /path/to/samples.csv \
  --per_sample \
  --aggregation mean \
  --ckpt outputs/exp1/checkpoints/best.pt \
  --output sample_predictions.csv
```

## Configuration

### Available Configurations

| Config File | GPU | Memory | Batch Size | Use Case |
|------------|-----|--------|-----------|----------|
| **`rtx4090_optimized.yaml`** ⭐ | 24GB | 13GB | 1 (grad_accum=8) | **RTX 4090 / Similar** |
| `default.yaml` | 40GB+ | 25GB+ | 16 | A100 / A6000 |
| `default_hf_tokenizer.yaml` | 40GB+ | 25GB+ | 16 | Using HF tokenizer |

### RTX 4090 Optimized Config

**Key optimizations in `configs/rtx4090_optimized.yaml`:**

```yaml
tokenizer:
  max_length: 128  # Reduced from 512 (saves 60% memory)
  use_hf_tokenizer: true  # Official HuggingFace tokenizer

model:
  gradient_checkpointing: true  # Saves 50% activation memory ⭐
  lora:
    r: 4  # Reduced rank (saves memory)
    alpha: 8
    target_modules: [q_proj, v_proj]  # Only essential modules

training:
  batch_size: 1
  grad_accum_steps: 8  # Effective batch size = 8
  precision: bf16-mixed

memory_optimization:
  empty_cache_steps: 10  # Clear cache periodically
```

### Standard Config Parameters

**Model:**
- `encoder_path`: HuggingFace model path (default: `metagene-ai/METAGENE-1`)
- `pooling`: Pooling method (default: `mean`)
- `gradient_checkpointing`: Enable to save memory (default: `false`, **`true` for RTX 4090**)
- `lora.r`: LoRA rank (default: 8, **4 for RTX 4090**)
- `lora.alpha`: LoRA alpha (default: 16, **8 for RTX 4090**)
- `lora.target_modules`: Modules to apply LoRA (default: all, **[q_proj, v_proj] for RTX 4090**)

**Tokenizer:**
- `name_or_path`: Tokenizer path or HuggingFace model name
- `use_hf_tokenizer`: Use HuggingFace AutoTokenizer (recommended: `true`)
- `max_length`: Maximum sequence length (default: 512, **128 for RTX 4090**)

**Training:**
- `batch_size`: Batch size (default: varies by GPU)
- `grad_accum_steps`: Gradient accumulation steps (default: 1, **8 for RTX 4090**)
- `max_epochs`: Maximum training epochs (default: 10)
- `precision`: Mixed precision (default: `bf16-mixed`)
- `early_stopping.patience`: Early stopping patience (default: 3)
- `early_stopping.metric`: Metric to monitor (default: `macro_f1`)

**Optimizer:**
- `lr`: Learning rate (default: 2e-4)
- `weight_decay`: Weight decay (default: 0.01)

**Scheduler:**
- `warmup_steps`: Warmup steps (default: 100, **50 for RTX 4090**)

## Data Format

### FASTA/FASTQ Headers

Expected format:
```
>lbl|<class_id>|<tax_id>|<readlen>|<name>/mate
```

Example:
```
>lbl|724|28129|120|Prevotella-24562/1
```

Fields:
- `class_id`: Class identifier (used for training labels)
- `tax_id`: Taxonomic ID (optional)
- `readlen`: Read length (optional)
- `name`: Sequence name
- `mate`: Mate pair indicator (optional)

### Mapping TSV

Required columns:
- `class_id`: Integer class identifier
- `label_name`: Human-readable class name

Optional columns:
- `tax_id`: Taxonomic ID
- `group`: Group/family information
- `notes`: Additional notes

Example:
```tsv
class_id	label_name	tax_id
724	Prevotella	28129
823	Bacteroides	816
```

## Output Files

### Training Outputs

```
outputs/exp1/
├── checkpoints/
│   ├── best.pt                    # Best model by macro-F1
│   └── checkpoint_epoch_N.pt      # Epoch checkpoints
├── final_model/
│   ├── model.safetensors          # Final model weights
│   ├── config.json                # Model configuration
│   ├── label2id.json              # Label mappings
│   ├── id2label.json              # Reverse mappings
│   └── seen_classes.txt           # Class IDs in training data
├── plots/
│   └── training_curves.png        # Loss and metrics over time
├── config.json                    # Training configuration
├── final_metrics.json             # Final metrics summary
├── train_class_distribution.csv   # Training class counts
└── val_class_distribution.csv     # Validation class counts
```

### Evaluation Outputs

```
eval_results/
├── val_metrics.json               # Metrics summary
├── val_classification_report.json # Detailed per-class report
├── val_confusion_matrix.png       # Confusion matrix heatmap
├── val_per_class_metrics.csv      # Per-class precision/recall/F1
└── val_predictions.csv            # Predictions with probabilities
```

### Prediction Outputs

Per-read predictions CSV:
```csv
sequence_id,predicted_class,confidence,sequence_length,prob_SpeciesA,prob_SpeciesB,...
read001,SpeciesA,0.95,120,0.95,0.03,...
```

Per-sample predictions CSV:
```csv
sample_id,num_sequences,predicted_class,confidence,prob_SpeciesA,prob_SpeciesB,...
sample001,45,SpeciesA,0.92,0.92,0.05,...
```

## Model Architecture

```
Input: DNA/RNA Sequence
    ↓
METAGENE BPE Tokenizer (vocab ~1k)
    ↓
Token IDs [batch_size, 512]
    ↓
METAGENE-1 Encoder (7B params)
  - LoRA adapters on Q,K,V,O projections
  - Frozen base weights
    ↓
Hidden States [batch_size, 512, hidden_dim]
    ↓
Mean Pooling (attention-mask weighted)
    ↓
Pooled Embedding [batch_size, hidden_dim]
    ↓
Dropout (p=0.1)
    ↓
Linear Classifier [hidden_dim → num_classes]
    ↓
Logits → Softmax → Predictions
```

## Metrics

**Primary Metrics:**
- **Macro-F1**: Average F1 across all classes (primary for early stopping)
- **Micro-F1**: Global F1 score
- **Accuracy**: Overall classification accuracy
- **MCC**: Matthews Correlation Coefficient

**Additional:**
- **AUROC**: Area under ROC curve (computed if ≤10 classes)
- **Per-class Precision/Recall/F1**
- **Confusion Matrix**: Normalized heatmap

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory errors during training

**Solutions for RTX 4090 (24GB):**

1. **Use the optimized config** (recommended):
   ```bash
   python train.py --config configs/rtx4090_optimized.yaml ...
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   model:
     gradient_checkpointing: true  # Saves ~50% activation memory
   ```

3. **Reduce sequence length**:
   ```yaml
   tokenizer:
     max_length: 128  # Or even 64 for ultra-safe mode
   ```

4. **Reduce LoRA rank**:
   ```yaml
   model:
     lora:
       r: 4  # Or even 2
       target_modules: [q_proj, v_proj]  # Fewer modules
   ```

5. **Use gradient accumulation**:
   ```yaml
   training:
     batch_size: 1
     grad_accum_steps: 8  # Effective batch size = 8
   ```

6. **Set memory optimization flags**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
   ```

**If still OOM after all optimizations**, you may need:
- Cloud GPU with 40GB+ (A100, A6000)
- Multiple GPUs with distributed training
- 4-bit quantization (QLoRA) - requires code modification

### Tokenizer Not Found

**Symptoms:** `FileNotFoundError: tokenizer not found`

**Solution:** Ensure the minbpe tokenizer is installed:
```bash
cd /media/user/disk2/METAGENE/metagene-pretrain/train/minbpe
pip install -e .
```

### CUDA Version Mismatch

**Symptoms:** PyTorch CUDA version doesn't match system CUDA

**Solution:** Reinstall PyTorch with correct CUDA version:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Header Parsing Errors

**Symptoms:** `Warning: Could not parse header: ...`

**Solution:** Verify your header regex matches your data format. The default regex expects:
```
>lbl|class_id|tax_id|readlen|name/mate
```

Customize in config:
```yaml
dataset:
  header_regex: "your_custom_regex_here"
```

### Class Not in Mapping

**Symptoms:** `ValueError: Class ID X not found in mapping`

**Solution:** Either:
1. Add missing classes to mapping TSV
2. Set `strict_classes: false` to skip unknown classes

## Performance Tips

### Speed Optimization

1. **Enable torch.compile** (PyTorch 2.0+):
   ```yaml
   training:
     torch_compile: true
   ```

2. **Increase DataLoader workers**:
   ```python
   num_workers: 8  # Adjust based on CPU cores
   ```

3. **Use larger batch sizes** (if GPU memory allows):
   ```bash
   --batch_size 256
   ```

### Memory Optimization

1. **Mixed precision training** (default):
   ```yaml
   training:
     precision: bf16-mixed
   ```

2. **Gradient checkpointing** (not yet implemented, but can be added)

3. **Smaller LoRA rank**:
   ```yaml
   model:
     lora:
       r: 4  # Default is 8
   ```

## Testing

### Quick Test (3 minutes)

Test the complete pipeline with synthetic data:

```bash
cd /media/user/disk2/METAGENE/classification
bash test_optimized_training.sh
```

This will:
- Download METAGENE-1 model (if not cached)
- Train on 9 synthetic samples
- Validate training works without OOM
- Save model and metrics

### Unit Tests

Run smoke tests:

```bash
pytest tests/test_pipeline.py -v
```

### Data Loading Test (no GPU needed)

Test data loading without model:

```bash
python test_dataloader_only.py
```

## Hardware Requirements

### ✅ Verified Configurations

| GPU | VRAM | Config | Batch Size | Memory Usage | Status |
|-----|------|--------|-----------|--------------|--------|
| **RTX 4090** | 24GB | `rtx4090_optimized.yaml` | 1 (grad_accum=8) | **13GB** | ✅ **Verified** |
| A100 | 40GB | `default.yaml` | 16 | ~25GB | ✅ Recommended |
| A100 | 80GB | `default.yaml` | 32+ | ~30GB | ✅ Optimal |
| RTX 4090 | 24GB | `default.yaml` | N/A | N/A | ❌ OOM |

### Minimum Requirements (RTX 4090 Optimized)

- **GPU**: 24GB VRAM (RTX 4090, RTX 4000 Ada, etc.)
- **RAM**: 32GB system memory
- **Storage**: 50GB (20GB for model, 30GB for data/outputs)
- **CUDA**: 11.8+ or 12.1+

### Recommended Setup

- **GPU**: RTX 4090 (24GB) or A100 (40GB+)
- **RAM**: 64GB system memory  
- **Storage**: 100GB SSD (fast I/O)
- **CUDA**: 12.1+

### Tested Performance (RTX 4090)

**With `rtx4090_optimized.yaml`:**
- ✅ Peak memory: 13.0GB / 24GB
- ✅ Training speed: ~3.79 it/s
- ✅ Effective batch size: 8 (via gradient accumulation)
- ✅ Trainable parameters: 2.1M (0.03% of total)

**Expected training time:**
- 1K reads, 10 epochs: ~30 minutes
- 10K reads, 10 epochs: ~5 hours  
- 100K reads, 10 epochs: ~50 hours

### Cloud GPU Options

If you don't have local GPU:

| Provider | GPU | Cost | Best For |
|----------|-----|------|----------|
| Google Colab Pro+ | A100 40GB | $50/month | Quick experiments |
| Lambda Labs | A100 40GB | $1.10/hour | Long training |
| AWS EC2 | A100 80GB | Variable | Production |
| Vast.ai | A100 40GB | $0.80-1.50/hour | Budget training |

  ## 📚 Documentation

### Main Guides

- **[USER_GUIDE.md](USER_GUIDE.md)** - 📖 **Complete User Guide** (Quick Start + Training Your Dataset + Output Files)
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - 🔧 **Developer Guide** (Hyperparameters + Optimizations + Advanced Topics)

### Quick References

- **[HYPERPARAMETERS_QUICK_REF.txt](HYPERPARAMETERS_QUICK_REF.txt)** - ⚡ One-page hyperparameter reference
- **[OUTPUT_STRUCTURE_SUMMARY.txt](OUTPUT_STRUCTURE_SUMMARY.txt)** - 🎯 Visual output structure overview

### Utility Scripts

- **`setup_env.sh`** - Quick environment setup
- **`test_optimized_training.sh`** - Test optimized training
- **`test_dataloader_only.py`** - Test data loading without GPU
- **`monitor_training.sh`** - Monitor training progress
- **`pre_training_checklist.sh`** - Pre-training validation

### Configuration Files

- **`configs/rtx4090_optimized.yaml`** - Optimized for 24GB GPU ⭐
- **`configs/default.yaml`** - Standard config for 40GB+ GPU
- **`configs/default_hf_tokenizer.yaml`** - Using HuggingFace tokenizer

## Citation

If you use this pipeline in your research, please cite the METAGENE-1 paper:

```bibtex
@article{liu2025metagene,
  title={METAGENE-1: Metagenomic Foundation Model for Pandemic Monitoring},
  author={Liu, Ollie and Jaghouar, Sami and Hagemann, Johannes and Wang, Shangshang and Wiemels, Jason and Kaufman, Jeff and Neiswanger, Willie},
  journal={arXiv preprint arXiv:2501.02045},
  year={2025}
}
```

**Reference**: [METAGENE-1 on HuggingFace](https://huggingface.co/metagene-ai/METAGENE-1) | [Paper on arXiv](https://arxiv.org/abs/2501.02045)

## License

This pipeline is provided under the Apache 2.0 License. See LICENSE file for details.

The METAGENE-1 model weights are subject to their own license terms from the model repository.

## Support

For issues and questions:
- **GitHub Issues**: [metagene-ai/metagene-classification]
- **Documentation**: See files listed above
- **Examples**: See `examples/` directory
- **Model**: [HuggingFace Model Card](https://huggingface.co/metagene-ai/METAGENE-1)

## Updates & Changelog

### 2025-11-02: RTX 4090 Support ✅
- ✨ **Successfully trained on RTX 4090 (24GB)**
- 🔧 Added gradient checkpointing support
- 📝 Added `rtx4090_optimized.yaml` configuration
- 🚀 HuggingFace tokenizer integration
- 📖 Comprehensive documentation added
- ⚡ Memory optimization strategies
- 🧪 Complete testing suite

## Acknowledgments

- **METAGENE-1 model** by [metagene-ai](https://huggingface.co/metagene-ai)
- **PEFT library** by HuggingFace for LoRA implementation
- **Transformers** by HuggingFace
- **minbpe** tokenizer implementation
- **PyTorch** deep learning framework

---

**Status**: ✅ Production Ready | **GPU**: RTX 4090 Verified | **Memory**: 13GB/24GB | **Last Updated**: 2025-11-02


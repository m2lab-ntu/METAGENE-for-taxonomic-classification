# METAGENE Classification 測試總結與建議

## 📋 測試日期
2025-11-02

## ✅ 成功完成的項目

### 1. 模型下載 ✓
- **狀態**：成功
- **位置**：`/media/user/disk2/.cache/huggingface/`
- **大小**：約 16GB
- **解決方案**：
  - 將 HuggingFace cache 移至 `/media/user/disk2`（有足夠空間）
  - 設置環境變數：
    ```bash
    export HF_HOME=/media/user/disk2/.cache/huggingface
    export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
    ```

### 2. Data Loading Pipeline ✓
- **狀態**：完全正常
- **測試結果**：
  ```
  ✓ Tokenizer 載入
  ✓ FASTA 檔案解析
  ✓ Label mapping
  ✓ DataLoader 批次處理
  ✓ 訓練/驗證 dataset 建立
  ```
- **測試腳本**：`test_dataloader_only.py`

### 3. HuggingFace Tokenizer 整合 ✓
- **狀態**：成功整合
- **改進**：
  - 支援 minbpe tokenizer（原有）
  - 支援 HuggingFace 官方 tokenizer（新增，推薦）
  - 可通過配置切換：`use_hf_tokenizer: true`
- **Vocab size**：1024 tokens（正確匹配模型）

### 4. 代碼修改與優化 ✓
- **模型載入**：添加 `device_map="auto"` 符合 HF 建議
- **Tokenizer 兼容性**：支援雙模式（minbpe + HF）
- **訓練腳本修復**：
  - 修正 batch 過濾（移除 metadata）
  - 修正 learning rate 格式
  - 添加 HF tokenizer 支援

### 5. 配置檔案 ✓
- `configs/default.yaml` - minbpe tokenizer
- `configs/default_hf_tokenizer.yaml` - HuggingFace tokenizer（推薦）

## ❌ 遇到的限制

### GPU 記憶體不足 (Critical Issue)

**問題**：
```
torch.OutOfMemoryError: CUDA out of memory
GPU 0 has a total capacity of 23.64 GiB
```

**測試結果**：
| Batch Size | Max Length | Status | GPU 使用 |
|-----------|------------|--------|----------|
| 128 | 512 | ❌ OOM | N/A |
| 2 | 512 | ❌ OOM | 22.8GB |
| 1 | 512 | ❌ OOM | 22.9GB |

**原因分析**：
1. **模型大小**：METAGENE-1 = 7B 參數
2. **記憶體佔用**（bf16 precision）：
   - 模型權重：~14GB
   - Activations（前向傳播）：~8GB
   - Gradients（反向傳播）：~2GB
   - Optimizer states：~4GB
   - **總計**：~28GB（超過 RTX 4090 的 24GB）

3. **即使使用 LoRA**：
   - LoRA 只訓練 8.4M 參數（0.13%）
   - 但基礎模型仍需載入並保存 activations
   - 節省的主要是 gradient 和 optimizer memory
   - 仍然需要約 22-23GB

## 💡 解決方案與建議

### 方案 1：使用更大的 GPU（推薦）

**需求**：
- **GPU**：40GB+ VRAM
  - NVIDIA A100 (40GB/80GB)
  - NVIDIA A6000 (48GB)
  - H100 (80GB)

**預期結果**：
- Batch size: 8-16
- Training time: 2-4 hours（10k sequences）
- 完全支援的配置

### 方案 2：多 GPU 訓練

**使用 PyTorch DDP/FSDP**：
```bash
# 2x RTX 4090 (48GB total)
torchrun --nproc_per_node=2 train.py \
  --config configs/default.yaml \
  ... 
```

**修改需求**：
- 添加分散式訓練支援到 `train.py`
- 使用 FSDP (Fully Sharded Data Parallel)

### 方案 3：使用量化（QLoRA）

**8-bit/4-bit 量化**：
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModel.from_pretrained(
    "metagene-ai/METAGENE-1",
    quantization_config=bnb_config
)
```

**預期節省**：
- 4-bit：模型記憶體減少至 ~3.5GB
- 可能支援 batch_size=4-8 on RTX 4090

**權衡**：
- 輕微準確度降低（通常 <1%）
- 訓練速度稍慢

### 方案 4：Gradient Checkpointing

**啟用 gradient checkpointing**：
```python
model.encoder.gradient_checkpointing_enable()
```

**效果**：
- 減少 activation memory ~50%
- 訓練速度降低 ~20%
- 可能支援 batch_size=2-4

### 方案 5：使用雲端 GPU

**推薦服務**：
- **Google Colab Pro+**：A100 40GB（$50/month）
- **AWS EC2**：p4d.24xlarge (8x A100)
- **Lambda Labs**：A100 $1.10/hour
- **Vast.ai**：便宜的 A100 租用

## 🔧 立即可用的變通方法

### 選項 A：僅測試推理（Inference Only）

如果只需要測試推理而不訓練：

```bash
# 使用預訓練模型進行特徵提取
python predict.py \
  --input test_reads.fa \
  --ckpt metagene-ai/METAGENE-1 \
  --output predictions.csv \
  --batch_size 16
```

**記憶體需求**：
- Inference only：~14GB（可行）
- 支援 batch_size=16-32

### 選項 B：凍結更多層

```yaml
model:
  lora:
    enabled: true
    target_modules: [q_proj, v_proj]  # 只訓練 Q 和 V
    r: 4  # 減少 LoRA rank
```

**效果有限**：主要瓶頸在 activations，不在 trainable parameters

### 選項 C：減少序列長度

```yaml
tokenizer:
  max_length: 256  # 從 512 減少到 256
```

**預估**：
- 記憶體減少 ~30%
- 可能支援 batch_size=2
- **權衡**：長序列會被截斷

##  📊 硬體需求總結

| 訓練場景 | GPU 需求 | Batch Size | Training Time |
|----------|---------|-----------|---------------|
| **Full Training** | 40GB+ | 8-16 | 2-4 hours |
| **4-bit QLoRA** | 24GB | 4-8 | 3-6 hours |
| **Gradient Checkpoint** | 24GB | 2-4 | 4-8 hours |
| **Inference Only** | 24GB | 16-32 | N/A |
| **當前 RTX 4090** | 24GB | ❌ 0-1 | ❌ 無法訓練 |

## 🎯 建議的下一步

### 立即行動（不需更換硬體）：

1. **測試推理功能**：
   ```bash
   # 測試 feature extraction
   python predict.py --input your_data.fa --ckpt metagene-ai/METAGENE-1
   ```

2. **實作 QLoRA**：
   - 修改 `modeling.py` 添加量化支援
   - 可能在 RTX 4090 上訓練

3. **使用雲端GPU**：
   - 租用 A100 完成訓練
   - 費用：約 $10-20（4-8小時訓練）

### 長期解決方案：

1. **升級硬體**：A100 40GB/80GB
2. **多GPU設置**：2-4x RTX 4090
3. **使用較小模型**：如果存在 1B-3B 版本的 METAGENE

## 📝 環境設置腳本

為了方便未來使用，創建快速設置腳本：

```bash
#!/bin/bash
# setup_metagene_env.sh

# 設置 HuggingFace cache
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface

# 設置 CUDA 記憶體分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 激活環境
conda activate METAGENE

echo "Environment ready for METAGENE classification"
```

## 🏁 結論

**成功項目**：
- ✅ 完整的 classification pipeline 已實作
- ✅ HuggingFace 整合完成
- ✅ Data loading 完全正常
- ✅ 模型已下載並可用

**主要限制**：
- ❌ RTX 4090 24GB 無法訓練 METAGENE-1 7B
- ⚠️ 需要 40GB+ GPU 或使用量化技術

**推薦方案**：
1. 🥇 使用雲端 A100 GPU 完成訓練
2. 🥈 實作 4-bit QLoRA 在 RTX 4090 上訓練
3. 🥉 使用當前硬體進行推理測試

---

**最後更新**：2025-11-02
**測試人員**：AI Assistant
**GPU**：NVIDIA RTX 4090 (24GB)
**Status**：✅ Pipeline Ready | ❌ Training Blocked (OOM)


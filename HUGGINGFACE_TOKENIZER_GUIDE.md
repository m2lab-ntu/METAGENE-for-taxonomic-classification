# HuggingFace 官方 Tokenizer 使用指南

## 📝 總覽

已更新 METAGENE classification pipeline 以支援 HuggingFace 官方 tokenizer，與 [METAGENE-1 HuggingFace 頁面](https://huggingface.co/metagene-ai/METAGENE-1) 建議的用法一致。

## 🔄 修改內容

### 1. 模型載入 (`modules/modeling.py`)
- ✅ **保持使用 `AutoModel`** - 這是正確的，因為我們做的是 classification 而不是 generation
- ✅ 添加 `device_map="auto"` 參數，符合 HuggingFace 官方範例
- ✅ 使用 `torch.bfloat16` 以獲得更好的性能

### 2. Tokenizer 支援 (`modules/dataloading.py`)
更新 `MetaGeneTokenizer` 類別以支援兩種模式：

#### **選項 A：minbpe tokenizer（原有，預設）**
```python
tokenizer = MetaGeneTokenizer(
    tokenizer_path="/path/to/minbpe/tokenizer.model",
    max_length=512,
    use_hf_tokenizer=False  # 預設
)
```

#### **選項 B：HuggingFace 官方 tokenizer（新增，推薦）**
```python
tokenizer = MetaGeneTokenizer(
    tokenizer_path="metagene-ai/METAGENE-1",
    max_length=512,
    use_hf_tokenizer=True  # 啟用 HuggingFace tokenizer
)
```

### 3. 配置檔案
提供兩個配置檔案：

**`configs/default.yaml`** - 使用 minbpe tokenizer（原有）
```yaml
tokenizer:
  name_or_path: /media/user/disk2/METAGENE/metagene-pretrain/train/minbpe/tokenizer/large-mgfm-1024.model
  use_hf_tokenizer: false
```

**`configs/default_hf_tokenizer.yaml`** - 使用 HuggingFace tokenizer（新增）
```yaml
tokenizer:
  name_or_path: metagene-ai/METAGENE-1
  use_hf_tokenizer: true
```

## 🚀 使用方法

### 方法 1：使用 minbpe tokenizer（原有方法）

```bash
python train.py \
  --config configs/default.yaml \
  --train_fasta examples/example_train.fa \
  --val_fasta examples/example_val.fa \
  --mapping_tsv examples/labels.tsv \
  --output_dir outputs/test_minbpe \
  --batch_size 4 \
  --max_epochs 2
```

### 方法 2：使用 HuggingFace tokenizer（推薦）

```bash
python train.py \
  --config configs/default_hf_tokenizer.yaml \
  --train_fasta examples/example_train.fa \
  --val_fasta examples/example_val.fa \
  --mapping_tsv examples/labels.tsv \
  --output_dir outputs/test_hf_tokenizer \
  --batch_size 4 \
  --max_epochs 2
```

### 方法 3：修改現有配置檔案

編輯 `configs/default.yaml`：

```yaml
tokenizer:
  # 改成 HuggingFace 模型名稱
  name_or_path: metagene-ai/METAGENE-1
  # 啟用 HuggingFace tokenizer
  use_hf_tokenizer: true
  max_length: 512
```

## 🎯 為什麼使用 `AutoModel` 而不是 `AutoModelForCausalLM`？

### Classification 任務（我們的情況）✅
```python
# 正確：用於 feature extraction + classification
encoder = AutoModel.from_pretrained("metagene-ai/METAGENE-1")
# 取得 hidden states → mean pooling → linear classifier
```

**為什麼：**
- 我們需要的是 encoder 的 hidden states
- 我們會加自己的 classification head
- 不需要 language model head（節省記憶體）

### Generation 任務（HuggingFace 範例）
```python
# 用於生成 DNA 序列
model = AutoModelForCausalLM.from_pretrained("metagene-ai/METAGENE-1")
# 直接生成下一個 token
```

**為什麼：**
- 需要 language model head 來預測下一個 token
- 用於序列生成任務

## 📊 比較

| 特性 | minbpe tokenizer | HuggingFace tokenizer |
|------|------------------|----------------------|
| 與官方範例一致 | ❌ | ✅ |
| 需要本地檔案 | ✅ 需要 | ❌ 自動下載 |
| 設定複雜度 | 中等 | 簡單 |
| 維護性 | 需手動更新 | 自動同步官方版本 |
| 推薦度 | 可用 | **推薦** |

## ⚠️ 注意事項

### 1. 首次使用 HuggingFace tokenizer 會下載模型檔案
```bash
# 約 200MB tokenizer 檔案
Attempting to load HuggingFace tokenizer from metagene-ai/METAGENE-1
Downloading tokenizer...
✓ Using HuggingFace official tokenizer
```

### 2. 模型下載問題
如果之前卡在模型下載（16GB），可以：

**選項 A：預先下載模型**
```bash
conda activate METAGENE
python -c "from transformers import AutoModel; AutoModel.from_pretrained('metagene-ai/METAGENE-1', torch_dtype='auto')"
```

**選項 B：使用已下載的模型**
檢查緩存：
```bash
ls ~/.cache/huggingface/hub/models--metagene-ai--METAGENE-1/
```

如果看到 `.incomplete` 檔案，表示下載未完成，需要重新下載。

## 🧪 測試

### 快速測試（無需 GPU 或模型）
```bash
cd /media/user/disk2/METAGENE/classification
conda activate METAGENE
python test_dataloader_only.py
```

這會測試：
- ✅ Tokenizer 載入（minbpe）
- ✅ FASTA 檔案解析
- ✅ Label mapping
- ✅ DataLoader 批次處理

### 完整測試（需要 GPU + 模型）
```bash
pytest tests/test_pipeline.py -v
```

## 📚 參考資料

- [METAGENE-1 HuggingFace 頁面](https://huggingface.co/metagene-ai/METAGENE-1)
- [METAGENE-1 論文](https://arxiv.org/abs/2501.02045)
- [HuggingFace Transformers 文檔](https://huggingface.co/docs/transformers)

## 🔧 故障排除

### 問題：無法載入 HuggingFace tokenizer
```
Warning: Could not load HuggingFace tokenizer
Falling back to minbpe tokenizer
```

**解決方法：**
1. 確保 `transformers` 已安裝：`pip install transformers`
2. 檢查網路連線
3. 使用 minbpe tokenizer（`use_hf_tokenizer: false`）

### 問題：模型下載卡住
```
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]
```

**解決方法：**
1. 檢查網路連線和速度
2. 使用代理或更換網路
3. 清理不完整的下載：
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--metagene-ai--METAGENE-1
   ```
4. 重新下載

### 問題：CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**解決方法：**
1. 減少 batch size：`--batch_size 32`
2. 減少序列長度：在 config 中設定 `max_length: 256`
3. 啟用梯度累積：
   ```yaml
   training:
     grad_accum_steps: 2
   ```

## ✨ 總結

所有修改都是**向後相容**的：
- ✅ 原有的 minbpe tokenizer 仍可正常使用
- ✅ 新增 HuggingFace tokenizer 支援（推薦）
- ✅ 可透過配置檔案輕鬆切換
- ✅ 代碼更符合 HuggingFace 官方最佳實踐



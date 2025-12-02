# 性能比較系統 (Performance Benchmark System)

完整的標準化評估框架，用於比較不同方法的性能

---

## 🎯 系統概覽

這個系統提供：

1. ✅ **獨立測試集創建** - 從 `full_labeled_species_sequences` 創建不重疊的測試集
2. ✅ **標準化評估流程** - 統一的評估指標和方法
3. ✅ **多方法比較** - 輕鬆比較不同 tokenizer、encoder、classifier
4. ✅ **詳細報告生成** - Markdown 和 JSON 格式的詳細報告
5. ✅ **快速啟動腳本** - 一鍵運行完整評估流程

---

## 📦 文件結構

```
classification/
├── create_test_dataset.py       # 創建測試數據集
├── benchmark_framework.py       # 性能評估框架
├── quick_benchmark.sh          # 快速啟動腳本 ⭐
├── 性能比較流程.md              # 詳細流程文檔
└── metaclassifier/
    └── configs/
        ├── metagene_bpe.yaml   # METAGENE + BPE
        ├── metagene_kmer.yaml  # METAGENE + K-mer (新)
        ├── dnabert_kmer.yaml   # DNABERT + K-mer
        └── evo2_nucleotide.yaml # Evo2 + Single-nucleotide
```

---

## 🚀 快速開始

### **選項 1: 使用快速腳本（推薦）** ⭐

```bash
# 小型測試（快速驗證）
./quick_benchmark.sh small

# 中型測試（推薦）
./quick_benchmark.sh medium

# 完整測試（全面評估）
./quick_benchmark.sh full
```

**這個腳本會自動：**
1. ✅ 創建測試數據集
2. ✅ 檢查可用模型
3. ✅ 生成方法配置
4. ✅ 運行性能評估
5. ✅ 顯示結果摘要

---

### **選項 2: 手動步驟（完全控制）**

#### **Step 1: 創建測試集**

```bash
python create_test_dataset.py \
  --source_dir /media/user/disk2/full_labeled_species_sequences \
  --output test_data/test_benchmark.fa \
  --train_dir /media/user/disk2/full_labeled_species_train_reads_shuffled \
  --val_dir /media/user/disk2/full_labeled_species_val_reads_shuffled \
  --reads_per_species 100 \
  --max_species 500 \
  --seed 42
```

#### **Step 2: 訓練要比較的方法**

```bash
# 方法 1: METAGENE + BPE (現有系統)
python train.py \
  --train_set_path /media/user/disk2/full_labeled_species_train_reads_shuffled \
  --validation_set_path /media/user/disk2/full_labeled_species_val_reads_shuffled \
  --mapping_df /media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab \
  --output_dir outputs/metagene_bpe_baseline

# 方法 2: METAGENE + K-mer (MetaClassifier)
python metaclassifier/train.py \
  --config metaclassifier/configs/metagene_kmer.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled \
  --mapping_tsv /media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab \
  --output_dir outputs/metagene_kmer_experiment

# 方法 3: DNABERT + K-mer (MetaClassifier)
python metaclassifier/train.py \
  --config metaclassifier/configs/dnabert_kmer.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled \
  --mapping_tsv /media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab \
  --output_dir outputs/dnabert_kmer_experiment
```

#### **Step 3: 創建方法配置 JSON**

創建 `methods_config.json`：

```json
[
  {
    "name": "METAGENE_BPE_Baseline",
    "checkpoint": "outputs/metagene_bpe_baseline/checkpoints/best.pt",
    "config": null
  },
  {
    "name": "METAGENE_KMER",
    "checkpoint": "outputs/metagene_kmer_experiment/checkpoints/best.pt",
    "config": "metaclassifier/configs/metagene_kmer.yaml"
  },
  {
    "name": "DNABERT_KMER",
    "checkpoint": "outputs/dnabert_kmer_experiment/checkpoints/best.pt",
    "config": "metaclassifier/configs/dnabert_kmer.yaml"
  }
]
```

#### **Step 4: 運行評估**

```bash
python benchmark_framework.py \
  --test_data test_data/test_benchmark.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir benchmark_results \
  --methods methods_config.json
```

#### **Step 5: 查看結果**

```bash
# 比較表格
cat benchmark_results/benchmark_comparison.csv

# 詳細報告
cat benchmark_results/benchmark_report_*.md

# JSON 格式（詳細）
cat benchmark_results/benchmark_report_*.json | jq '.'
```

---

## 📊 可以比較什麼？

### **1. 不同的 Tokenizer**

| Tokenizer | 配置文件 | 特點 |
|-----------|----------|------|
| BPE | `metagene_bpe.yaml` | 子詞單元，靈活 |
| K-mer | `metagene_kmer.yaml` | 固定長度，傳統方法 |
| Single-nucleotide | `evo2_nucleotide.yaml` | 單核苷酸，最細粒度 |

### **2. 不同的 Encoder**

| Encoder | 配置文件 | 參數量 |
|---------|----------|--------|
| METAGENE-1 | `metagene_*.yaml` | ~100M |
| DNABERT-2 | `dnabert_*.yaml` | ~117M |
| Evo2 | `evo2_*.yaml` | ~7B |

### **3. 不同的 Classifier**

| Classifier | 配置 | 特點 |
|------------|------|------|
| Linear | `classifier_type: linear` | 簡單、快速 |
| Transformer | `classifier_type: transformer` | 複雜、可能更準確 |

### **4. 不同的超參數**

- Learning rate: `0.0001`, `0.0002`, `0.0005`
- Max length: `128`, `192`, `256`, `512`
- LoRA rank: `4`, `8`, `16`
- Batch size: `1`, `2`, `4`

---

## 📈 輸出結果

### **benchmark_comparison.csv**

```csv
Method,Accuracy,Macro Accuracy,Weighted Accuracy,Avg Confidence,Num Classes,Total Samples
METAGENE_BPE_Baseline,0.8523,0.8234,0.8456,0.7845,500,50000
METAGENE_KMER,0.8312,0.8045,0.8234,0.7623,500,50000
DNABERT_KMER,0.8678,0.8456,0.8589,0.8012,500,50000
```

### **benchmark_report.md**

```markdown
# 性能比較報告

## 整體比較
| Method | Accuracy | Macro Accuracy | ... |
|--------|----------|----------------|-----|
| ...    | ...      | ...            | ... |

## 詳細指標
### METAGENE_BPE_Baseline
- 總樣本數: 50,000
- 正確預測數: 42,615
- 準確率: 0.8523
...

## 結論
最佳方法: DNABERT_KMER
- 準確率: 0.8678
...
```

---

## 🎯 實驗範例

### **實驗 1: Tokenizer 比較**

**目標**: 比較 BPE vs K-mer tokenizer

```bash
# 訓練 METAGENE + BPE
python train.py --output_dir outputs/exp1_bpe

# 訓練 METAGENE + K-mer  
python metaclassifier/train.py \
  --config metaclassifier/configs/metagene_kmer.yaml \
  --output_dir outputs/exp1_kmer

# 運行 benchmark
./quick_benchmark.sh medium
```

---

### **實驗 2: Encoder 比較**

**目標**: 比較 METAGENE-1 vs DNABERT-2

```bash
# METAGENE-1 (已有)
# ...

# DNABERT-2
python metaclassifier/train.py \
  --config metaclassifier/configs/dnabert_kmer.yaml \
  --output_dir outputs/exp2_dnabert

# 運行 benchmark
./quick_benchmark.sh medium
```

---

### **實驗 3: 超參數優化**

**目標**: 找到最佳 learning rate

```bash
# 創建不同 lr 的配置
for LR in 0.0001 0.0002 0.0005; do
  cp configs/fast_training.yaml configs/lr_${LR}.yaml
  sed -i "s/lr: .*/lr: $LR/" configs/lr_${LR}.yaml
  
  python train.py \
    --config configs/lr_${LR}.yaml \
    --output_dir outputs/exp3_lr_${LR}
done

# 運行 benchmark
./quick_benchmark.sh medium
```

---

## 📊 評估指標說明

| 指標 | 公式 | 說明 | 適用場景 |
|------|------|------|----------|
| **Accuracy** | `correct / total` | 總體準確率 | 平衡數據集 |
| **Macro Accuracy** | `mean(per_class_acc)` | 每類權重相同 | 不平衡數據集 |
| **Weighted Accuracy** | `sum(acc * count) / total` | 按樣本數加權 | 不平衡數據集 |
| **Avg Confidence** | `mean(confidence)` | 平均預測置信度 | 模型信心 |

**建議：**
- 平衡數據集：看 **Accuracy**
- 不平衡數據集：看 **Macro Accuracy**
- 生產環境：看 **Weighted Accuracy** + **Avg Confidence**

---

## ⚙️ 高級配置

### **自定義測試集大小**

```bash
python create_test_dataset.py \
  --source_dir /media/user/disk2/full_labeled_species_sequences \
  --output test_data/test_custom.fa \
  --reads_per_species 150 \
  --max_species 300 \
  --min_length 100
```

### **並行運行多個預測**

```bash
# 使用 GNU Parallel（如果有很多方法要測試）
parallel -j 4 \
  'python predict.py --ckpt outputs/{}/checkpoints/best.pt --input test_data/test.fa --output preds/{}.csv' \
  ::: method1 method2 method3 method4
```

### **自定義評估指標**

修改 `benchmark_framework.py` 添加更多指標：
- Precision / Recall / F1
- Per-class metrics
- Confusion matrix
- ROC curves

---

## 🔍 故障排除

### **問題 1: 測試集創建太慢**

**解決方案：**
- 減少 `--max_species`
- 減少 `--reads_per_species`
- 先不檢查重疊（移除 `--train_dir` 和 `--val_dir`）

### **問題 2: 預測失敗**

**檢查：**
1. 配置文件路徑是否正確
2. 檢查點文件是否存在
3. 測試數據格式是否正確

**調試：**
```bash
# 手動運行預測，查看錯誤信息
python predict.py \
  --ckpt outputs/method/checkpoints/best.pt \
  --input test_data/test.fa \
  --output test_pred.csv \
  --batch_size 32
```

### **問題 3: GPU 記憶體不足**

**解決方案：**
```bash
# 減小預測 batch size
python benchmark_framework.py \
  ... \
  --batch_size 64  # 預設是 256
```

---

## 📚 相關文檔

- **詳細流程**: `性能比較流程.md`
- **MetaClassifier 配置**: `metaclassifier/配置修改指南.md`
- **系統對比**: `metaclassifier/系統對比說明.md`
- **快速入門**: `metaclassifier/快速入門.md`

---

## 💡 最佳實踐

1. ✅ **先用小型測試集驗證流程**（5分鐘）
2. ✅ **確認測試集不重疊**（使用 `--train_dir` 和 `--val_dir`）
3. ✅ **記錄所有實驗配置**（使用有意義的輸出目錄名）
4. ✅ **定期備份結果**（benchmark_results/ 目錄）
5. ✅ **使用相同的隨機種子**（確保可重現性）

---

## 🎉 總結

### **快速評估（5分鐘）**
```bash
./quick_benchmark.sh small
```

### **標準評估（1小時）**
```bash
./quick_benchmark.sh medium
```

### **完整評估（4-6小時）**
```bash
./quick_benchmark.sh full
```

---

## 📞 需要幫助？

1. 查看詳細文檔: `性能比較流程.md`
2. 查看配置說明: `metaclassifier/配置修改指南.md`
3. 檢查腳本輸出的錯誤信息
4. 使用 `--help` 查看工具選項

---

**祝實驗順利！** 🚀


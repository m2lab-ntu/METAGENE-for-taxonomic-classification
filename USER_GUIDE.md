# METAGENE Classification User Guide
# 使用者完整指南

本指南包含所有使用者需要的資訊：快速開始、訓練自己的資料集、理解輸出檔案。

---

## 📖 目錄

1. [快速開始 - RTX 4090](#快速開始---rtx-4090)
2. [訓練您的資料集](#訓練您的資料集)
3. [輸出檔案詳解](#輸出檔案詳解)
4. [常見問題與故障排除](#常見問題與故障排除)

---

# 快速開始 - RTX 4090

## ✅ 已驗證可行！

RTX 4090 (24GB) 可以成功訓練 METAGENE-1 (7B) - **峰值記憶體使用：13GB**

---

## 🚀 三步驟開始訓練

### 步驟 1：設置環境

```bash
cd /media/user/disk2/METAGENE/classification
source setup_env.sh
```

### 步驟 2：測試（可選）

```bash
# 用範例資料測試 (~3分鐘)
bash test_optimized_training.sh
```

### 步驟 3：訓練你的資料

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta YOUR_TRAIN.fa \
  --val_fasta YOUR_VAL.fa \
  --mapping_tsv YOUR_MAPPING.tsv \
  --output_dir outputs/my_experiment \
  --max_epochs 10
```

---

## 📝 完整範例

```bash
# 設置環境
export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 激活環境
conda activate METAGENE

# 清空 GPU 記憶體
python -c "import torch; torch.cuda.empty_cache()"

# 開始訓練
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta data/train.fa \
  --val_fasta data/val.fa \
  --mapping_tsv data/mapping.tsv \
  --output_dir outputs/species_classification \
  --batch_size 1 \
  --max_epochs 10 \
  2>&1 | tee training.log
```

---

## 🎛️ RTX 4090 關鍵優化設置

| 參數 | 值 | 節省記憶體 |
|------|-----|----------|
| `max_length` | 128 | 60% |
| `batch_size` | 1 | 最小佔用 |
| `grad_accum_steps` | 8 | 保持有效 batch=8 |
| `lora.r` | 4 | 減少參數 |
| `lora.target_modules` | [q_proj, v_proj] | 只訓練關鍵模組 |
| `gradient_checkpointing` | true | **50% activation memory** |

**結果**: 記憶體使用 13GB / 24GB ✓

---

## 📊 預期結果

### 記憶體使用
- **峰值**: 13GB / 24GB ✓
- **平均**: ~13GB
- **安全邊界**: 11GB 剩餘

### 訓練速度
- **小資料集** (1K reads): ~30 分鐘/10 epochs
- **中資料集** (10K reads): ~5 小時/10 epochs
- **大資料集** (100K reads): ~50 小時/10 epochs

### 輸出檔案
```
outputs/YOUR_EXPERIMENT/
├── checkpoints/best.pt          # 最佳模型
├── final_model/                 # 用於推理
├── plots/training_curves.png    # 訓練曲線
└── final_metrics.json           # 最終指標
```

---

## 🔧 故障排除

### 問題 1：還是 OOM？

**解決方案 A**: 減少序列長度
```bash
python train.py --config configs/rtx4090_optimized.yaml --max_length 64 ...
```

**解決方案 B**: Ultra-safe 模式
```yaml
tokenizer:
  max_length: 64
model:
  lora:
    r: 2
    target_modules: [q_proj]
```

### 問題 2：訓練太慢？

**解決方案**: 稍大的 batch size
```yaml
training:
  batch_size: 2
  grad_accum_steps: 4
tokenizer:
  max_length: 96
```

### 問題 3：準確度不夠？

**解決方案 A**: 增加訓練時間
```bash
python train.py --config configs/rtx4090_optimized.yaml --max_epochs 20 ...
```

**解決方案 B**: 更大的 LoRA rank
```yaml
model:
  lora:
    r: 8
    target_modules: [q_proj, k_proj, v_proj, o_proj]
```
**注意**: 這會增加記憶體使用到 ~16-18GB

---

## 📈 監控訓練

```bash
# 檢查 GPU 使用
watch -n 1 nvidia-smi

# 查看訓練 log
tail -f training.log

# 即時查看指標
cat outputs/YOUR_EXPERIMENT/final_metrics.json
```

---

# 訓練您的資料集

## 資料格式要求

### FASTA Header 格式
```
>lbl|{class_id}|{tax_id}|{readlen}|{species_name}/{mate}
```

範例：
```
>lbl|85|301|45|Pseudomonas-61537/2
CTTCACGGCTGCTCTGGAAACTTTCGGCCTGGGCGGCCAGTTGCGCTTTGAGGTTGGCGTTGAGCTC...
```

### Mapping 檔案格式 (TSV)
```tsv
class_id    label_name                      tax_id
0           Azorhizobium caulinodans        7
1           Buchnera aphidicola             9
2           Dictyoglomus thermophilum       14
```

---

## 訓練流程

### 1. 準備 Mapping 檔案

```bash
# 自動載入物種名稱
python prepare_species_mapping.py \
  --input_path your_mapping.tab \
  --output_path species_mapping_converted.tsv \
  --species_name_csv species_database.csv

# 或不需要物種名稱（使用 TaxID_* 作為標籤）
python prepare_species_mapping.py --no_label_name
```

### 2. 快速測試（建議先執行）

執行 1 個 epoch 來驗證資料格式：

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta data/train.fa \
  --val_fasta data/val.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir outputs/quick_test \
  --max_epochs 1
```

### 3. 完整訓練

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta data/train.fa \
  --val_fasta data/val.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir outputs/full_training \
  --max_epochs 10
```

---

## 訓練配置說明

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `batch_size` | 1 | 最小 batch size |
| `grad_accum_steps` | 8 | 有效 batch size = 8 |
| `max_length` | 128 | 序列長度 |
| `gradient_checkpointing` | true | 節省記憶體 |
| `lora.r` | 4 | LoRA rank |
| `lora.target_modules` | [q_proj, v_proj] | 訓練的層 |
| `precision` | bf16-mixed | 混合精度 |

---

## 調整訓練參數

### 方法 1：修改 config 檔案

編輯 `configs/rtx4090_optimized.yaml`：

```yaml
training:
  max_epochs: 20
  batch_size: 2
  
tokenizer:
  max_length: 256
```

### 方法 2：命令列參數

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta data/train.fa \
  --val_fasta data/val.fa \
  --mapping_tsv data/mapping.tsv \
  --output_dir outputs/custom_experiment \
  --batch_size 2 \
  --max_epochs 5
```

---

## 評估模型

```bash
python evaluate.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --split val \
  --output_dir outputs/evaluation_results
```

---

## 使用模型進行預測

```bash
python predict.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --input new_sequences.fa \
  --output predictions.csv
```

---

# 輸出檔案詳解

## 🗂️ 訓練輸出檔案 (outputs/my_experiment/)

### 最關鍵的 5 個檔案 ⭐

1. **`checkpoints/best.pt`** (13GB)
   - 訓練好的最佳模型
   - 用於預測和評估
   - **最重要的檔案，務必備份！**

2. **`final_model/id2label.json`** (~50KB)
   - ID→物種名稱映射
   - 將預測的數字轉換為物種名稱
   ```json
   {
     "0": "Escherichia coli",
     "1": "Staphylococcus aureus"
   }
   ```

3. **`config.json`** (~2KB)
   - 完整訓練配置
   - 可重現訓練結果的關鍵

4. **`plots/training_curves.png`** (~1MB)
   - 訓練/驗證 loss 和指標曲線
   - 判斷訓練是否過擬合

5. **`final_metrics.json`** (~5KB)
   - 最終驗證指標
   ```json
   {
     "accuracy": 0.8234,
     "macro_f1": 0.8122
   }
   ```

---

### 完整檔案列表

| 檔案 | 大小 | 重要性 | 用途 |
|------|------|--------|------|
| `config.json` | ~2KB | ⭐⭐⭐⭐⭐ | 完整訓練配置 |
| `training.log` | ~249MB | ⭐⭐⭐⭐ | 訓練日誌 |
| `checkpoints/best.pt` | ~13GB | ⭐⭐⭐⭐⭐ | 最佳模型權重 |
| `checkpoints/last.pt` | ~13GB | ⭐⭐⭐ | 最後 checkpoint |
| `final_model/label2id.json` | ~50KB | ⭐⭐⭐⭐⭐ | 物種名→ID |
| `final_model/id2label.json` | ~50KB | ⭐⭐⭐⭐⭐ | ID→物種名 |
| `final_model/seen_classes.txt` | ~10KB | ⭐⭐⭐ | 訓練過的類別 |
| `plots/training_curves.png` | ~1MB | ⭐⭐⭐⭐ | 訓練曲線圖 |
| `plots/confusion_matrix.png` | ~2MB | ⭐⭐⭐⭐ | 混淆矩陣 |
| `final_metrics.json` | ~5KB | ⭐⭐⭐⭐ | 最終指標 |
| `train_class_distribution.csv` | ~100KB | ⭐⭐⭐ | 訓練集類別分布 |
| `val_class_distribution.csv` | ~25KB | ⭐⭐⭐ | 驗證集類別分布 |

---

## 🧪 測試輸出檔案 (outputs/my_test/)

| 檔案 | 重要性 | 用途 |
|------|--------|------|
| `test_metrics.json` | ⭐⭐⭐⭐⭐ | 整體測試性能 |
| `test_predictions.csv` | ⭐⭐⭐⭐⭐ | 每個樣本的預測結果 |
| `test_classification_report.json` | ⭐⭐⭐⭐ | 每類別 Precision/Recall/F1 |
| `test_per_class_metrics.csv` | ⭐⭐⭐⭐ | CSV 格式（易分析） |
| `test_confusion_matrix.png` | ⭐⭐⭐⭐ | 混淆矩陣圖 |

---

### test_predictions.csv 格式

```csv
sequence_id,true_label,predicted_label,true_class_id,predicted_class_id,confidence,correct
seq_001,Escherichia coli,Escherichia coli,0,0,0.9823,True
seq_002,Staphylococcus aureus,Enterococcus faecalis,1,5,0.6234,False
```

**重要欄位**：
- `confidence < 0.7`：模型不確定的預測
- `correct = False`：需要重點分析的錯誤樣本

---

## 🔧 常用命令

```bash
# 監控訓練
tail -f outputs/my_experiment/training.log

# 查看最終性能
cat outputs/my_experiment/final_metrics.json | jq '.macro_f1'

# 評估測試集
python evaluate.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --split test \
  --output_dir outputs/my_test

# 預測新數據
python predict.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --input new_sequences.fa \
  --output predictions.csv

# 找出錯誤樣本
grep ",False$" outputs/my_test/test_predictions.csv | head -20

# 找出低置信度預測
awk -F',' '$6 < 0.7 {print}' outputs/my_test/test_predictions.csv

# 找出表現最差的類別
sort -t',' -k5 -n outputs/my_test/test_per_class_metrics.csv | head -10
```

---

## 📈 結果分析流程

### 步驟 1: 檢查整體性能

```bash
cat outputs/my_test/test_metrics.json | jq '{
  accuracy: .accuracy,
  macro_f1: .macro_f1,
  weighted_f1: .weighted_f1
}'
```

### 步驟 2: 找出表現差的類別

```bash
cat outputs/my_test/test_per_class_metrics.csv | \
  awk -F',' '$5 < 0.7 {print $1, $5}' | column -t
```

### 步驟 3: 分析錯誤樣本

```bash
grep ",False$" outputs/my_test/test_predictions.csv | \
  awk -F',' '{print $2, "→", $3, "(" $6 ")"}' | head -20
```

### 步驟 4: 查看混淆矩陣

```bash
display outputs/my_test/test_confusion_matrix.png
```

---

# 常見問題與故障排除

## 🚨 CUDA Out of Memory

如果仍然出現 OOM 錯誤：

### 解決方案 1: 減少序列長度（最有效！）
```yaml
tokenizer:
  max_length: 64  # 從 128 減少到 64
```

### 解決方案 2: 增加梯度累積
```yaml
training:
  grad_accum_steps: 16  # 從 8 增加到 16
```

### 解決方案 3: 更小的 LoRA rank
```yaml
model:
  lora:
    r: 2  # 從 4 減少到 2
```

### 解決方案 4: 減少 target modules
```yaml
model:
  lora:
    target_modules: [q_proj]  # 只用一個
```

---

## ⏱️ 訓練速度太慢

### 解決方案 1: 使用資料子集
```bash
# 創建小型測試集
head -n 10000000 train.fa > train_mini.fa
```

### 解決方案 2: 減少 epochs
```bash
python train.py ... --max_epochs 3
```

### 解決方案 3: 增加 batch size（如果記憶體允許）
```yaml
training:
  batch_size: 2
  grad_accum_steps: 4
```

---

## 📊 判斷訓練是否過擬合

查看 `plots/training_curves.png`：

- ✅ **正常**：Train Loss 和 Val Loss 都下降
- ⚠️ **輕度過擬合**：Val Loss 不再下降，Train Loss 繼續下降
- 🚫 **嚴重過擬合**：Val Loss 開始上升，Train Loss 持續下降

**解決過擬合**：
- 增加 dropout
- 增加 weight_decay
- 使用 label_smoothing
- 啟用 early stopping

---

## 🎯 Confidence 分數解讀

- `> 0.9`：高置信度，通常正確
- `0.7-0.9`：中等置信度，大多正確
- `< 0.7`：低置信度，**建議人工復核**
- `< 0.5`：非常不確定，**高風險預測**

---

## 💾 必須備份的檔案

1. **`checkpoints/best.pt`** (13GB) - **最重要！**
2. **`config.json`** (2KB)
3. **`final_model/`** 目錄 (~100KB)
4. **`species_mapping_converted.tsv`** (原始映射檔案)

---

## 📚 更多資源

- **README.md** - 專案整體說明
- **DEVELOPER_GUIDE.md** - 進階配置和超參數調整
- **configs/rtx4090_optimized.yaml** - RTX 4090 配置檔案
- **configs/default.yaml** - 標準配置檔案

---

**最後更新**: 2025-11-10  
**版本**: 2.0 (整合版)


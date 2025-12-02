# 訓練速度調優指南

## 🚀 快速調優參數（按影響大小排序）

### 1️⃣ **減少訓練 Epochs** ⚡⚡⚡⚡⚡
**影響**: 最大 | **風險**: 低

```yaml
training:
  max_epochs: 3  # 原本 10
```

**速度提升**: 70% 時間節省  
**建議**: 先用 3-5 個 epochs 測試，看 early stopping 是否觸發

---

### 2️⃣ **減少訓練數據量** ⚡⚡⚡⚡⚡
**影響**: 最大 | **風險**: 中等

```bash
# 當前: 1M 訓練樣本
# 建議: 100K-500K 樣本

./create_manageable_dataset.sh 500000 125000  # 500K train, 125K val
```

**速度提升**: 50% 時間節省（用 500K 樣本）  
**風險**: 可能降低模型泛化能力

---

### 3️⃣ **減少序列長度** ⚡⚡⚡⚡
**影響**: 很大 | **風險**: 低（DNA reads 通常較短）

```yaml
tokenizer:
  max_length: 64   # 從 128 降到 64
  # max_length: 32  # 更激進的選擇
```

**速度提升**: 30-40% 時間節省  
**說明**: 
- 64: 適合大多數短 reads (Illumina ~150bp)
- 32: 適合超短 reads 或極速訓練

---

### 4️⃣ **減少 Gradient Accumulation Steps** ⚡⚡⚡
**影響**: 中等 | **風險**: 低

```yaml
training:
  grad_accum_steps: 4  # 從 8 降到 4
  # grad_accum_steps: 2  # 更快但可能降低準確度
```

**速度提升**: 20-30% 時間節省  
**權衡**: Effective batch size 變小，可能稍微影響收斂速度

---

### 5️⃣ **Early Stopping 更激進** ⚡⚡
**影響**: 中等 | **風險**: 低

```yaml
training:
  early_stopping:
    patience: 2  # 從 3 降到 2
    # patience: 1  # 極速模式
```

**速度提升**: 可能提前 1-2 個 epochs 停止  
**建議**: patience=2 是不錯的平衡

---

### 6️⃣ **禁用非必要的 Metrics** ⚡⚡
**影響**: 小到中等 | **風險**: 無

```yaml
metrics:
  primary: macro_f1
  compute_auroc: false      # 關閉 AUROC 計算
  confusion_matrix: false   # 關閉混淆矩陣
  per_class_report: false   # 關閉詳細報告
```

**速度提升**: 驗證階段快 10-15%  
**說明**: 訓練結束後可以用 `evaluate.py` 單獨計算這些指標

---

### 7️⃣ **增加 Batch Size（如果記憶體允許）** ⚡⚡
**影響**: 中等 | **風險**: 可能 OOM

```yaml
training:
  batch_size: 2  # 從 1 增加到 2
  grad_accum_steps: 4
  # 有效 batch size = 2 * 4 = 8
```

**速度提升**: 15-25% 時間節省  
**注意**: RTX 4090 24GB 可能勉強可以，建議測試

---

### 8️⃣ **減少 Logging 頻率** ⚡
**影響**: 很小 | **風險**: 無

```yaml
logging:
  log_interval: 20  # 從 5 增加到 20
```

**速度提升**: <5%  
**說明**: 減少 I/O 操作

---

### 9️⃣ **減少 Cache Clearing 頻率** ⚡
**影響**: 很小 | **風險**: 可能記憶體碎片化

```yaml
memory_optimization:
  empty_cache_steps: 20  # 從 10 增加到 20
```

**速度提升**: <5%  
**風險**: 長時間運行可能遇到記憶體問題

---

## 📊 預設配置對比

| 配置 | Epochs | 數據量 | Max Length | 預計時間 | 準確度 |
|------|--------|--------|------------|----------|--------|
| **原始 (rtx4090_optimized.yaml)** | 10 | 1M | 128 | ~12 天 | 最高 |
| **快速 (fast_training.yaml)** | 3 | 1M | 64 | ~2.5 天 | 高 |
| **極速 (見下方)** | 3 | 500K | 64 | ~1.2 天 | 中高 |
| **超快測試** | 3 | 100K | 64 | ~6 小時 | 中 |

---

## 🎯 推薦配置方案

### 方案 A：平衡速度與準確度（推薦）
```bash
# 使用 fast_training.yaml + 500K 數據
./create_manageable_dataset.sh 500000 125000
bash train_fast.sh
```
**預計時間**: 1-1.5 天  
**準確度**: 高

### 方案 B：極速訓練
```bash
# 使用 fast_training.yaml + 100K 數據
./create_manageable_dataset.sh 100000 25000
bash train_ultrafast.sh
```
**預計時間**: 4-6 小時  
**準確度**: 中高（適合快速迭代）

### 方案 C：測試模式
```bash
# 1 epoch + 10K 數據
./create_manageable_dataset.sh 10000 2500
python train.py --config configs/fast_training.yaml \
  --train_fasta train_manageable.fa \
  --val_fasta val_manageable.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --max_epochs 1 \
  --output_dir outputs/quick_test
```
**預計時間**: 30 分鐘  
**用途**: 快速驗證配置

---

## ⚠️ 重要注意事項

1. **序列長度**: 先檢查您的數據實際長度分佈
   ```bash
   # 查看序列長度統計
   grep "readlen" train_reads_shuffled_fixed.fa | head -1000 | \
     awk -F'|' '{print $4}' | sort -n | uniq -c
   ```

2. **Early Stopping**: 如果 validation loss 不再下降，會自動停止

3. **數據量**: 減少數據量前，確保保留足夠的類別代表性

4. **Checkpoint**: `best.pt` 始終保存最佳模型，即使訓練提前停止

---

## 🔧 如何應用

### 停止當前訓練（如果需要）
```bash
# 找到訓練進程
ps aux | grep "python train.py"

# 停止訓練 (使用實際的 PID)
kill <PID>
```

### 使用快速配置重新訓練
```bash
# 方案 A: 快速但保持質量
bash train_fast.sh

# 方案 B: 極速
bash train_ultrafast.sh
```

---

## 📈 監控速度

```bash
# 實時監控
bash monitor_training.sh

# 查看當前 iteration 速度
tail -f outputs/*/training.log | grep "it/s"
```

---

## 💡 進階優化（需要修改代碼）

如果您熟悉 Python，還可以：

1. **實現數據流式加載**: 避免一次性載入全部數據
2. **使用 DataLoader num_workers**: 多進程數據加載
3. **啟用 torch.compile**: PyTorch 2.0+ 編譯加速
4. **使用更小的 LoRA rank**: `r=2` 而不是 4
5. **混合精度優化**: 確保所有操作都用 bf16

---

## 🎓 最佳實踐

1. **先快後慢**: 用快速配置迭代，最終用完整配置訓練
2. **監控指標**: 確保 macro_f1 沒有大幅下降
3. **保存實驗**: 記錄不同配置的結果
4. **漸進式調整**: 一次改一個參數，觀察影響


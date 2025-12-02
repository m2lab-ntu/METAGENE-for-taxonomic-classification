# 下一步：使用數據子集進行訓練

## 當前狀況總結

### ❌ 兩次訓練失敗
1. **原始方法** (15:17-16:55): 數據載入太慢，需要 4-7 小時
2. **索引優化** (17:03-17:38): 建立索引時 OOM killed

### 根本原因
- 訓練數據太大：100GB, 5.58 億條序列
- 無法在合理時間內載入到內存或建立索引

## ✅ 解決方案：使用數據子集

已創建採樣腳本：`create_training_subset.py`

### 特點
- 📊 兩遍算法：統計 + 採樣
- 🎲 Reservoir sampling：公平隨機採樣
- 💾 流式處理：極低內存使用
- ⚡ 高效：處理 5.58 億條序列僅需 10-30 分鐘

## 執行步驟

### 步驟 1：創建訓練集子集

```bash
cd /media/user/disk2/METAGENE/classification

# 創建輸出目錄
mkdir -p training_data_subsets

# 創建訓練集子集（每物種 10,000 條序列）
nohup python3 create_training_subset.py \
  -i /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
  -o training_data_subsets/train_10k_per_species.fa \
  -n 10000 \
  > create_subset_train.log 2>&1 &

# 記錄 PID
echo $! > subset_train.pid
echo "訓練集採樣進程 PID: $(cat subset_train.pid)"
```

**預計時間**: 15-30 分鐘  
**預計輸出大小**: ~6-8 GB  
**預計序列數**: ~35,070,000 (3507 物種 × 10,000)

**監控進度**:
```bash
# 查看日誌
tail -f create_subset_train.log

# 檢查進程
ps aux | grep create_training_subset.py

# 檢查輸出文件大小
watch -n 30 'ls -lh training_data_subsets/train_10k_per_species.fa 2>/dev/null || echo "文件尚未創建"'
```

### 步驟 2：創建驗證集子集

```bash
cd /media/user/disk2/METAGENE/classification

# 創建驗證集子集（每物種 2,000 條序列）
nohup python3 create_training_subset.py \
  -i /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
  -o training_data_subsets/val_2k_per_species.fa \
  -n 2000 \
  > create_subset_val.log 2>&1 &

echo $! > subset_val.pid
echo "驗證集採樣進程 PID: $(cat subset_val.pid)"
```

**預計時間**: 3-10 分鐘  
**預計輸出大小**: ~1-1.5 GB  
**預計序列數**: ~7,014,000 (3507 物種 × 2,000)

**監控進度**:
```bash
tail -f create_subset_val.log
```

### 步驟 3：修改訓練腳本使用子集

修改 `train_benchmark_2models.sh`:

```bash
# 數據路徑
TRAIN_FASTA="/media/user/disk2/METAGENE/classification/training_data_subsets/train_10k_per_species.fa"
VAL_FASTA="/media/user/disk2/METAGENE/classification/training_data_subsets/val_2k_per_species.fa"
MAPPING_TSV="/media/user/disk2/METAGENE/classification/species_mapping_filtered.tsv"
```

### 步驟 4：重新開始訓練

```bash
cd /media/user/disk2/METAGENE/classification

source /home/user/anaconda3/bin/activate METAGENE

export HF_HOME=/media/user/disk2/.cache/huggingface
export TRANSFORMERS_CACHE=/media/user/disk2/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 運行訓練
nohup ./train_benchmark_2models.sh > train_benchmark_subset.log 2>&1 &
echo $! > train_subset.pid
echo "訓練進程 PID: $(cat train_subset.pid)"
```

**預計時間**:
- 數據載入：2-5 分鐘（極快！）
- METAGENE 訓練：2-4 小時
- DNABERT 訓練：2-4 小時
- **總計：4-8 小時**

**監控訓練**:
```bash
# 實時查看訓練日誌
tail -f train_benchmark_subset.log

# 檢查 GPU 使用
watch -n 5 nvidia-smi

# 使用監控腳本
/tmp/monitor_training.sh
```

## 數據子集規模選項

### 小規模（快速測試）
```bash
-n 1000   # 每物種 1,000 條
# 總計: ~3.5M 序列, ~600-800 MB
# 訓練時間: 1-2 小時
```

### 中等規模（推薦）⭐
```bash
-n 10000  # 每物種 10,000 條
# 總計: ~35M 序列, ~6-8 GB
# 訓練時間: 4-8 小時
```

### 大規模
```bash
-n 50000  # 每物種 50,000 條
# 總計: ~175M 序列, ~30-35 GB
# 訓練時間: 12-20 小時
# ⚠️  可能仍有內存問題
```

## 驗證子集質量

子集創建完成後，驗證數據：

```bash
# 檢查文件大小
ls -lh training_data_subsets/

# 統計序列數
grep -c "^>" training_data_subsets/train_10k_per_species.fa
grep -c "^>" training_data_subsets/val_2k_per_species.fa

# 檢查物種分佈
grep "^>" training_data_subsets/train_10k_per_species.fa | \
  cut -d'|' -f2 | sort | uniq -c | head -20
```

## 優勢

使用數據子集的好處：

1. ✅ **快速啟動**: 數據載入從數小時降低到幾分鐘
2. ✅ **內存可控**: 完全避免 OOM 問題
3. ✅ **快速迭代**: 可以快速測試不同配置
4. ✅ **有意義的結果**: 每物種 10,000 條序列足夠訓練和評估
5. ✅ **可擴展**: 驗證成功後可以增加數據量

## 後續計劃

完成子集訓練後，根據需要：

1. **如果結果滿意**: 
   - 直接使用這些模型進行 benchmark
   - 發布結果

2. **如果需要完整數據**:
   - 實現真正的流式 IterableDataset
   - 或轉換數據為 HDF5 格式
   - 使用完整數據重新訓練

## 故障排除

### 如果採樣進程被殺
- 檢查可用內存：`free -h`
- 嘗試更小的子集（-n 5000 或 -n 1000）

### 如果訓練仍然 OOM
- 減少 batch_size（在 config.yaml 中）
- 減少數據子集大小
- 增加 grad_accum_steps

### 如果訓練太慢
- 增加 num_workers（在 train.py 的 DataLoader 中）
- 檢查 I/O 瓶頸

## 文件清單

已創建的文件：
- ✅ `create_training_subset.py` - 子集採樣腳本
- ✅ `train_benchmark_2models.sh` - 訓練腳本（需要更新路徑）
- ✅ `metaclassifier/train.py` - 已優化使用 IndexedFastaDataset
- ✅ `species_mapping_filtered.tsv` - 過濾後的物種映射
- ✅ `TRAINING_STATUS_REPORT.md` - 詳細狀態報告
- ✅ `DATASET_LOADING_ANALYSIS.md` - 數據載入分析

## 預計完整時間線

```
T+0:00   開始創建訓練集子集
T+0:20   訓練集子集完成
T+0:20   開始創建驗證集子集  
T+0:30   驗證集子集完成
T+0:30   修改訓練腳本
T+0:35   開始訓練
T+0:40   數據載入完成，開始實際訓練
T+4:00   METAGENE 訓練完成
T+8:00   DNABERT 訓練完成
T+8:00   ✅ 完成！
```

總時間：約 **8 小時**（相比原方案的 12-20 小時節省大量時間）


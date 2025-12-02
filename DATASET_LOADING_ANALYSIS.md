# 數據載入效率分析

## 當前狀態 (2025-11-18 16:51)

### 訓練進程
- **啟動時間**: 15:17:31
- **運行時間**: 1 小時 38 分鐘
- **當前階段**: 載入訓練數據到內存
- **內存使用**: 6.5 GB (持續增長)
- **進程狀態**: R (running), 100% CPU
- **GPU 使用**: 極低 (數據載入階段)

### 數據規模
- **訓練文件**: `/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa`
  - 大小: **100 GB**
  - 行數: **11.17 億行**
  - 序列數: 約 **5.58 億條** (每條序列 2 行)
  - 物種數: 3507

- **驗證文件**: `/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa`
  - 大小: 需確認

## 問題分析

### 當前實現問題
`metaclassifier/train.py` 中的 `SequenceDataset` 類在 `__init__` 方法中：
1. 使用 `SeqIO.parse()` 遍歷整個 FASTA 文件
2. 將所有序列載入到 `self.sequences` 列表（內存）
3. 將所有標籤載入到 `self.labels` 列表（內存）

```python
# 第 48-58 行
for record in SeqIO.parse(fasta_path, 'fasta'):  # 無進度輸出
    # ... 處理每條序列並添加到內存列表
    self.sequences.append(str(record.seq))
    self.labels.append(self.label_to_id[label])
```

### 效率預估
基於當前進度：
- 1.5 小時載入 6.5 GB
- 載入速度: 約 4-5 GB/小時
- 如果需要載入 20-30 GB: **還需 3-5 小時**
- **總預估載入時間: 4-7 小時**

### 為什麼這麼慢？
1. **BioPython SeqIO.parse()** 對大文件較慢
2. 每條序列都需要：
   - 解析 FASTA 格式
   - 提取 class_id
   - 查詢 mapping_df
   - 字符串拷貝到內存
3. 處理 5.58 億條序列，每條操作都很昂貴
4. 無進度輸出，看起來像卡住了

## 解決方案

### 方案 1: 等待當前訓練完成
**優點:**
- 不需要修改代碼
- 讓當前進程繼續

**缺點:**
- 可能還需要 3-5 小時載入數據
- 驗證集也需要同樣長時間
- 總數據載入時間: 6-10 小時
- DNABERT 模型也需要相同時間
- **總計可能需要 12-20 小時僅用於數據載入**

**建議**: ❌ 不推薦

### 方案 2: 使用優化的數據載入器 (已創建)
文件: `metaclassifier/dataset_optimized.py`

#### 2a. IndexedFastaDataset (建議)
**工作原理:**
1. 第一次運行時建立索引文件 (可能需要 30-60 分鐘)
2. 索引包含每條序列在文件中的位置
3. 訓練時按需從文件讀取序列（零內存載入）
4. 後續訓練重用索引（秒級啟動）

**優點:**
- ✅ 支持隨機訪問和 shuffling
- ✅ 極低內存使用
- ✅ 索引建立後可重複使用
- ✅ 訓練啟動快速

**缺點:**
- 首次需要建立索引（一次性成本）
- 訓練時需要頻繁 I/O（但 SSD 很快）

#### 2b. StreamingFastaDataset
**工作原理:**
- 流式讀取文件，不載入到內存
- 邊讀邊訓練

**優點:**
- ✅ 零內存載入
- ✅ 無需預處理

**缺點:**
- ❌ 不支持 shuffling（影響訓練質量）
- ❌ 每個 epoch 都需要重新讀取文件

### 方案 3: 使用數據子集
創建較小的訓練集進行快速實驗：
- 每個物種 1000-10000 條序列
- 總計 350 萬 - 3500 萬條序列
- 估計 2-20 GB

**優點:**
- ✅ 快速驗證流程
- ✅ 適合開發和調試

**缺點:**
- ❌ 不是完整訓練
- ❌ 性能可能不如完整數據集

## 建議行動方案

### 立即行動 (推薦)
1. **停止當前訓練** (已運行 1.5 小時，但還需要 3-5 小時僅載入數據)
2. **使用 IndexedFastaDataset** 修改 `train.py`
3. **首次建立索引** (30-60 分鐘，一次性)
4. **開始訓練** (秒級啟動)

### 保守行動
1. **繼續等待** 3-5 小時讓數據載入完成
2. **同時準備優化版本**，供後續訓練使用

## 優化實施步驟

### 1. 修改 train.py
```python
# 替換 SequenceDataset 為 IndexedFastaDataset
from metaclassifier.dataset_optimized import IndexedFastaDataset

# 在 main() 函數中:
train_dataset = IndexedFastaDataset(
    fasta_path=args.train_fasta,
    mapping_df=mapping_df,
    tokenizer=tokenizer,
    max_length=config['tokenizer']['max_length'],
    index_cache_dir='./dataset_cache'  # 索引文件存放位置
)
```

### 2. 建立索引 (一次性)
```bash
# 可選：預先建立索引
python -c "
from metaclassifier.dataset_optimized import IndexedFastaDataset
import pandas as pd
from metaclassifier.tokenization import BPETokenizer

mapping_df = pd.read_csv('species_mapping_filtered.tsv', sep='\t')
tokenizer = BPETokenizer('metagene-ai/METAGENE-1', 192, True)

# 建立訓練集索引
print('Building train index...')
train_ds = IndexedFastaDataset(
    '/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa',
    mapping_df, tokenizer, 192, './dataset_cache'
)

# 建立驗證集索引
print('Building val index...')
val_ds = IndexedFastaDataset(
    '/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa',
    mapping_df, tokenizer, 192, './dataset_cache'
)
print('Done!')
"
```

### 3. 重新啟動訓練
使用相同的訓練腳本，數據載入將從數小時降低到數秒。

## 時間對比

| 方案 | 首次啟動 | 後續訓練啟動 | 內存使用 | Shuffling |
|------|---------|-------------|----------|-----------|
| 當前實現 | 4-7 小時 | 4-7 小時 | 20-30 GB | ✅ |
| IndexedFastaDataset | 30-60 分鐘 (建索引) | < 1 分鐘 | < 1 GB | ✅ |
| StreamingFastaDataset | < 1 分鐘 | < 1 分鐘 | < 1 GB | ❌ |

## 決策

**推薦**: 採用 **方案 2a (IndexedFastaDataset)**

理由：
1. 當前方法太慢（數據載入需要 12-20 小時總計）
2. 優化後首次仍需 1-2 小時，但後續訓練秒級啟動
3. 支持完整功能（shuffling, random access）
4. 極低內存使用，可以訓練更大的模型

## 下一步
等待用戶決定：
- [ ] 繼續當前訓練（還需 3-5 小時載入數據）
- [ ] 停止並使用優化版本（推薦）
- [ ] 使用數據子集進行快速驗證


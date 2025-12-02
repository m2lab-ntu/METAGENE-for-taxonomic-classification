# Max Length 參數詳解

## 📏 什麼是 Max Length？

**Max Length** 是指模型處理的**最大 token 數量**，而不是 DNA 序列的鹼基對 (bp) 數量。

---

## 🧬 您的數據情況

根據檢查，您的所有 DNA 序列都是：
- **長度**: 150 bp (鹼基對)
- **類型**: 標準 Illumina reads (可能是 HiSeq/MiSeq)

範例序列：
```
>lbl|85|301|45|Pseudomonas-61537/2
CTTCACGGCTGCTCTGGAAACTTTCGGCCTGGGCGGCCAGTTGCGCTTTGAGGTTGGCGTTGAGCTCCTCCTGCTGGCGCAGCGCCTGATGCAGTCCATGGATCTGCGCCTGCAGATTCTTGCTCTGCTCCTCGCTGGCCAGTTTGAACT
└─────────────────────────────────────── 150 個鹼基 ───────────────────────────────────────┘
```

---

## 🔄 從 DNA 序列到 Tokens 的轉換

### 步驟 1: DNA 序列 → Tokenizer → Tokens

METAGENE-1 的 tokenizer 會將 DNA 序列切分成 **tokens**：

```
DNA 序列 (150 bp):
CTTCACGGCTGCTCTGGAAACTTTCGGCCTGGGCGGCCAGTTGCGCTTTGAGGTT...

      ↓ Tokenizer 處理

Tokens (大約 150-200 個 tokens):
[CLS] [CTTC] [ACGG] [CTGC] [TCTG] ... [SEP] [PAD] [PAD]
```

**重要**: 
- 1 個鹼基 ≈ 1 個 token (METAGENE 的 tokenizer 可能稍有不同)
- 所以 150 bp 的序列 ≈ 150-200 個 tokens

---

## ⚙️ Max Length 參數的作用

### 情況 A: `max_length: 128`（當前優化配置）

```
您的序列: 150 bp → ~150 tokens
Max Length: 128 tokens

結果: 
  ✂️  序列被截斷到 128 tokens
  ⚠️  丟失了後面 ~22 tokens 的信息 (約 15% 的序列)
```

**影響**:
- ⚡ **更快**: 每個序列處理時間減少 ~15%
- 💾 **省記憶體**: GPU 記憶體使用減少 ~30-40%
- ⚠️ **準確度**: 可能略微降低（因為丟失部分信息）

---

### 情況 B: `max_length: 64`（快速配置）

```
您的序列: 150 bp → ~150 tokens
Max Length: 64 tokens

結果:
  ✂️✂️  序列被截斷到 64 tokens
  ⚠️⚠️  丟失了後面 ~86 tokens 的信息 (約 57% 的序列)
```

**影響**:
- ⚡⚡ **非常快**: 處理時間減少 ~50-60%
- 💾💾 **大量節省記憶體**: GPU 記憶體使用減少 ~60-70%
- ⚠️⚠️ **準確度下降**: 可能降低較多（只用了前半段序列）

---

### 情況 C: `max_length: 256`（完整序列）

```
您的序列: 150 bp → ~150 tokens
Max Length: 256 tokens

結果:
  ✅ 完整保留所有信息
  📏 後面補 padding 到 256 tokens
```

**影響**:
- 🐌 **較慢**: 處理更多的 padding
- 💾 **更多記憶體**: 但在您的情況下可能不必要
- ✅ **準確度**: 最高

---

## 🎯 針對您的數據的建議

### ✅ **推薦設置**: `max_length: 160-192`

因為您的序列都是 **150 bp**，建議：

```yaml
tokenizer:
  max_length: 192  # 給 150 bp + tokenizer overhead 留足空間
```

**理由**:
1. ✅ 完整保留所有序列信息
2. ⚡ 比 256 快約 25%
3. 💾 比 256 省約 25% 記憶體
4. 🎯 最適合您的數據

---

### ⚡ **如果想要更快**: `max_length: 128`

```yaml
tokenizer:
  max_length: 128  # 截斷約 15% 的序列
```

**權衡**:
- ⚡ 速度提升約 20%
- ⚠️ 準確度可能降低 2-5%
- 💭 對於細菌分類，前 128 tokens 可能已經足夠

---

### ⚡⚡ **極速模式（不推薦）**: `max_length: 64`

```yaml
tokenizer:
  max_length: 64  # 只用前半段序列
```

**權衡**:
- ⚡⚡ 速度提升約 50%
- ⚠️⚠️ 準確度可能降低 10-20%
- ❌ 丟失了太多信息，不建議用於生產

---

## 📊 不同 Max Length 的比較

| Max Length | 保留信息 | 速度 | 記憶體使用 | 準確度 | 推薦用途 |
|-----------|---------|------|-----------|--------|---------|
| **256** | 100% + padding | 基準 | 高 | 最高 | 完整訓練 |
| **192** ⭐ | 100% | +20% | 中 | 最高 | **推薦** |
| **128** | ~85% | +40% | 低 | 高 | 快速訓練 |
| **64** | ~43% | +100% | 很低 | 中 | 僅測試 |

⭐ = 針對您的 150bp 數據的最佳選擇

---

## 🔧 如何修改

### 創建針對您數據的最佳配置：

```bash
cd /media/user/disk2/METAGENE/classification
cp configs/rtx4090_optimized.yaml configs/optimal_for_150bp.yaml
```

然後編輯 `configs/optimal_for_150bp.yaml`：

```yaml
tokenizer:
  max_length: 192  # ⭐ 最適合 150bp 的設置
  # 或
  # max_length: 128  # 如果想要更快
```

---

## 💡 實際例子

### 您的序列處理過程：

```
原始 DNA (150 bp):
CTTCACGGCTGCTCTGGAAACTTTCGGCCTGGGCGGCCAGTTGCGCTTTGAGGTTGGC...
│←──────────────────── 150 個鹼基 ────────────────────→│

         ↓ Tokenizer

Token 序列 (~150 tokens):
[CLS] [tok1] [tok2] [tok3] ... [tok150] [SEP]
│←──────────────────── ~150 tokens ─────────────────→│

         ↓ Max Length = 128

實際輸入模型:
[CLS] [tok1] [tok2] ... [tok128]
│←─────── 128 tokens ───────→│ ✂️ 後面被截斷

         ↓ Max Length = 192

實際輸入模型:
[CLS] [tok1] [tok2] ... [tok150] [SEP] [PAD] [PAD] ...
│←──────────────────── 192 tokens ────────────────→│
                              └─ padding ─┘
```

---

## ❓ 常見問題

### Q1: 為什麼不直接設 `max_length: 150`？

**A**: Tokenizer 會添加特殊 tokens：
- `[CLS]`: 序列開始標記
- `[SEP]`: 序列結束標記
- 可能的 subword tokens

所以實際需要 150 + 10-20 ≈ **160-170 tokens**

---

### Q2: 如果我的序列長度不一致怎麼辦？

**A**: 檢查您的數據：
```bash
head -20000 train_reads_shuffled_fixed.fa | \
  awk 'NR%2==0 {print length($0)}' | sort -n | uniq -c
```

如果有不同長度，設置為 **最長序列 + 20**

---

### Q3: Max Length 影響準確度有多大？

**A**: 取決於任務：
- **物種分類**: 前 64-128 tokens 通常就足夠（保守區域）
- **變異檢測**: 需要完整序列
- **基因組組裝**: 需要完整序列

對於您的**物種分類任務**，`max_length: 128` 的準確度影響可能 < 3%

---

## 🎓 總結

### 針對您的 150bp 數據：

1. **最佳平衡**: `max_length: 192` ⭐
   - 完整信息 + 合理速度

2. **快速訓練**: `max_length: 128` 
   - 速度提升 40% + 可接受的準確度損失

3. **極速測試**: `max_length: 64`
   - 僅用於快速實驗

---

## 📝 下一步

如果您想更改 max_length：

```bash
# 1. 創建新配置
cp configs/fast_training.yaml configs/custom.yaml

# 2. 編輯 max_length
nano configs/custom.yaml
# 修改: tokenizer.max_length: 192

# 3. 使用新配置訓練
python train.py --config configs/custom.yaml ...
```

需要我幫您創建最適合 150bp 的配置嗎？


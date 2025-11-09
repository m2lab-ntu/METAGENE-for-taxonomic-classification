# 輸出檔案快速參考 (一頁紙版本)

## 🗂️ 訓練輸出檔案（outputs/my_experiment/）

| 檔案 | 大小 | 重要性 | 用途 |
|------|------|--------|------|
| `config.json` | ~2KB | ⭐⭐⭐⭐⭐ | 完整訓練配置，可重現結果 |
| `training.log` | ~249MB | ⭐⭐⭐⭐ | 訓練日誌（loss、GPU 記憶體） |
| `checkpoints/best.pt` | ~13GB | ⭐⭐⭐⭐⭐ | **最佳模型權重（最重要！）** |
| `final_model/label2id.json` | ~50KB | ⭐⭐⭐⭐⭐ | 物種名→ID 映射 |
| `final_model/id2label.json` | ~50KB | ⭐⭐⭐⭐⭐ | ID→物種名 映射 |
| `plots/training_curves.png` | ~1MB | ⭐⭐⭐⭐ | 訓練/驗證曲線圖 |
| `final_metrics.json` | ~5KB | ⭐⭐⭐⭐ | 最終驗證指標 |

---

## 🧪 測試輸出檔案（outputs/my_test/）

| 檔案 | 大小 | 重要性 | 用途 |
|------|------|--------|------|
| `test_metrics.json` | ~5KB | ⭐⭐⭐⭐⭐ | 整體測試性能 |
| `test_predictions.csv` | ~50MB | ⭐⭐⭐⭐⭐ | **每個樣本的預測結果** |
| `test_classification_report.json` | ~500KB | ⭐⭐⭐⭐ | 每類別 Precision/Recall/F1 |
| `test_per_class_metrics.csv` | ~500KB | ⭐⭐⭐⭐ | CSV 格式（易分析） |
| `test_confusion_matrix.png` | ~2MB | ⭐⭐⭐⭐ | 混淆矩陣圖 |

---

## 🎯 最關鍵的 5 個檔案

1. **`checkpoints/best.pt`** - 訓練好的模型（用於預測和評估）
2. **`id2label.json`** - 將預測的數字轉換為物種名稱
3. **`test_predictions.csv`** - 詳細預測結果（錯誤分析）
4. **`config.json`** - 重現訓練的完整配置
5. **`training_curves.png`** - 判斷訓練是否過擬合

---

## 📊 test_predictions.csv 格式

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
python evaluate.py --ckpt outputs/my_experiment/checkpoints/best.pt \
  --split test --output_dir outputs/my_test

# 預測新數據
python predict.py --ckpt outputs/my_experiment/checkpoints/best.pt \
  --input new_sequences.fa --output predictions.csv

# 找出錯誤樣本
grep ",False$" outputs/my_test/test_predictions.csv | head -20

# 找出低置信度預測
awk -F',' '$6 < 0.7 {print}' outputs/my_test/test_predictions.csv

# 找出表現最差的類別
sort -t',' -k5 -n outputs/my_test/test_per_class_metrics.csv | head -10
```

---

## 🚨 常見問題

### Q: 如何知道訓練是否過擬合？
**A**: 查看 `plots/training_curves.png`：
- ✅ 正常：Train Loss 和 Val Loss 都下降
- ⚠️ 輕度過擬合：Val Loss 不再下降，但 Train Loss 繼續下降
- 🚫 嚴重過擬合：Val Loss 開始上升，Train Loss 持續下降

### Q: confidence 低於多少算不好？
**A**: 
- `> 0.9`：高置信度，通常正確
- `0.7-0.9`：中等置信度，大多正確
- `< 0.7`：低置信度，**建議人工復核**
- `< 0.5`：非常不確定，**高風險預測**

### Q: 為什麼某些類別 F1-score 很低？
**A**: 可能原因：
1. 訓練樣本太少（查看 `support` 欄位）
2. 與其他類別相似（查看 `confusion_matrix.png`）
3. 數據標註錯誤

### Q: 最重要的備份檔案是哪些？
**A**: 必須備份：
1. `checkpoints/best.pt` (13GB) - **最重要！**
2. `config.json` (2KB)
3. `final_model/` 目錄 (~100KB)
4. `species_mapping_converted.tsv`（原始映射檔案）

---

## 📈 結果分析流程

```bash
# 步驟 1: 檢查整體性能
cat outputs/my_test/test_metrics.json | jq '{
  accuracy: .accuracy,
  macro_f1: .macro_f1,
  weighted_f1: .weighted_f1
}'

# 步驟 2: 找出表現差的類別
cat outputs/my_test/test_per_class_metrics.csv | \
  awk -F',' '$5 < 0.7 {print $1, $5}' | column -t

# 步驟 3: 分析錯誤樣本
grep ",False$" outputs/my_test/test_predictions.csv | \
  awk -F',' '{print $2, "→", $3, "(" $6 ")"}' | head -20

# 步驟 4: 查看混淆矩陣（找出經常混淆的類別對）
display outputs/my_test/test_confusion_matrix.png
```

---

## 💡 快速診斷表

| 症狀 | 查看檔案 | 可能原因 |
|------|---------|---------|
| 訓練很慢 | `training.log` | GPU 使用率低、序列太長 |
| Val Loss 上升 | `training_curves.png` | 過擬合、learning rate 太高 |
| 某類別 F1 低 | `test_per_class_metrics.csv` | 樣本少、類別相似 |
| 預測置信度低 | `test_predictions.csv` | 樣本質量差、類別模糊 |
| 兩類別經常混淆 | `confusion_matrix.png` | 生物學上相似、需要更長序列 |

---

## 📚 詳細文檔

- **OUTPUT_FILES_GUIDE.md** - 完整詳細說明（590 行）
- **WORKFLOW_AND_FILES.md** - 工作流程與檔案關係
- **OUTPUT_STRUCTURE_SUMMARY.txt** - 視覺化結構總覽

---

**版本**: 1.0 | **更新**: 2025-11-09

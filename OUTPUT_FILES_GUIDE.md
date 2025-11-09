# 訓練與測試輸出檔案完整指南

本文檔詳細說明 METAGENE 分類模型訓練和測試完成後所產生的所有輸出檔案的結構和意義。

---

## 📁 目錄結構總覽

### 完整訓練輸出結構

```
outputs/
└── {experiment_name}/              # 訓練實驗目錄（如：subset_training_20251107_122024）
    ├── config.json                 # 訓練配置文件
    ├── training.log                # 訓練日誌
    ├── train_class_distribution.csv  # 訓練集類別分布
    ├── val_class_distribution.csv    # 驗證集類別分布
    │
    ├── checkpoints/                # 模型檢查點目錄
    │   ├── best.pt                 # 最佳模型檢查點
    │   └── last.pt                 # 最後一個 epoch 的檢查點（可選）
    │
    ├── final_model/                # 最終模型（訓練完成後生成）
    │   ├── label2id.json           # 標籤名稱到 ID 的映射
    │   ├── id2label.json           # ID 到標籤名稱的映射
    │   └── seen_classes.txt        # 訓練過程中見過的所有類別
    │
    ├── plots/                      # 訓練可視化圖表
    │   ├── training_curves.png     # 訓練/驗證 loss 和指標曲線
    │   └── confusion_matrix.png    # 混淆矩陣（如果啟用）
    │
    └── final_metrics.json          # 最終評估指標
```

### 測試/評估輸出結構

```
outputs/
└── {evaluation_name}/              # 評估實驗目錄
    ├── {split}_metrics.json        # 整體評估指標（如：test_metrics.json）
    ├── {split}_classification_report.json  # 詳細分類報告
    ├── {split}_per_class_metrics.csv      # 每個類別的詳細指標
    ├── {split}_predictions.csv            # 所有樣本的預測結果
    └── {split}_confusion_matrix.png       # 混淆矩陣可視化
```

---

## 📄 檔案詳細說明

### 1️⃣ 訓練階段產生的檔案

#### **config.json**
- **用途**：保存完整的訓練配置
- **內容**：
  - 模型架構參數（LoRA、gradient checkpointing 等）
  - 數據集路徑和處理參數
  - 訓練超參數（batch size、learning rate、epochs）
  - 優化器和調度器設置
  - 記憶體優化配置
- **範例**：
```json
{
  "seed": 42,
  "model": {
    "encoder_path": "metagene-ai/METAGENE-1",
    "pooling": "mean",
    "num_classes": 3179,
    "lora": {
      "enabled": true,
      "r": 4,
      "alpha": 8
    }
  },
  "training": {
    "batch_size": 1,
    "grad_accum_steps": 8,
    "max_epochs": 10
  }
}
```
- **重要性**：🌟🌟🌟🌟🌟（可重現訓練結果的關鍵）

---

#### **training.log**
- **用途**：記錄完整的訓練過程
- **內容**：
  - 每個訓練步驟的 loss 和 learning rate
  - GPU 記憶體使用情況
  - 每個 epoch 的驗證指標
  - 模型檢查點保存記錄
  - 錯誤和警告訊息
- **大小**：通常數百 MB（取決於訓練長度）
- **範例片段**：
```
[12:20:30] INFO     Starting METAGENE classification training
[12:23:58] INFO     Starting training for 10 epochs
Training:   1%|▏| 14519/1000000 [40:46<47:09:12, 5.81it/s, loss=0.3086, lr=1.97e-04]
[15:11:04] INFO     GPU Memory - Allocated: 13.24GB, Reserved: 14.00GB
Validation: 100%|██████████| 250000/250000 [2:36:51<00:00, 26.56it/s]
[17:58:55] INFO     Epoch 1 - Val Loss: 0.6543, Val Acc: 0.8234, Val F1: 0.8156
```
- **重要性**：🌟🌟🌟🌟（除錯和監控的重要工具）

---

#### **train_class_distribution.csv** / **val_class_distribution.csv**
- **用途**：記錄數據集中每個類別的樣本數量和比例
- **格式**：
```csv
class,count,percentage
Pseudomonas oleovorans,291,0.0291
Gordonia westfalica,304,0.0304
Clostridium perfringens,296,0.0296
```
- **用途**：
  - 檢查類別不平衡問題
  - 驗證數據集是否正確加載
  - 分析模型在少數類別上的表現
- **重要性**：🌟🌟🌟（數據理解和問題診斷）

---

#### **checkpoints/best.pt**
- **用途**：保存驗證集上表現最佳的模型
- **內容**：
  - 模型權重（state_dict）
  - 優化器狀態
  - 調度器狀態
  - 當前 epoch 和 step
  - 驗證指標
- **大小**：約 13GB（7B 參數模型 + LoRA）
- **使用方式**：
```bash
# 用於預測
python predict.py --ckpt outputs/experiment/checkpoints/best.pt --input test.fa

# 用於評估
python evaluate.py --ckpt outputs/experiment/checkpoints/best.pt --split test
```
- **重要性**：🌟🌟🌟🌟🌟（最終模型的核心檔案）

---

#### **checkpoints/last.pt**
- **用途**：保存最後一個 epoch 的模型（可選）
- **用途**：
  - 從訓練中斷處恢復訓練
  - 比較最佳模型和最終模型的差異
- **重要性**：🌟🌟🌟（訓練恢復和實驗比較）

---

#### **final_model/label2id.json** / **id2label.json**
- **用途**：類別標籤與內部 ID 的映射
- **格式**：
```json
// label2id.json
{
  "Escherichia coli": 0,
  "Staphylococcus aureus": 1,
  "Pseudomonas aeruginosa": 2
}

// id2label.json
{
  "0": "Escherichia coli",
  "1": "Staphylococcus aureus",
  "2": "Pseudomonas aeruginosa"
}
```
- **用途**：
  - 將模型預測的數字 ID 轉換回物種名稱
  - 與 mapping TSV 文件配合使用
- **重要性**：🌟🌟🌟🌟🌟（理解預測結果的必需文件）

---

#### **final_model/seen_classes.txt**
- **用途**：記錄訓練過程中實際出現的所有類別 ID
- **格式**：
```
0
1
2
5
7
...
```
- **用途**：
  - 驗證是否所有預期類別都出現在訓練集中
  - 識別缺失的類別
- **重要性**：🌟🌟🌟（數據完整性檢查）

---

#### **plots/training_curves.png**
- **用途**：可視化訓練過程
- **內容**：
  - 訓練和驗證 Loss 曲線
  - 準確率（Accuracy）曲線
  - Macro F1-score 曲線
  - 學習率變化（如果啟用調度器）
- **範例**：
```
┌─────────────────┬─────────────────┬─────────────────┐
│   Loss          │   Accuracy      │   Macro F1      │
│   ↓             │   ↑             │   ↑             │
│   Train  Val    │   Train  Val    │   Train  Val    │
│   ----   ----   │   ----   ----   │   ----   ----   │
│   隨 epoch 變化   │   隨 epoch 變化   │   隨 epoch 變化   │
└─────────────────┴─────────────────┴─────────────────┘
```
- **用途**：
  - 判斷模型是否過擬合（training loss 持續下降但 validation loss 上升）
  - 判斷訓練是否收斂
  - 選擇最佳的訓練停止點
- **重要性**：🌟🌟🌟🌟（訓練健康狀況的視覺診斷）

---

#### **final_metrics.json**
- **用途**：保存訓練結束時在驗證集上的最終評估指標
- **內容**：
```json
{
  "loss": 0.6543,
  "accuracy": 0.8234,
  "macro_precision": 0.8156,
  "macro_recall": 0.8089,
  "macro_f1": 0.8122,
  "weighted_f1": 0.8245,
  "confusion_matrix": [[...], [...], ...]
}
```
- **用途**：
  - 快速查看模型最終表現
  - 與其他實驗進行比較
- **重要性**：🌟🌟🌟🌟（模型性能總結）

---

### 2️⃣ 測試/評估階段產生的檔案

#### **{split}_metrics.json**
- **用途**：整體評估指標（針對 test/val/train 集）
- **內容**：
```json
{
  "split": "test",
  "num_samples": 25000,
  "accuracy": 0.8456,
  "macro_precision": 0.8345,
  "macro_recall": 0.8267,
  "macro_f1": 0.8306,
  "weighted_precision": 0.8478,
  "weighted_recall": 0.8456,
  "weighted_f1": 0.8467,
  "top_5_accuracy": 0.9234,
  "auroc": 0.9678,
  "confusion_matrix": [[...], [...], ...]
}
```
- **重要性**：🌟🌟🌟🌟🌟（評估結果的核心總結）

---

#### **{split}_classification_report.json**
- **用途**：詳細的分類報告（sklearn 風格）
- **內容**：
```json
{
  "Escherichia coli": {
    "precision": 0.89,
    "recall": 0.87,
    "f1-score": 0.88,
    "support": 150
  },
  "Staphylococcus aureus": {
    "precision": 0.92,
    "recall": 0.91,
    "f1-score": 0.91,
    "support": 180
  },
  ...
  "macro avg": {
    "precision": 0.8345,
    "recall": 0.8267,
    "f1-score": 0.8306,
    "support": 25000
  },
  "weighted avg": {
    "precision": 0.8478,
    "recall": 0.8456,
    "f1-score": 0.8467,
    "support": 25000
  }
}
```
- **用途**：
  - 識別哪些類別預測效果好/差
  - 發現需要更多訓練樣本的類別
- **重要性**：🌟🌟🌟🌟（細粒度性能分析）

---

#### **{split}_per_class_metrics.csv**
- **用途**：每個類別的詳細指標（CSV 格式，便於分析）
- **格式**：
```csv
class_name,class_id,precision,recall,f1_score,support,accuracy
Escherichia coli,0,0.8900,0.8700,0.8800,150,0.8733
Staphylococcus aureus,1,0.9200,0.9100,0.9150,180,0.9111
Pseudomonas aeruginosa,2,0.7800,0.8200,0.7995,120,0.8083
```
- **用途**：
  - 用 Excel/Pandas 進行進一步分析
  - 繪製自定義圖表
  - 識別性能異常值
- **重要性**：🌟🌟🌟🌟（數據分析友好格式）

---

#### **{split}_predictions.csv**
- **用途**：每個樣本的詳細預測結果
- **格式**：
```csv
sequence_id,true_label,predicted_label,true_class_id,predicted_class_id,confidence,correct
seq_001,Escherichia coli,Escherichia coli,0,0,0.9823,True
seq_002,Staphylococcus aureus,Enterococcus faecalis,1,5,0.6234,False
seq_003,Pseudomonas aeruginosa,Pseudomonas aeruginosa,2,2,0.9567,True
```
- **用途**：
  - 識別被錯誤分類的具體樣本
  - 分析低置信度預測
  - 進行錯誤分析和 case study
- **大小**：數百 MB（取決於測試集大小）
- **重要性**：🌟🌟🌟🌟🌟（錯誤分析和模型改進的關鍵）

---

#### **{split}_confusion_matrix.png**
- **用途**：混淆矩陣的視覺化
- **內容**：熱力圖顯示每個類別之間的混淆情況
- **解讀**：
  - 對角線：正確分類的樣本數量
  - 非對角線：被誤分類的樣本數量
  - 顏色深淺：樣本數量的多少
- **範例**：
```
            Predicted
          E.coli  S.aureus  P.aeruginosa
True E.coli   130      15          5
     S.aureus   8      164          8
     P.aeruginosa 10     12         98
```
- **用途**：
  - 識別經常被混淆的類別對
  - 理解模型的錯誤模式
- **重要性**：🌟🌟🌟🌟（直觀的錯誤模式識別）

---

## 🎯 使用場景和工作流程

### 場景 1：訓練新模型

1. **啟動訓練**：
```bash
python train.py --config configs/rtx4090_optimized.yaml \
  --train_fasta data/train.fa \
  --val_fasta data/val.fa \
  --mapping_tsv data/mapping.tsv \
  --output_dir outputs/my_experiment
```

2. **監控訓練**：
```bash
# 實時查看日誌
tail -f outputs/my_experiment/training.log

# 查看 GPU 使用
nvidia-smi
```

3. **訓練完成後檢查**：
   - 查看 `plots/training_curves.png` 判斷訓練健康狀況
   - 查看 `final_metrics.json` 了解最終性能
   - 檢查 `train_class_distribution.csv` 和 `val_class_distribution.csv` 確認數據平衡性

---

### 場景 2：評估已訓練模型

1. **在測試集上評估**：
```bash
python evaluate.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --split test \
  --output_dir outputs/my_experiment_test
```

2. **分析結果**：
   - 查看 `test_metrics.json` 了解整體性能
   - 查看 `test_classification_report.json` 找出表現差的類別
   - 打開 `test_predictions.csv` 分析錯誤樣本
   - 查看 `test_confusion_matrix.png` 識別混淆模式

---

### 場景 3：模型預測

```bash
python predict.py \
  --ckpt outputs/my_experiment/checkpoints/best.pt \
  --input new_sequences.fa \
  --output predictions.csv
```

**使用 mapping 文件解釋結果**：
- 結合 `final_model/id2label.json` 將 class ID 轉換為物種名稱
- 參考 `species_mapping_converted.tsv` 獲取 tax_id 等額外資訊

---

### 場景 4：比較多個實驗

創建一個比較表格：

| 實驗名稱 | Accuracy | Macro F1 | 訓練時間 | 備註 |
|---------|----------|----------|---------|------|
| baseline | 0.82 | 0.81 | 20天 | 預設配置 |
| lora_r8 | 0.84 | 0.83 | 18天 | LoRA rank=8 |
| seq_256 | 0.85 | 0.84 | 25天 | 序列長度256 |

從各實驗的 `final_metrics.json` 提取指標進行比較。

---

## 🔍 常見問題解答

### Q1: 檢查點文件太大，如何縮小？

**A**: 檢查點包含完整的模型權重、優化器狀態等。可以：
- 僅保存模型權重（不保存優化器）
- 使用量化（8-bit/4-bit）
- 只保留 LoRA 權重（如果使用 LoRA）

---

### Q2: training.log 太大怎麼辦？

**A**: 
- 增加 `config['logging']['log_interval']`（例如從 5 改為 50）
- 使用 `gzip` 壓縮舊日誌：`gzip training.log`
- 定期清理不需要的實驗目錄

---

### Q3: 如何從檢查點恢復訓練？

**A**:
```python
# 在 train.py 中添加：
checkpoint = torch.load("outputs/experiment/checkpoints/last.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

### Q4: predictions.csv 中 confidence 低於 0.5 意味著什麼？

**A**: 
- 模型對該樣本的預測不確定
- 可能是：
  - 樣本質量差（測序錯誤、污染）
  - 屬於訓練集中少見的類別
  - 位於多個類別的邊界區域
- **建議**：人工檢查這些樣本，考慮是否需要二次驗證

---

### Q5: confusion_matrix.png 顯示兩個類別經常混淆，怎麼辦？

**A**:
1. 檢查這兩個類別在生物學上是否相似（例如同屬不同種）
2. 查看 `predictions.csv` 中這些錯誤樣本的序列
3. 考慮：
   - 增加這兩個類別的訓練樣本
   - 使用更長的序列長度
   - 檢查數據標註是否正確

---

## 📊 推薦的分析流程

1. **訓練完成後**：
   ```bash
   # 1. 檢查訓練曲線
   open plots/training_curves.png
   
   # 2. 查看最終指標
   cat final_metrics.json | jq '.'
   
   # 3. 檢查類別分布
   head -20 train_class_distribution.csv
   ```

2. **測試評估後**：
   ```bash
   # 1. 整體性能
   cat test_metrics.json | jq '.accuracy, .macro_f1'
   
   # 2. 找出表現最差的 10 個類別
   cat test_per_class_metrics.csv | sort -t',' -k5 -n | head -10
   
   # 3. 找出錯誤預測
   grep "False" test_predictions.csv | head -20
   ```

3. **生成報告**：
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # 讀取數據
   metrics = pd.read_json('test_metrics.json')
   per_class = pd.read_csv('test_per_class_metrics.csv')
   predictions = pd.read_csv('test_predictions.csv')
   
   # 分析低 F1-score 的類別
   low_f1 = per_class[per_class['f1_score'] < 0.7]
   print("Low F1-score classes:")
   print(low_f1[['class_name', 'f1_score', 'support']])
   
   # 分析錯誤分類
   errors = predictions[predictions['correct'] == False]
   print(f"\nTotal errors: {len(errors)} / {len(predictions)}")
   print(f"Error rate: {len(errors)/len(predictions):.2%}")
   ```

---

## 💡 最佳實踐建議

1. **組織實驗**：
   - 使用有意義的實驗名稱（例如：`lora_r8_seq256_20251109`）
   - 在 `config.json` 中添加 notes 欄位記錄實驗目的

2. **定期備份**：
   - `checkpoints/best.pt` 是最重要的文件，務必備份
   - `config.json` 和 mapping 文件也要保存

3. **文檔記錄**：
   - 維護一個實驗記錄表（Excel/Notion）
   - 記錄每次實驗的目的、結果和觀察

4. **清理舊文件**：
   - 訓練完成後可以刪除 `last.pt`（如果不需要恢復訓練）
   - 壓縮或刪除不重要的 `training.log`

5. **版本控制**：
   - 將 `config.json` 納入 git 版本控制
   - 使用 git tag 標記重要的實驗版本

---

## 📚 延伸閱讀

- [QUICK_START_RTX4090.md](QUICK_START_RTX4090.md) - RTX 4090 快速開始指南
- [SUCCESS_RTX4090_TRAINING.md](SUCCESS_RTX4090_TRAINING.md) - 成功訓練案例
- [TRAINING_YOUR_DATASET.md](TRAINING_YOUR_DATASET.md) - 使用自己的數據集

---

## 🆘 需要幫助？

如果遇到問題或有疑問：
1. 檢查 `training.log` 中的錯誤訊息
2. 查看 [README.md](README.md) 中的 Troubleshooting 部分
3. 參考 GitHub Issues

---

**最後更新**：2025-11-09  
**版本**：1.0


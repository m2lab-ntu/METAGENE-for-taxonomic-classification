# 使用您的資料集訓練 METAGENE 分類模型

## 資料集資訊

- **訓練集**: `/media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa` (100GB)
- **驗證集**: `/media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa` (25GB)
- **物種數量**: 3,507 種
- **Mapping 檔案**: `species_mapping_converted.tsv` (包含 class_id, label_name, tax_id)

## 資料格式

### FASTA Header 格式
```
>lbl|{class_id}|{tax_id}|{readlen}|{species_name}/{mate}
```

範例：
```
>lbl|85|301|45|Pseudomonas-61537/2
CTTCACGGCTGCTCTGGAAACTTTCGGCCTGGGCGGCCAGTTGCGCTTTGAGGTTGGCGTTGAGCTCCTCCTGCTGGCGCAGCGCCTGATGCAGTCCATG...
```

### Mapping 檔案格式
```tsv
class_id    label_name                      tax_id
0           Azorhizobium caulinodans        7
1           Buchnera aphidicola             9
2           Dictyoglomus thermophilum       14
...
```

物種名稱從 `/home/user/Metagenomics/database/selected_disease_results/all_union_species_0305_availability.csv` 自動對應。

## 快速開始

### 1. 準備 Mapping 檔案（已完成）

```bash
# 預設會自動載入物種名稱
python prepare_species_mapping.py

# 如果不需要物種名稱（使用 TaxID_* 作為標籤）
python prepare_species_mapping.py --no_names
```

### 2. 快速測試（建議先執行）

執行 1 個 epoch 來驗證資料格式和配置：

```bash
bash quick_test_species.sh
```

這會：
- 載入 METAGENE-1 模型
- 驗證資料格式正確性
- 訓練 1 個 epoch（約數小時）
- 輸出結果到 `outputs/species_test_YYYYMMDD_HHMMSS/`

### 3. 完整訓練

```bash
bash train_species_classification.sh
```

這會：
- 使用 RTX 4090 最佳化配置
- 訓練 10 個 epochs（預計數天）
- 自動保存最佳模型
- 輸出結果到 `outputs/species_classification_YYYYMMDD_HHMMSS/`

## 訓練配置

使用 `configs/rtx4090_optimized.yaml`：

| 參數 | 值 | 說明 |
|------|-----|------|
| `batch_size` | 1 | 最小 batch size |
| `grad_accum_steps` | 8 | 有效 batch size = 8 |
| `max_length` | 128 | 序列長度（從 512 減少） |
| `gradient_checkpointing` | true | 節省 ~50% activation memory |
| `lora.r` | 4 | LoRA rank |
| `lora.target_modules` | [q_proj, v_proj] | 只訓練 Q 和 V projection |
| `precision` | bf16-mixed | 混合精度訓練 |

預期 GPU 記憶體使用：**13-15GB / 24GB**

## 調整訓練參數

### 方法 1：修改 config 檔案

編輯 `configs/rtx4090_optimized.yaml`：

```yaml
training:
  max_epochs: 20  # 增加訓練輪數
  batch_size: 2   # 如果記憶體充足可增加
  
tokenizer:
  max_length: 256  # 增加序列長度（會增加記憶體使用）
```

### 方法 2：使用命令列參數

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta /media/user/disk2/full_labeled_species_train_reads_shuffled/train_reads_shuffled_fixed.fa \
  --val_fasta /media/user/disk2/full_labeled_species_val_reads_shuffled/val_reads_shuffled_fixed.fa \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir outputs/custom_experiment \
  --batch_size 2 \
  --max_epochs 5
```

## 監控訓練

### 即時監控

```bash
# 查看訓練 log
tail -f outputs/species_*/training.log

# 查看 GPU 使用情況
watch -n 1 nvidia-smi
```

### 查看結果

訓練完成後：

```bash
# 在輸出目錄會有：
outputs/species_classification_*/
├── checkpoints/
│   ├── best.pt       # 最佳模型
│   └── last.pt       # 最後一個 checkpoint
├── training.log      # 完整訓練 log
└── metrics/          # 訓練指標
```

## 評估模型

```bash
python evaluate.py \
  --ckpt outputs/species_classification_*/checkpoints/best.pt \
  --split val \
  --output_dir outputs/evaluation_results
```

## 故障排除

### CUDA Out of Memory

如果仍然出現 OOM 錯誤：

1. **減少序列長度**：
   ```yaml
   tokenizer:
     max_length: 64  # 從 128 減少到 64
   ```

2. **增加梯度累積**：
   ```yaml
   training:
     grad_accum_steps: 16  # 從 8 增加到 16
   ```

3. **使用更小的 LoRA rank**：
   ```yaml
   model:
     lora:
       r: 2  # 從 4 減少到 2
   ```

### 訓練速度太慢

如果 100GB 資料集訓練太久：

1. **使用資料子集進行測試**：
   ```bash
   # 創建小型測試集（例如前 1GB）
   head -n 10000000 train_reads_shuffled_fixed.fa > train_mini.fa
   ```

2. **減少 epochs**：
   ```bash
   python train.py ... --max_epochs 3
   ```

## 進階選項

### 繼續訓練（Resume）

```bash
python train.py \
  --config configs/rtx4090_optimized.yaml \
  --train_fasta ... \
  --val_fasta ... \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir outputs/species_classification_EXISTING \
  --resume_from outputs/species_classification_EXISTING/checkpoints/last.pt
```

### 使用 W&B 追蹤實驗

編輯 config：
```yaml
logging:
  use_wandb: true
  wandb_project: metagene-species-classification
  wandb_entity: your_username
```

## 預期結果

對於 3,507 種物種的分類任務：

- **訓練時間**：預計 2-5 天（取決於資料量和 GPU）
- **記憶體使用**：~13-15GB / 24GB（RTX 4090）
- **評估指標**：
  - Top-1 Accuracy
  - Macro F1-score
  - Per-class precision/recall
  - Confusion matrix

## 備註

1. **大型資料集**：100GB 訓練集建議使用完整的訓練流程，可能需要數天完成
2. **物種名稱**：自動從 CSV 資料庫載入，確保結果的可讀性
3. **GPU 記憶體**：RTX 4090 (24GB) 足夠訓練 METAGENE-1 (7B) 模型
4. **checkpointing**：最佳模型和最後一個 checkpoint 都會自動保存

## 相關文件

- `README.md` - 專案整體說明
- `QUICK_START_RTX4090.md` - RTX 4090 快速入門
- `SUCCESS_RTX4090_TRAINING.md` - RTX 4090 訓練成功報告
- `ATTRIBUTION_AND_CONTRIBUTIONS.md` - 程式碼貢獻說明


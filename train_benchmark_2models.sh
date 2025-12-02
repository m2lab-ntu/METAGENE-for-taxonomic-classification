#!/bin/bash
# 訓練兩個模型用於 Benchmark 比較
# Training 2 models: METAGENE + DNABERT

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Benchmark 模型訓練腳本（2模型版）        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# 數據路徑 (使用子集以避免內存問題)
TRAIN_FASTA="/media/user/disk2/METAGENE/classification/training_data_subsets/train_10k_per_species.fa"
VAL_FASTA="/media/user/disk2/METAGENE/classification/training_data_subsets/val_2k_per_species.fa"
MAPPING_TSV="/media/user/disk2/METAGENE/classification/species_mapping_filtered.tsv"

# 檢查數據是否存在
if [ ! -f "$TRAIN_FASTA" ] || [ ! -f "$VAL_FASTA" ] || [ ! -f "$MAPPING_TSV" ]; then
    echo -e "${RED}❌ 錯誤: 訓練數據不存在${NC}"
    echo "TRAIN_FASTA: $TRAIN_FASTA"
    echo "VAL_FASTA: $VAL_FASTA"
    echo "MAPPING_TSV: $MAPPING_TSV"
    exit 1
fi

# 創建時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${YELLOW}訓練計劃: 2 個模型${NC}"
echo -e "${YELLOW}1. METAGENE + BPE${NC}"
echo -e "${YELLOW}2. DNABERT + K-mer${NC}"
echo -e "${YELLOW}預計總時間: ~6-8 小時${NC}"
echo ""

# 檢查 GPU
echo -e "${BLUE}檢查 GPU 狀態...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 自動開始訓練（已移除互動式確認）
echo -e "${GREEN}自動開始訓練...${NC}"
echo ""

# ═══════════════════════════════════════════
# 模型 1: METAGENE + BPE
# ═══════════════════════════════════════════
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}模型 1/2: METAGENE + BPE (Baseline)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}配置: metaclassifier/configs/metagene_bpe.yaml${NC}"
echo -e "${YELLOW}輸出: outputs/benchmark_metagene_bpe_${TIMESTAMP}${NC}"
echo -e "${YELLOW}開始時間: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

START_TIME_1=$(date +%s)

# 設置 PYTHONPATH
export PYTHONPATH="/media/user/disk2/METAGENE/classification:$PYTHONPATH"

python metaclassifier/train.py \
  --config metaclassifier/configs/metagene_bpe.yaml \
  --train_fasta "$TRAIN_FASTA" \
  --val_fasta "$VAL_FASTA" \
  --mapping_tsv "$MAPPING_TSV" \
  --output_dir "outputs/benchmark_metagene_bpe_${TIMESTAMP}"

END_TIME_1=$(date +%s)
DURATION_1=$((END_TIME_1 - START_TIME_1))

echo ""
echo -e "${GREEN}✅ 模型 1 訓練完成${NC}"
echo -e "${YELLOW}訓練時間: $((DURATION_1/3600))h $((DURATION_1%3600/60))m $((DURATION_1%60))s${NC}"
echo ""

# ═══════════════════════════════════════════
# 模型 2: DNABERT + K-mer
# ═══════════════════════════════════════════
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}模型 2/2: DNABERT + K-mer${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}配置: metaclassifier/configs/dnabert_kmer.yaml${NC}"
echo -e "${YELLOW}輸出: outputs/benchmark_dnabert_kmer_${TIMESTAMP}${NC}"
echo -e "${YELLOW}開始時間: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""

START_TIME_2=$(date +%s)

# PYTHONPATH 已在模型1設置

python metaclassifier/train.py \
  --config metaclassifier/configs/dnabert_kmer.yaml \
  --train_fasta "$TRAIN_FASTA" \
  --val_fasta "$VAL_FASTA" \
  --mapping_tsv "$MAPPING_TSV" \
  --output_dir "outputs/benchmark_dnabert_kmer_${TIMESTAMP}"

END_TIME_2=$(date +%s)
DURATION_2=$((END_TIME_2 - START_TIME_2))

echo ""
echo -e "${GREEN}✅ 模型 2 訓練完成${NC}"
echo -e "${YELLOW}訓練時間: $((DURATION_2/3600))h $((DURATION_2%3600/60))m $((DURATION_2%60))s${NC}"
echo ""

# ═══════════════════════════════════════════
# 生成 benchmark 配置
# ═══════════════════════════════════════════
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}生成 Benchmark 配置文件${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

cat > "benchmark_methods_${TIMESTAMP}.json" << METHODS_EOF
[
  {
    "name": "METAGENE_BPE_${TIMESTAMP}",
    "checkpoint": "outputs/benchmark_metagene_bpe_${TIMESTAMP}/checkpoints/best.pt",
    "config": "metaclassifier/configs/metagene_bpe.yaml",
    "description": "METAGENE-1 with BPE tokenizer (baseline)"
  },
  {
    "name": "DNABERT_KMER_${TIMESTAMP}",
    "checkpoint": "outputs/benchmark_dnabert_kmer_${TIMESTAMP}/checkpoints/best.pt",
    "config": "metaclassifier/configs/dnabert_kmer.yaml",
    "description": "DNABERT-2 with k-mer tokenizer"
  }
]
METHODS_EOF

echo -e "${GREEN}✅ 配置文件已創建: benchmark_methods_${TIMESTAMP}.json${NC}"
echo ""

# ═══════════════════════════════════════════
# 總結
# ═══════════════════════════════════════════
TOTAL_DURATION=$((DURATION_1 + DURATION_2))

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  所有模型訓練完成！                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}訓練摘要:${NC}"
echo -e "  模型 1 (METAGENE): $((DURATION_1/3600))h $((DURATION_1%3600/60))m $((DURATION_1%60))s"
echo -e "  模型 2 (DNABERT):  $((DURATION_2/3600))h $((DURATION_2%3600/60))m $((DURATION_2%60))s"
echo -e "  總訓練時間: $((TOTAL_DURATION/3600))h $((TOTAL_DURATION%3600/60))m $((TOTAL_DURATION%60))s"
echo ""
echo -e "${YELLOW}輸出目錄:${NC}"
echo -e "  outputs/benchmark_metagene_bpe_${TIMESTAMP}/"
echo -e "  outputs/benchmark_dnabert_kmer_${TIMESTAMP}/"
echo ""
echo -e "${YELLOW}配置文件:${NC}"
echo -e "  benchmark_methods_${TIMESTAMP}.json"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}下一步: 運行 Benchmark 評估${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}使用小型測試集 (快速驗證):${NC}"
echo "python benchmark_framework.py \\"
echo "  --test_data test_data/test_small.fa \\"
echo "  --mapping_tsv species_mapping_converted.tsv \\"
echo "  --output_dir benchmark_results_${TIMESTAMP} \\"
echo "  --methods benchmark_methods_${TIMESTAMP}.json"
echo ""
echo -e "${YELLOW}使用中型測試集 (完整評估):${NC}"
echo "python benchmark_framework.py \\"
echo "  --test_data test_data/test_medium.fa \\"
echo "  --mapping_tsv species_mapping_converted.tsv \\"
echo "  --output_dir benchmark_results_${TIMESTAMP} \\"
echo "  --methods benchmark_methods_${TIMESTAMP}.json"
echo ""
echo -e "${GREEN}訓練完成時間: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""


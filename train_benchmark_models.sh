#!/bin/bash
# 訓練多個模型用於 Benchmark 比較
# Training multiple models for benchmark comparison

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Benchmark 模型訓練腳本                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# 數據路徑
TRAIN_DIR="/media/user/disk2/full_labeled_species_train_reads_shuffled"
VAL_DIR="/media/user/disk2/full_labeled_species_val_reads_shuffled"
MAPPING_TSV="/media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab"

# 檢查數據是否存在
if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$VAL_DIR" ] || [ ! -f "$MAPPING_TSV" ]; then
    echo -e "${RED}❌ 錯誤: 訓練數據不存在${NC}"
    exit 1
fi

# 創建時間戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}模型 1: METAGENE + BPE (Baseline)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}配置: configs/fast_training.yaml${NC}"
echo -e "${YELLOW}輸出: outputs/benchmark_metagene_bpe_${TIMESTAMP}${NC}"
echo ""

python train.py \
  --train_set_path "$TRAIN_DIR" \
  --validation_set_path "$VAL_DIR" \
  --mapping_df "$MAPPING_TSV" \
  --output_dir "outputs/benchmark_metagene_bpe_${TIMESTAMP}" \
  --config configs/fast_training.yaml

echo ""
echo -e "${GREEN}✅ 模型 1 訓練完成${NC}"
echo ""

echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}模型 2: METAGENE + K-mer${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}配置: metaclassifier/configs/metagene_kmer.yaml${NC}"
echo -e "${YELLOW}輸出: outputs/benchmark_metagene_kmer_${TIMESTAMP}${NC}"
echo ""

python metaclassifier/train.py \
  --config metaclassifier/configs/metagene_kmer.yaml \
  --train_fasta "$TRAIN_DIR" \
  --val_fasta "$VAL_DIR" \
  --mapping_tsv "$MAPPING_TSV" \
  --output_dir "outputs/benchmark_metagene_kmer_${TIMESTAMP}"

echo ""
echo -e "${GREEN}✅ 模型 2 訓練完成${NC}"
echo ""

echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}模型 3: DNABERT + K-mer${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}配置: metaclassifier/configs/dnabert_kmer.yaml${NC}"
echo -e "${YELLOW}輸出: outputs/benchmark_dnabert_kmer_${TIMESTAMP}${NC}"
echo ""

python metaclassifier/train.py \
  --config metaclassifier/configs/dnabert_kmer.yaml \
  --train_fasta "$TRAIN_DIR" \
  --val_fasta "$VAL_DIR" \
  --mapping_tsv "$MAPPING_TSV" \
  --output_dir "outputs/benchmark_dnabert_kmer_${TIMESTAMP}"

echo ""
echo -e "${GREEN}✅ 模型 3 訓練完成${NC}"
echo ""

# 自動生成 benchmark 配置
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}生成 Benchmark 配置文件${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

cat > "benchmark_methods_${TIMESTAMP}.json" << METHODS_EOF
[
  {
    "name": "METAGENE_BPE_${TIMESTAMP}",
    "checkpoint": "outputs/benchmark_metagene_bpe_${TIMESTAMP}/checkpoints/best.pt",
    "config": "configs/fast_training.yaml",
    "description": "METAGENE-1 with BPE tokenizer (baseline)"
  },
  {
    "name": "METAGENE_KMER_${TIMESTAMP}",
    "checkpoint": "outputs/benchmark_metagene_kmer_${TIMESTAMP}/checkpoints/best.pt",
    "config": "metaclassifier/configs/metagene_kmer.yaml",
    "description": "METAGENE-1 with k-mer tokenizer"
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

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  所有模型訓練完成！                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}下一步: 運行 Benchmark 評估${NC}"
echo ""
echo -e "${GREEN}使用小型測試集:${NC}"
echo "  python benchmark_framework.py \\"
echo "    --test_data test_data/test_small.fa \\"
echo "    --mapping_tsv species_mapping_converted.tsv \\"
echo "    --output_dir benchmark_results_${TIMESTAMP} \\"
echo "    --methods benchmark_methods_${TIMESTAMP}.json"
echo ""
echo -e "${GREEN}使用中型測試集:${NC}"
echo "  python benchmark_framework.py \\"
echo "    --test_data test_data/test_medium.fa \\"
echo "    --mapping_tsv species_mapping_converted.tsv \\"
echo "    --output_dir benchmark_results_${TIMESTAMP} \\"
echo "    --methods benchmark_methods_${TIMESTAMP}.json"
echo ""


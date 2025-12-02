#!/bin/bash
# 快速性能比較腳本 (Quick Benchmark Script)
# 用途: 快速創建測試集並進行性能評估

set -e  # 遇到錯誤立即退出

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  快速性能比較腳本 (Quick Benchmark)     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# ===== 配置參數 =====
SOURCE_DIR="/media/user/disk2/full_labeled_species_sequences"
TRAIN_DIR="/media/user/disk2/full_labeled_species_train_reads_shuffled"
VAL_DIR="/media/user/disk2/full_labeled_species_val_reads_shuffled"
MAPPING_TSV="/media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab"

# 測試集大小選項
TEST_SIZE="${1:-small}"  # small, medium, full

case $TEST_SIZE in
  small)
    READS_PER_SPECIES=50
    MAX_SPECIES=100
    TEST_OUTPUT="test_data/test_small.fa"
    echo -e "${YELLOW}📊 測試集大小: 小型 (Small)${NC}"
    echo -e "   - 每物種: $READS_PER_SPECIES 條讀"
    echo -e "   - 最多物種: $MAX_SPECIES"
    echo -e "   - 預計: ~5,000 條讀"
    ;;
  medium)
    READS_PER_SPECIES=100
    MAX_SPECIES=500
    TEST_OUTPUT="test_data/test_medium.fa"
    echo -e "${YELLOW}📊 測試集大小: 中型 (Medium)${NC}"
    echo -e "   - 每物種: $READS_PER_SPECIES 條讀"
    echo -e "   - 最多物種: $MAX_SPECIES"
    echo -e "   - 預計: ~50,000 條讀"
    ;;
  full)
    READS_PER_SPECIES=200
    MAX_SPECIES=""
    TEST_OUTPUT="test_data/test_full.fa"
    echo -e "${YELLOW}📊 測試集大小: 完整 (Full)${NC}"
    echo -e "   - 每物種: $READS_PER_SPECIES 條讀"
    echo -e "   - 所有物種"
    echo -e "   - 預計: ~700,000 條讀"
    ;;
  *)
    echo -e "${RED}❌ 錯誤: 無效的測試集大小: $TEST_SIZE${NC}"
    echo "用法: $0 [small|medium|full]"
    exit 1
    ;;
esac

echo ""

# ===== Step 1: 創建測試數據集 =====
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}Step 1: 創建測試數據集${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

if [ -f "$TEST_OUTPUT" ]; then
  echo -e "${YELLOW}⚠️  測試數據集已存在: $TEST_OUTPUT${NC}"
  read -p "是否重新創建? (y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$TEST_OUTPUT"
    rm -f "${TEST_OUTPUT%.fa}_stats.txt"
  else
    echo -e "${BLUE}→ 跳過創建，使用現有測試集${NC}"
    SKIP_CREATE=1
  fi
fi

if [ -z "$SKIP_CREATE" ]; then
  echo -e "${BLUE}→ 創建測試數據集...${NC}"
  
  CMD="python create_test_dataset.py \
    --source_dir $SOURCE_DIR \
    --output $TEST_OUTPUT \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --reads_per_species $READS_PER_SPECIES \
    --seed 42"
  
  if [ -n "$MAX_SPECIES" ]; then
    CMD="$CMD --max_species $MAX_SPECIES"
  fi
  
  echo -e "${BLUE}   命令: $CMD${NC}"
  eval $CMD
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 測試數據集創建完成!${NC}"
  else
    echo -e "${RED}❌ 測試數據集創建失敗${NC}"
    exit 1
  fi
fi

echo ""

# ===== Step 2: 檢查可用的模型 =====
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}Step 2: 檢查可用的模型${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

AVAILABLE_MODELS=()

# 檢查現有訓練結果
echo -e "${BLUE}→ 掃描 outputs/ 目錄...${NC}"

for dir in outputs/*/; do
  if [ -f "${dir}checkpoints/best.pt" ]; then
    MODEL_NAME=$(basename "$dir")
    AVAILABLE_MODELS+=("$MODEL_NAME")
    echo -e "   ${GREEN}✓${NC} $MODEL_NAME"
  fi
done

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
  echo -e "${RED}❌ 沒有找到可用的模型!${NC}"
  echo "請先訓練至少一個模型"
  exit 1
fi

echo -e "${GREEN}✅ 找到 ${#AVAILABLE_MODELS[@]} 個可用模型${NC}"
echo ""

# ===== Step 3: 創建方法配置 =====
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}Step 3: 創建方法配置${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

METHODS_CONFIG="benchmark_methods_config.json"

echo -e "${BLUE}→ 生成配置文件: $METHODS_CONFIG${NC}"

cat > "$METHODS_CONFIG" << 'EOF_START'
[
EOF_START

FIRST=1
for MODEL_NAME in "${AVAILABLE_MODELS[@]}"; do
  if [ $FIRST -eq 0 ]; then
    echo "," >> "$METHODS_CONFIG"
  fi
  FIRST=0
  
  # 檢查是否有配置文件
  CONFIG_FILE=""
  if [ -f "outputs/$MODEL_NAME/config.json" ]; then
    # 嘗試找對應的 yaml
    if [ -f "configs/${MODEL_NAME}.yaml" ]; then
      CONFIG_FILE="configs/${MODEL_NAME}.yaml"
    fi
  fi
  
  cat >> "$METHODS_CONFIG" << EOF
  {
    "name": "$MODEL_NAME",
    "checkpoint": "outputs/$MODEL_NAME/checkpoints/best.pt",
    "config": $([ -n "$CONFIG_FILE" ] && echo "\"$CONFIG_FILE\"" || echo "null")
  }
EOF
done

cat >> "$METHODS_CONFIG" << 'EOF_END'
]
EOF_END

echo -e "${GREEN}✅ 配置文件已創建${NC}"
cat "$METHODS_CONFIG"
echo ""

# ===== Step 4: 運行 Benchmark =====
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}Step 4: 運行性能評估${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"

BENCHMARK_OUTPUT="benchmark_results_${TEST_SIZE}_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}→ 開始評估...${NC}"
echo -e "   測試數據: $TEST_OUTPUT"
echo -e "   輸出目錄: $BENCHMARK_OUTPUT"
echo ""

python benchmark_framework.py \
  --test_data "$TEST_OUTPUT" \
  --mapping_tsv species_mapping_converted.tsv \
  --output_dir "$BENCHMARK_OUTPUT" \
  --methods "$METHODS_CONFIG"

if [ $? -eq 0 ]; then
  echo ""
  echo -e "${GREEN}✅ Benchmark 完成!${NC}"
  echo ""
  
  # ===== Step 5: 顯示結果 =====
  echo -e "${GREEN}══════════════════════════════════════════${NC}"
  echo -e "${GREEN}Step 5: 結果摘要${NC}"
  echo -e "${GREEN}══════════════════════════════════════════${NC}"
  
  if [ -f "$BENCHMARK_OUTPUT/benchmark_comparison.csv" ]; then
    echo ""
    cat "$BENCHMARK_OUTPUT/benchmark_comparison.csv" | column -t -s,
    echo ""
  fi
  
  echo -e "${BLUE}詳細報告:${NC}"
  ls -lh "$BENCHMARK_OUTPUT/"
  echo ""
  
  echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║            評估完成！                      ║${NC}"
  echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${BLUE}查看結果:${NC}"
  echo -e "   cat $BENCHMARK_OUTPUT/benchmark_report_*.md"
  echo -e "   cat $BENCHMARK_OUTPUT/benchmark_comparison.csv"
  echo ""
else
  echo -e "${RED}❌ Benchmark 失敗${NC}"
  exit 1
fi


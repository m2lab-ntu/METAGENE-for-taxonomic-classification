#!/usr/bin/env python3
"""
從大型訓練文件中創建子集，每個物種採樣指定數量的序列。
使用流式處理以避免內存問題。
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import random


def sample_fasta_by_species(input_fasta, output_fasta, per_species, seed=42):
    """
    從 FASTA 文件中按物種採樣序列。
    
    策略：
    1. 第一遍：統計每個物種的序列數量
    2. 第二遍：按比例採樣（reservoir sampling）
    
    Args:
        input_fasta: 輸入 FASTA 文件路徑
        output_fasta: 輸出 FASTA 文件路徑
        per_species: 每個物種採樣的序列數量
        seed: 隨機種子
    """
    random.seed(seed)
    
    print(f"📊 第一遍：統計每個物種的序列數量...")
    print(f"輸入文件: {input_fasta}")
    
    # 第一遍：統計每個物種有多少序列
    species_counts = defaultdict(int)
    total_sequences = 0
    
    with open(input_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # 解析 header: >lbl|class_id|...
                parts = line[1:].strip().split('|')
                if len(parts) >= 2:
                    try:
                        class_id = int(parts[1])
                        species_counts[class_id] += 1
                        total_sequences += 1
                        
                        if total_sequences % 10000000 == 0:
                            print(f"  已統計 {total_sequences:,} 條序列...")
                    except (ValueError, IndexError):
                        pass
    
    num_species = len(species_counts)
    print(f"\n✅ 統計完成:")
    print(f"  總序列數: {total_sequences:,}")
    print(f"  物種數: {num_species}")
    print(f"  平均每物種: {total_sequences // num_species:,} 條序列")
    
    # 計算採樣策略
    print(f"\n🎯 採樣策略:")
    species_to_sample = {}
    species_with_fewer_seqs = 0
    
    for class_id, count in species_counts.items():
        if count <= per_species:
            species_to_sample[class_id] = count  # 全部採樣
            species_with_fewer_seqs += 1
        else:
            species_to_sample[class_id] = per_species  # 採樣指定數量
    
    total_output_sequences = sum(species_to_sample.values())
    
    print(f"  目標每物種: {per_species:,} 條序列")
    print(f"  序列數 < {per_species} 的物種: {species_with_fewer_seqs}")
    print(f"  預期輸出總序列數: {total_output_sequences:,}")
    print(f"  壓縮率: {total_output_sequences / total_sequences * 100:.1f}%")
    
    # 第二遍：採樣序列
    print(f"\n📝 第二遍：採樣並寫入序列...")
    
    # 使用 reservoir sampling 算法
    # 為每個物種維護一個 reservoir
    reservoirs = {class_id: [] for class_id in species_counts.keys()}
    current_counts = defaultdict(int)
    
    sequences_read = 0
    with open(input_fasta, 'r') as f:
        current_header = None
        current_class_id = None
        
        for line in f:
            if line.startswith('>'):
                # 新的序列
                current_header = line
                parts = line[1:].strip().split('|')
                
                if len(parts) >= 2:
                    try:
                        current_class_id = int(parts[1])
                        sequences_read += 1
                        
                        if sequences_read % 10000000 == 0:
                            print(f"  已處理 {sequences_read:,} / {total_sequences:,} 序列 ({sequences_read/total_sequences*100:.1f}%)")
                    except (ValueError, IndexError):
                        current_class_id = None
            else:
                # 序列行
                if current_class_id is not None and current_header is not None:
                    sequence = line
                    current_counts[current_class_id] += 1
                    k = species_to_sample[current_class_id]
                    
                    # Reservoir sampling
                    if len(reservoirs[current_class_id]) < k:
                        # Reservoir 還沒滿，直接添加
                        reservoirs[current_class_id].append((current_header, sequence))
                    else:
                        # Reservoir 已滿，隨機替換
                        j = random.randint(0, current_counts[current_class_id] - 1)
                        if j < k:
                            reservoirs[current_class_id][j] = (current_header, sequence)
                    
                    current_header = None
                    current_class_id = None
    
    print(f"\n💾 寫入輸出文件: {output_fasta}")
    
    # 寫入採樣的序列
    sequences_written = 0
    with open(output_fasta, 'w') as f:
        for class_id in sorted(reservoirs.keys()):
            for header, sequence in reservoirs[class_id]:
                f.write(header)
                f.write(sequence)
                sequences_written += 1
                
                if sequences_written % 1000000 == 0:
                    print(f"  已寫入 {sequences_written:,} / {total_output_sequences:,} 序列")
    
    print(f"\n✅ 完成!")
    print(f"  輸出文件: {output_fasta}")
    print(f"  實際寫入序列數: {sequences_written:,}")
    
    # 顯示採樣統計
    print(f"\n📈 採樣統計:")
    sampled_per_species = defaultdict(int)
    for class_id, samples in reservoirs.items():
        sampled_per_species[class_id] = len(samples)
    
    print(f"  最小每物種: {min(sampled_per_species.values()):,}")
    print(f"  最大每物種: {max(sampled_per_species.values()):,}")
    print(f"  平均每物種: {sum(sampled_per_species.values()) // len(sampled_per_species):,}")


def main():
    parser = argparse.ArgumentParser(
        description="從大型訓練文件創建子集（流式處理，低內存）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 每個物種採樣 10,000 條序列
  %(prog)s -i train.fa -o train_10k.fa -n 10000
  
  # 每個物種採樣 1,000 條序列（快速測試）
  %(prog)s -i train.fa -o train_1k.fa -n 1000
  
  # 每個物種採樣 50,000 條序列（大規模）
  %(prog)s -i train.fa -o train_50k.fa -n 50000
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='輸入 FASTA 文件')
    parser.add_argument('-o', '--output', required=True,
                        help='輸出 FASTA 文件')
    parser.add_argument('-n', '--per-species', type=int, required=True,
                        help='每個物種採樣的序列數量')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (default: 42)')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 錯誤: 輸入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # 檢查輸出文件
    output_path = Path(args.output)
    if output_path.exists():
        response = input(f"⚠️  輸出文件已存在: {args.output}\n是否覆蓋? (yes/no): ")
        if response.lower() != 'yes':
            print("取消操作")
            sys.exit(0)
    
    # 創建輸出目錄
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🔬 創建訓練數據子集")
    print("="*60)
    
    try:
        sample_fasta_by_species(
            input_fasta=args.input,
            output_fasta=args.output,
            per_species=args.per_species,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


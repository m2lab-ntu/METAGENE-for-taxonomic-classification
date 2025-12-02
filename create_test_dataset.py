#!/usr/bin/env python3
"""
創建獨立的測試數據集，用於評估不同方法的性能
Creates an independent test dataset for evaluating different methods

特點 (Features):
1. 確保與訓練/驗證集不重疊 (No overlap with train/val sets)
2. 每個物種採樣固定數量的讀 (Fixed number of reads per species)
3. 保留 ground truth 標籤 (Keep ground truth labels)
4. 支持多種採樣策略 (Support multiple sampling strategies)
"""

import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import gzip


def parse_fasta_header(header):
    """解析 FASTA header: >lbl|class_id|tax_id|length|species_name"""
    parts = header.strip('>').split('|')
    if len(parts) >= 5:
        return {
            'class_id': parts[1],
            'tax_id': parts[2],
            'length': parts[3],
            'species_name': parts[4]
        }
    return None


def read_fasta_sequences(fasta_file, max_reads_per_file=None):
    """讀取 FASTA 文件中的序列"""
    sequences = []
    
    if fasta_file.endswith('.gz'):
        opener = lambda f: gzip.open(f, 'rt')
    else:
        opener = open
    
    with opener(fasta_file) as f:
        header = None
        sequence_lines = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header and sequence_lines:
                    sequences.append({
                        'header': header,
                        'sequence': ''.join(sequence_lines)
                    })
                    if max_reads_per_file and len(sequences) >= max_reads_per_file:
                        break
                header = line
                sequence_lines = []
            else:
                sequence_lines.append(line)
        
        # 添加最後一條序列
        if header and sequence_lines:
            sequences.append({
                'header': header,
                'sequence': ''.join(sequence_lines)
            })
    
    return sequences


def get_train_val_read_ids(train_dir, val_dir):
    """獲取訓練和驗證集中的所有讀 ID（避免重疊）"""
    read_ids = set()
    
    print("📖 讀取訓練集和驗證集的序列 ID...")
    
    for data_dir, name in [(train_dir, "訓練集"), (val_dir, "驗證集")]:
        if not os.path.exists(data_dir):
            print(f"⚠️  {name} 目錄不存在: {data_dir}")
            continue
            
        files = [f for f in os.listdir(data_dir) if f.endswith('.fa') or f.endswith('.fasta')]
        
        for filename in tqdm(files, desc=f"處理{name}"):
            filepath = os.path.join(data_dir, filename)
            sequences = read_fasta_sequences(filepath)
            
            for seq in sequences:
                # 使用 header + sequence 作為唯一 ID
                read_id = f"{seq['header']}_{seq['sequence'][:50]}"
                read_ids.add(read_id)
    
    print(f"✅ 共找到 {len(read_ids):,} 條訓練/驗證序列")
    return read_ids


def create_test_dataset(
    source_dir,
    output_file,
    train_dir=None,
    val_dir=None,
    reads_per_species=100,
    max_species=None,
    min_sequence_length=50,
    seed=42
):
    """
    創建測試數據集
    
    Args:
        source_dir: 源數據目錄 (full_labeled_species_sequences)
        output_file: 輸出文件路徑
        train_dir: 訓練集目錄（用於檢查重疊）
        val_dir: 驗證集目錄（用於檢查重疊）
        reads_per_species: 每個物種採樣的讀數
        max_species: 最大物種數（None = 全部）
        min_sequence_length: 最小序列長度
        seed: 隨機種子
    """
    random.seed(seed)
    
    # 獲取訓練/驗證集的讀 ID（用於去重）
    existing_read_ids = set()
    if train_dir or val_dir:
        existing_read_ids = get_train_val_read_ids(train_dir or "", val_dir or "")
    
    print(f"\n🔬 創建測試數據集...")
    print(f"   源目錄: {source_dir}")
    print(f"   輸出文件: {output_file}")
    print(f"   每物種讀數: {reads_per_species}")
    print(f"   最小序列長度: {min_sequence_length}")
    
    # 獲取所有物種文件
    species_files = [f for f in os.listdir(source_dir) if f.endswith('.fa')]
    
    if max_species:
        species_files = random.sample(species_files, min(max_species, len(species_files)))
    
    print(f"   處理物種數: {len(species_files)}")
    
    # 為每個物種採樣序列
    test_sequences = []
    species_stats = defaultdict(lambda: {'total': 0, 'sampled': 0, 'filtered': 0, 'overlap': 0})
    
    for species_file in tqdm(species_files, desc="採樣序列"):
        species_path = os.path.join(source_dir, species_file)
        sequences = read_fasta_sequences(species_path)
        
        species_name = species_file.replace('.fa', '')
        species_stats[species_name]['total'] = len(sequences)
        
        # 過濾序列
        valid_sequences = []
        for seq in sequences:
            # 檢查長度
            if len(seq['sequence']) < min_sequence_length:
                species_stats[species_name]['filtered'] += 1
                continue
            
            # 檢查是否與訓練/驗證集重疊
            read_id = f"{seq['header']}_{seq['sequence'][:50]}"
            if read_id in existing_read_ids:
                species_stats[species_name]['overlap'] += 1
                continue
            
            valid_sequences.append(seq)
        
        # 採樣
        num_to_sample = min(reads_per_species, len(valid_sequences))
        sampled = random.sample(valid_sequences, num_to_sample)
        
        species_stats[species_name]['sampled'] = len(sampled)
        test_sequences.extend(sampled)
    
    # 打亂順序
    random.shuffle(test_sequences)
    
    # 寫入輸出文件
    print(f"\n📝 寫入測試數據集...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for seq in test_sequences:
            f.write(f"{seq['header']}\n")
            f.write(f"{seq['sequence']}\n")
    
    # 統計信息
    total_sequences = len(test_sequences)
    total_species = len([s for s in species_stats.values() if s['sampled'] > 0])
    total_filtered = sum(s['filtered'] for s in species_stats.values())
    total_overlap = sum(s['overlap'] for s in species_stats.values())
    
    print(f"\n✅ 測試數據集創建完成！")
    print(f"\n📊 統計信息:")
    print(f"   總序列數: {total_sequences:,}")
    print(f"   物種數: {total_species}")
    print(f"   過濾掉的序列 (長度不足): {total_filtered:,}")
    print(f"   過濾掉的序列 (與訓練集重疊): {total_overlap:,}")
    print(f"   平均每物種序列數: {total_sequences/total_species:.1f}")
    
    # 保存統計信息
    stats_file = output_file.replace('.fa', '_stats.txt').replace('.fasta', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"測試數據集統計信息\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"總序列數: {total_sequences:,}\n")
        f.write(f"物種數: {total_species}\n")
        f.write(f"過濾序列 (長度): {total_filtered:,}\n")
        f.write(f"過濾序列 (重疊): {total_overlap:,}\n")
        f.write(f"平均每物種: {total_sequences/total_species:.1f}\n\n")
        f.write(f"每個物種的詳細統計:\n")
        f.write(f"{'-'*60}\n")
        
        for species, stats in sorted(species_stats.items()):
            if stats['sampled'] > 0:
                f.write(f"{species}: {stats['sampled']}/{stats['total']} "
                       f"(過濾: {stats['filtered']}, 重疊: {stats['overlap']})\n")
    
    print(f"   統計信息已保存: {stats_file}")
    
    return test_sequences, species_stats


def main():
    parser = argparse.ArgumentParser(
        description="創建用於性能比較的測試數據集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 創建小型測試集（每物種 50 條讀，最多 100 個物種）
  python create_test_dataset.py \\
    --source_dir /media/user/disk2/full_labeled_species_sequences \\
    --output test_data/test_small.fa \\
    --reads_per_species 50 \\
    --max_species 100

  # 創建完整測試集，並檢查與訓練集的重疊
  python create_test_dataset.py \\
    --source_dir /media/user/disk2/full_labeled_species_sequences \\
    --output test_data/test_full.fa \\
    --train_dir /media/user/disk2/full_labeled_species_train_reads_shuffled \\
    --val_dir /media/user/disk2/full_labeled_species_val_reads_shuffled \\
    --reads_per_species 100
        """
    )
    
    parser.add_argument('--source_dir', required=True,
                       help='源數據目錄')
    parser.add_argument('--output', required=True,
                       help='輸出文件路徑')
    parser.add_argument('--train_dir', default=None,
                       help='訓練集目錄（用於檢查重疊）')
    parser.add_argument('--val_dir', default=None,
                       help='驗證集目錄（用於檢查重疊）')
    parser.add_argument('--reads_per_species', type=int, default=100,
                       help='每個物種採樣的讀數 (預設: 100)')
    parser.add_argument('--max_species', type=int, default=None,
                       help='最大物種數 (None = 全部)')
    parser.add_argument('--min_length', type=int, default=50,
                       help='最小序列長度 (預設: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子 (預設: 42)')
    
    args = parser.parse_args()
    
    # 創建測試數據集
    create_test_dataset(
        source_dir=args.source_dir,
        output_file=args.output,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        reads_per_species=args.reads_per_species,
        max_species=args.max_species,
        min_sequence_length=args.min_length,
        seed=args.seed
    )


if __name__ == '__main__':
    main()


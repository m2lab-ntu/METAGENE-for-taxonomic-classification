#!/usr/bin/env python3
"""
過濾 species_mapping_converted.tsv，只保留在 all_available_species_mapping.tab 中的 Tax ID
"""

import pandas as pd
import sys

def filter_mapping():
    # 讀取原始映射文件（包含可用的 Tax ID）
    reference_file = "/media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab"
    print(f"讀取參考文件: {reference_file}")
    ref_df = pd.read_csv(reference_file, sep='\t')
    
    # 獲取所有可用的 Tax ID
    available_tax_ids = set(ref_df['Tax ID'].values)
    print(f"可用的 Tax ID 數量: {len(available_tax_ids)}")
    
    # 讀取需要過濾的映射文件
    input_file = "/media/user/disk2/METAGENE/classification/species_mapping_converted.tsv"
    print(f"讀取輸入文件: {input_file}")
    input_df = pd.read_csv(input_file, sep='\t')
    print(f"原始記錄數: {len(input_df)}")
    
    # 過濾：只保留 Tax ID 在可用列表中的記錄
    filtered_df = input_df[input_df['tax_id'].isin(available_tax_ids)].copy()
    print(f"過濾後記錄數: {len(filtered_df)}")
    
    # 重新分配 class_id（從 0 開始連續編號）
    filtered_df = filtered_df.sort_values('tax_id').reset_index(drop=True)
    filtered_df['class_id'] = range(len(filtered_df))
    
    # 保存到新文件
    output_file = "/media/user/disk2/METAGENE/classification/species_mapping_filtered.tsv"
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✅ 已保存過濾後的映射文件: {output_file}")
    print(f"   包含 {len(filtered_df)} 個物種")
    
    # 顯示前幾行
    print("\n前 5 行:")
    print(filtered_df.head())
    
    # 檢查是否有遺失的 Tax ID
    missing_tax_ids = available_tax_ids - set(filtered_df['tax_id'].values)
    if missing_tax_ids:
        print(f"\n⚠️  警告: 有 {len(missing_tax_ids)} 個 Tax ID 在參考文件中但不在輸入文件中")
        print(f"   前 10 個遺失的 Tax ID: {sorted(list(missing_tax_ids))[:10]}")
    else:
        print("\n✅ 所有參考 Tax ID 都已包含")

if __name__ == "__main__":
    try:
        filter_mapping()
    except Exception as e:
        print(f"❌ 錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)



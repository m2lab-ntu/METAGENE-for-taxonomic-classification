#!/usr/bin/env python3
"""
性能比較評估框架 (Benchmark Framework)
用於標準化評估和比較不同方法的性能

功能 (Features):
1. 統一的評估指標 (Unified metrics)
2. 多方法比較 (Multi-method comparison)
3. 詳細的性能報告 (Detailed performance reports)
4. 可視化結果 (Visualization)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class BenchmarkEvaluator:
    """評估器類"""
    
    def __init__(self, test_data_path: str, mapping_tsv: str, output_dir: str):
        self.test_data_path = test_data_path
        self.mapping_tsv = mapping_tsv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存儲結果
        self.results = {}
        
        print(f"📊 評估器初始化")
        print(f"   測試數據: {test_data_path}")
        print(f"   標籤映射: {mapping_tsv}")
        print(f"   輸出目錄: {output_dir}")
    
    def load_ground_truth(self) -> Dict[str, str]:
        """從測試數據中提取 ground truth"""
        print("\n📖 讀取 Ground Truth...")
        
        ground_truth = {}
        current_header = None
        
        with open(self.test_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_header = line
                    # 解析: >lbl|class_id|tax_id|length|species_name
                    parts = line.strip('>').split('|')
                    if len(parts) >= 2:
                        seq_id = line
                        true_class_id = parts[1]
                        ground_truth[seq_id] = true_class_id
        
        print(f"✅ 讀取 {len(ground_truth):,} 條 Ground Truth")
        return ground_truth
    
    def run_prediction(self, 
                      method_name: str,
                      model_checkpoint: str,
                      config_file: str = None,
                      script_path: str = "predict.py",
                      batch_size: int = 256) -> str:
        """
        運行預測
        
        Args:
            method_name: 方法名稱
            model_checkpoint: 模型檢查點路徑
            config_file: 配置文件路徑（如果需要）
            script_path: 預測腳本路徑
            batch_size: Batch size
        
        Returns:
            預測結果文件路徑
        """
        print(f"\n🔮 運行預測: {method_name}")
        
        prediction_file = self.output_dir / f"predictions_{method_name}.csv"
        
        # 構建命令
        cmd = [
            "python", script_path,
            "--ckpt", model_checkpoint,
            "--split", "test",
            "--input", self.test_data_path,
            "--output", str(prediction_file),
            "--batch_size", str(batch_size)
        ]
        
        if config_file:
            cmd.extend(["--config", config_file])
        
        print(f"   命令: {' '.join(cmd)}")
        
        # 運行預測
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ 預測完成: {prediction_file}")
            return str(prediction_file)
        except subprocess.CalledProcessError as e:
            print(f"❌ 預測失敗: {e}")
            print(f"   標準輸出: {e.stdout}")
            print(f"   標準錯誤: {e.stderr}")
            return None
    
    def evaluate_predictions(self,
                            method_name: str,
                            prediction_file: str,
                            ground_truth: Dict[str, str]) -> Dict:
        """
        評估預測結果
        
        Args:
            method_name: 方法名稱
            prediction_file: 預測結果文件
            ground_truth: Ground truth 字典
        
        Returns:
            評估指標字典
        """
        print(f"\n📈 評估預測結果: {method_name}")
        
        # 讀取預測結果
        predictions_df = pd.read_csv(prediction_file)
        
        # 計算指標
        correct = 0
        total = 0
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        confidence_scores = []
        
        for _, row in predictions_df.iterrows():
            seq_id = row['sequence_id']
            predicted_class = str(row['predicted_class_id'])
            confidence = row.get('confidence', 0.0)
            
            if seq_id in ground_truth:
                true_class = ground_truth[seq_id]
                total += 1
                
                per_class_total[true_class] += 1
                confidence_scores.append(confidence)
                
                if predicted_class == true_class:
                    correct += 1
                    per_class_correct[true_class] += 1
        
        # 計算總體準確率
        accuracy = correct / total if total > 0 else 0
        
        # 計算每類準確率
        per_class_accuracy = {}
        for class_id in per_class_total:
            per_class_accuracy[class_id] = (
                per_class_correct[class_id] / per_class_total[class_id]
                if per_class_total[class_id] > 0 else 0
            )
        
        # 計算宏平均準確率
        macro_accuracy = np.mean(list(per_class_accuracy.values())) if per_class_accuracy else 0
        
        # 計算加權準確率
        weighted_accuracy = sum(
            per_class_accuracy[c] * per_class_total[c] 
            for c in per_class_accuracy
        ) / total if total > 0 else 0
        
        # 平均置信度
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        metrics = {
            'method_name': method_name,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'macro_accuracy': macro_accuracy,
            'weighted_accuracy': weighted_accuracy,
            'average_confidence': avg_confidence,
            'num_classes': len(per_class_total),
            'per_class_accuracy': per_class_accuracy,
            'per_class_total': dict(per_class_total)
        }
        
        print(f"✅ 評估完成")
        print(f"   準確率: {accuracy:.4f}")
        print(f"   宏平均準確率: {macro_accuracy:.4f}")
        print(f"   平均置信度: {avg_confidence:.4f}")
        
        return metrics
    
    def compare_methods(self, methods: List[Dict]) -> pd.DataFrame:
        """
        比較多個方法
        
        Args:
            methods: 方法列表，每個方法包含 name, checkpoint, config 等
        
        Returns:
            比較結果 DataFrame
        """
        print(f"\n🔬 開始比較 {len(methods)} 個方法...")
        
        # 加載 ground truth
        ground_truth = self.load_ground_truth()
        
        # 評估每個方法
        all_results = []
        
        for method in methods:
            method_name = method['name']
            checkpoint = method['checkpoint']
            config = method.get('config', None)
            
            print(f"\n{'='*60}")
            print(f"方法: {method_name}")
            print(f"{'='*60}")
            
            # 運行預測
            prediction_file = self.run_prediction(
                method_name=method_name,
                model_checkpoint=checkpoint,
                config_file=config
            )
            
            if prediction_file is None:
                print(f"⚠️  跳過方法: {method_name}")
                continue
            
            # 評估結果
            metrics = self.evaluate_predictions(
                method_name=method_name,
                prediction_file=prediction_file,
                ground_truth=ground_truth
            )
            
            all_results.append(metrics)
            self.results[method_name] = metrics
        
        # 創建比較表
        comparison_df = pd.DataFrame([
            {
                'Method': r['method_name'],
                'Accuracy': r['accuracy'],
                'Macro Accuracy': r['macro_accuracy'],
                'Weighted Accuracy': r['weighted_accuracy'],
                'Avg Confidence': r['average_confidence'],
                'Num Classes': r['num_classes'],
                'Total Samples': r['total_samples']
            }
            for r in all_results
        ])
        
        # 按準確率排序
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """生成詳細報告"""
        print(f"\n📝 生成報告...")
        
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 性能比較報告 (Benchmark Report)\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 測試配置\n\n")
            f.write(f"- **測試數據**: `{self.test_data_path}`\n")
            f.write(f"- **標籤映射**: `{self.mapping_tsv}`\n")
            f.write(f"- **評估方法數**: {len(comparison_df)}\n\n")
            
            f.write("## 整體比較\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## 詳細指標\n\n")
            
            for method_name, metrics in self.results.items():
                f.write(f"### {method_name}\n\n")
                f.write(f"- **總樣本數**: {metrics['total_samples']:,}\n")
                f.write(f"- **正確預測數**: {metrics['correct_predictions']:,}\n")
                f.write(f"- **準確率**: {metrics['accuracy']:.4f}\n")
                f.write(f"- **宏平均準確率**: {metrics['macro_accuracy']:.4f}\n")
                f.write(f"- **加權準確率**: {metrics['weighted_accuracy']:.4f}\n")
                f.write(f"- **平均置信度**: {metrics['average_confidence']:.4f}\n")
                f.write(f"- **類別數**: {metrics['num_classes']}\n\n")
            
            f.write("## 結論\n\n")
            
            best_method = comparison_df.iloc[0]
            f.write(f"**最佳方法**: {best_method['Method']}\n")
            f.write(f"- 準確率: {best_method['Accuracy']:.4f}\n")
            f.write(f"- 宏平均準確率: {best_method['Macro Accuracy']:.4f}\n\n")
        
        print(f"✅ 報告已保存: {report_file}")
        
        # 同時保存 JSON 格式
        json_file = report_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON 結果已保存: {json_file}")
        
        return str(report_file)
    
    def save_comparison_csv(self, comparison_df: pd.DataFrame) -> str:
        """保存比較結果為 CSV"""
        csv_file = self.output_dir / "benchmark_comparison.csv"
        comparison_df.to_csv(csv_file, index=False)
        print(f"✅ 比較結果已保存: {csv_file}")
        return str(csv_file)


def main():
    parser = argparse.ArgumentParser(
        description="性能比較評估框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python benchmark_framework.py \\
    --test_data test_data/test_full.fa \\
    --mapping_tsv species_mapping_converted.tsv \\
    --output_dir benchmark_results \\
    --methods methods_config.json
        """
    )
    
    parser.add_argument('--test_data', required=True,
                       help='測試數據路徑')
    parser.add_argument('--mapping_tsv', required=True,
                       help='標籤映射文件')
    parser.add_argument('--output_dir', required=True,
                       help='輸出目錄')
    parser.add_argument('--methods', required=True,
                       help='方法配置 JSON 文件')
    
    args = parser.parse_args()
    
    # 讀取方法配置
    with open(args.methods, 'r') as f:
        methods = json.load(f)
    
    # 創建評估器
    evaluator = BenchmarkEvaluator(
        test_data_path=args.test_data,
        mapping_tsv=args.mapping_tsv,
        output_dir=args.output_dir
    )
    
    # 比較方法
    comparison_df = evaluator.compare_methods(methods)
    
    # 生成報告
    evaluator.generate_report(comparison_df)
    evaluator.save_comparison_csv(comparison_df)
    
    print(f"\n{'='*60}")
    print("🎉 評估完成！")
    print(f"{'='*60}\n")
    print(comparison_df.to_string(index=False))


if __name__ == '__main__':
    main()


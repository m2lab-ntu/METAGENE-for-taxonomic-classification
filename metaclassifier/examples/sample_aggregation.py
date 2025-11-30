"""
Example: Sample-level aggregation and diversity metrics.

Demonstrates:
- Aggregating per-read predictions to per-sample abundance
- Computing diversity metrics
- Filtering by abundance threshold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Import aggregate module
import aggregate as agg


def create_mock_predictions():
    """Create mock per-read predictions."""
    np.random.seed(42)
    
    samples = []
    
    # Sample 1: Dominated by E. coli
    for i in range(150):
        samples.append({
            'sequence_id': f'sample1_read{i}',
            'sample_id': 'Sample01',
            'predicted_class': 'Escherichia_coli',
            'confidence': np.random.uniform(0.7, 0.99)
        })
    
    for i in range(50):
        samples.append({
            'sequence_id': f'sample1_read{150+i}',
            'sample_id': 'Sample01',
            'predicted_class': 'Bacteroides_fragilis',
            'confidence': np.random.uniform(0.6, 0.95)
        })
    
    # Sample 2: More diverse
    species = ['Salmonella_enterica', 'Klebsiella_pneumoniae', 'Pseudomonas_aeruginosa']
    counts = [100, 60, 40]
    
    for sp, count in zip(species, counts):
        for i in range(count):
            samples.append({
                'sequence_id': f'sample2_read{i}_{sp}',
                'sample_id': 'Sample02',
                'predicted_class': sp,
                'confidence': np.random.uniform(0.5, 0.98)
            })
    
    return pd.DataFrame(samples)


def main():
    print("=" * 80)
    print("Sample Aggregation Example")
    print("=" * 80)
    
    # 1. Create mock data
    print("\n1. Creating mock per-read predictions...")
    predictions_df = create_mock_predictions()
    print(f"✓ Created {len(predictions_df)} predictions")
    print(f"✓ Samples: {predictions_df['sample_id'].nunique()}")
    print(f"✓ Species: {predictions_df['predicted_class'].nunique()}")
    
    print("\nPreview:")
    print(predictions_df.head())
    
    # 2. Basic aggregation
    print("\n" + "-" * 80)
    print("2. Aggregating to sample level...")
    print("-" * 80)
    
    abundance_df = agg.aggregate_predictions_to_sample(
        predictions_df,
        confidence_threshold=0.5
    )
    
    print(f"\n✓ Aggregated to {len(abundance_df)} sample-species pairs")
    print("\nAbundance table:")
    print(abundance_df.to_string(index=False))
    
    # 3. Diversity metrics
    print("\n" + "-" * 80)
    print("3. Computing diversity metrics...")
    print("-" * 80)
    
    diversity_df = agg.compute_diversity_metrics(abundance_df)
    
    print("\nDiversity metrics:")
    print(diversity_df.to_string(index=False))
    
    # 4. Filter by abundance
    print("\n" + "-" * 80)
    print("4. Filtering by abundance threshold...")
    print("-" * 80)
    
    filtered_df = agg.filter_by_abundance_threshold(
        abundance_df,
        min_abundance=0.1,
        min_read_count=20
    )
    
    print("\nFiltered abundance table:")
    print(filtered_df.to_string(index=False))
    
    # 5. Create abundance matrix
    print("\n" + "-" * 80)
    print("5. Creating abundance matrix...")
    print("-" * 80)
    
    matrix = agg.create_abundance_matrix(abundance_df)
    
    print("\nAbundance matrix (samples x species):")
    print(matrix)
    
    # 6. Visualization
    print("\n" + "-" * 80)
    print("6. Summary statistics...")
    print("-" * 80)
    
    for sample_id in abundance_df['sample_id'].unique():
        sample_data = abundance_df[abundance_df['sample_id'] == sample_id]
        diversity = diversity_df[diversity_df['sample_id'] == sample_id].iloc[0]
        
        print(f"\n{sample_id}:")
        print(f"  Total reads: {diversity['total_reads']}")
        print(f"  Species richness: {diversity['num_species']}")
        print(f"  Shannon diversity: {diversity['shannon_diversity']:.3f}")
        print(f"  Simpson diversity: {diversity['simpson_diversity']:.3f}")
        print(f"\n  Top 3 species:")
        for i, row in sample_data.head(3).iterrows():
            print(f"    - {row['species']}: {row['abundance']:.2%} ({row['read_count']} reads)")
    
    print("\n" + "=" * 80)
    print("✓ Sample aggregation example complete!")
    print("=" * 80)
    print("\nTo save results:")
    print("  agg.save_abundance_table(abundance_df, 'abundance.csv')")
    print("  agg.save_abundance_table(abundance_df, 'abundance.xlsx', include_diversity=True)")


if __name__ == "__main__":
    main()


"""
Aggregation module for computing relative abundance from per-read predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


def aggregate_predictions_to_sample(
    predictions_df: pd.DataFrame,
    sample_id_column: str = 'sample_id',
    species_column: str = 'predicted_class',
    confidence_column: str = 'confidence',
    confidence_threshold: float = 0.0,
    tax_id_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate per-read predictions to per-sample species abundance.
    
    Args:
        predictions_df: DataFrame with per-read predictions
        sample_id_column: Column name for sample IDs
        species_column: Column name for predicted species
        confidence_column: Column name for prediction confidence
        confidence_threshold: Minimum confidence to include prediction
        tax_id_column: Column name for tax ID
    
    Returns:
        DataFrame with columns: [Taxon, Species_Name, Prediction, Norm_Prediction]
        (and sample_id if multiple samples)
    """
    # Filter by confidence threshold
    if confidence_threshold > 0:
        predictions_df = predictions_df[predictions_df[confidence_column] >= confidence_threshold]
        print(f"Filtered to {len(predictions_df)} reads with confidence >= {confidence_threshold}")
    
    # Group columns
    group_cols = [sample_id_column, species_column]
    if tax_id_column and tax_id_column in predictions_df.columns:
        group_cols.append(tax_id_column)
    
    # Group by sample and species
    grouped = predictions_df.groupby(group_cols).size().reset_index(name='read_count')
    
    # Calculate total reads per sample
    sample_totals = grouped.groupby(sample_id_column)['read_count'].sum().reset_index()
    sample_totals.columns = [sample_id_column, 'total_reads']
    
    # Merge and calculate abundance
    result = grouped.merge(sample_totals, on=sample_id_column)
    result['abundance'] = result['read_count'] / result['total_reads']
    
    # Sort by sample and abundance
    result = result.sort_values([sample_id_column, 'abundance'], ascending=[True, False])
    
    # Rename columns to match requested format
    # Taxon,Species_Name,Prediction,Norm_Prediction
    
    rename_map = {
        species_column: 'Species_Name',
        'read_count': 'Prediction',
        'abundance': 'Norm_Prediction'
    }
    
    if tax_id_column and tax_id_column in result.columns:
        rename_map[tax_id_column] = 'Taxon'
    else:
        # If no tax_id, use 0 or placeholder? Or maybe species name as Taxon?
        # For now, let's create a placeholder if missing
        result['Taxon'] = 0
    
    result = result.rename(columns=rename_map)
    
    # Select final columns
    final_cols = ['Taxon', 'Species_Name', 'Prediction', 'Norm_Prediction']
    if sample_id_column in result.columns:
        # Keep sample_id for internal use, but maybe not for final output if it's one file per sample?
        # The user example didn't show sample_id.
        # But if we process multiple samples, we need it.
        # Let's keep it for now.
        final_cols = [sample_id_column] + final_cols
        
    result = result[final_cols]
    
    return result


def compute_diversity_metrics(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute diversity metrics (Shannon, Simpson) for each sample.
    
    Args:
        abundance_df: DataFrame with columns [sample_id, species, read_count, abundance]
    
    Returns:
        DataFrame with diversity metrics per sample
    """
    metrics = []
    
    for sample_id in abundance_df['sample_id'].unique():
        sample_data = abundance_df[abundance_df['sample_id'] == sample_id]
        abundances = sample_data['Norm_Prediction'].values
        
        # Shannon diversity
        shannon = -np.sum(abundances * np.log(abundances + 1e-10))
        
        # Simpson diversity
        simpson = 1 - np.sum(abundances ** 2)
        
        # Species richness
        richness = len(sample_data)
        
        metrics.append({
            'sample_id': sample_id,
            'num_species': richness,
            'shannon_diversity': shannon,
            'simpson_diversity': simpson,
            'total_reads': sample_data['Prediction'].sum()
        })
    
    return pd.DataFrame(metrics)


def filter_by_abundance_threshold(
    abundance_df: pd.DataFrame,
    min_abundance: float = 0.01,
    min_read_count: int = 10
) -> pd.DataFrame:
    """
    Filter species by minimum abundance or read count.
    
    Args:
        abundance_df: DataFrame with abundance data
        min_abundance: Minimum relative abundance threshold
        min_read_count: Minimum read count threshold
    
    Returns:
        Filtered DataFrame
    """
    filtered = abundance_df[
        (abundance_df['Norm_Prediction'] >= min_abundance) |
        (abundance_df['Prediction'] >= min_read_count)
    ]
    
    print(f"Filtered from {len(abundance_df)} to {len(filtered)} entries")
    print(f"  Min abundance: {min_abundance}")
    print(f"  Min read count: {min_read_count}")
    
    return filtered


def save_abundance_table(
    abundance_df: pd.DataFrame,
    output_path: Union[str, Path],
    include_diversity: bool = True
):
    """
    Save abundance table with optional diversity metrics.
    
    Args:
        abundance_df: Abundance DataFrame
        output_path: Output file path
        include_diversity: Include diversity metrics sheet (Excel only)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.xlsx' and include_diversity:
        # Save with diversity metrics
        diversity_df = compute_diversity_metrics(abundance_df)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            abundance_df.to_excel(writer, sheet_name='Abundance', index=False)
            diversity_df.to_excel(writer, sheet_name='Diversity', index=False)
        
        print(f"✓ Saved abundance table with diversity metrics to {output_path}")
    else:
        # Save as CSV
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        
        abundance_df.to_csv(output_path, index=False)
        print(f"✓ Saved abundance table to {output_path}")


def create_abundance_matrix(abundance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample x species abundance matrix.
    
    Args:
        abundance_df: Abundance DataFrame
    
    Returns:
        Pivot table with samples as rows, species as columns
    """
    matrix = abundance_df.pivot(
        index='sample_id',
        columns='Species_Name',
        values='Norm_Prediction'
    ).fillna(0)
    
    return matrix


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate per-read predictions to sample-level abundance")
    parser.add_argument("--input", required=True, help="Input predictions CSV")
    parser.add_argument("--output", required=True, help="Output abundance table")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Minimum confidence")
    parser.add_argument("--min_abundance", type=float, default=0.0, help="Minimum abundance threshold")
    parser.add_argument("--min_reads", type=int, default=0, help="Minimum read count")
    parser.add_argument("--sample_column", default="sample_id", help="Sample ID column name")
    parser.add_argument("--species_column", default="predicted_class", help="Species column name")
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.input}")
    predictions_df = pd.read_csv(args.input)
    print(f"Loaded {len(predictions_df)} predictions")
    
    # Aggregate
    print("Aggregating to sample level...")
    abundance_df = aggregate_predictions_to_sample(
        predictions_df,
        sample_id_column=args.sample_column,
        species_column=args.species_column,
        confidence_threshold=args.confidence_threshold
    )
    
    # Filter if thresholds specified
    if args.min_abundance > 0 or args.min_reads > 0:
        abundance_df = filter_by_abundance_threshold(
            abundance_df,
            min_abundance=args.min_abundance,
            min_read_count=args.min_reads
        )
    
    # Save
    save_abundance_table(abundance_df, args.output)
    
    # Print summary
    print("\nSummary:")
    print(f"  Samples: {abundance_df['sample_id'].nunique()}")
    print(f"  Unique species: {abundance_df['Species_Name'].nunique()}")
    print(f"  Total entries: {len(abundance_df)}")
    
    # Print top species per sample
    print("\nTop species per sample:")
    for sample_id in abundance_df['sample_id'].unique():
        sample_data = abundance_df[abundance_df['sample_id'] == sample_id].head(3)
        print(f"\n{sample_id}:")
        for _, row in sample_data.iterrows():
            print(f"  {row['Species_Name']}: {row['Norm_Prediction']:.3f} ({row['Prediction']} reads)")


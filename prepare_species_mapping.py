"""
Convert species mapping from Tax ID format to classification format.
Optionally add species names from the species database CSV.
"""

import pandas as pd
import sys
from pathlib import Path

def load_species_names(species_csv_path):
    """Load species names from the database CSV file."""
    print(f"\nLoading species names from: {species_csv_path}")
    df = pd.read_csv(species_csv_path)
    
    # Create tax_id -> name mapping
    tax_to_name = dict(zip(df['ncbi_taxon_id'], df['name']))
    print(f"✓ Loaded {len(tax_to_name)} species names")
    
    return tax_to_name

def convert_mapping(input_path, output_path, species_csv_path=None):
    """Convert mapping file to required format."""
    
    print(f"Reading mapping from: {input_path}")
    df = pd.read_csv(input_path, sep='\t')
    
    print(f"Original format columns: {df.columns.tolist()}")
    print(f"Number of entries: {len(df)}")
    
    # The input has: Class, Tax ID
    # We need: class_id, label_name, tax_id
    
    # Load species names if provided
    tax_to_name = None
    if species_csv_path and Path(species_csv_path).exists():
        tax_to_name = load_species_names(species_csv_path)
    
    # Create label names
    if tax_to_name:
        print("\n✓ Using species names from database")
        label_names = []
        missing_count = 0
        for tax_id in df['Tax ID']:
            if tax_id in tax_to_name:
                label_names.append(tax_to_name[tax_id])
            else:
                label_names.append(f'TaxID_{tax_id}')
                missing_count += 1
        
        if missing_count > 0:
            print(f"⚠ {missing_count} tax IDs not found in species database, using TaxID_* as fallback")
    else:
        print("\n⚠ No species database provided, using tax IDs as labels")
        label_names = df['Tax ID'].apply(lambda x: f'TaxID_{x}')
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'class_id': df['Class'],
        'label_name': label_names,
        'tax_id': df['Tax ID']
    })
    
    print(f"\nConverted format:")
    print(output_df.head(10))
    print(f"\nSample class IDs: {sorted(output_df['class_id'].unique())[:10]}")
    print(f"Number of classes: {len(output_df)}")
    
    # Show some statistics
    print(f"\nLabel name examples:")
    for i, row in output_df.head(10).iterrows():
        print(f"  Class {row['class_id']:4d} | Tax {row['tax_id']:6d} | {row['label_name']}")
    
    # Save
    output_df.to_csv(output_path, sep='\t', index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return output_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert species mapping for METAGENE training')
    parser.add_argument('--species_csv', type=str, 
                       default='/home/user/Metagenomics/database/selected_disease_results/all_union_species_0305_availability.csv',
                       help='Species database CSV with ncbi_taxon_id and name columns')
    parser.add_argument('--no_names', action='store_true',
                       help='Skip loading species names, use TaxID_* as labels')
    args = parser.parse_args()
    
    input_path = "/media/user/disk2/MetaTransformer_new_pipeline/myScript/all_available_species_mapping.tab"
    output_path = "/media/user/disk2/METAGENE/classification/species_mapping_converted.tsv"
    
    species_csv = None if args.no_names else args.species_csv
    
    df = convert_mapping(input_path, output_path, species_csv)
    
    print("\n" + "="*80)
    print("✓ Mapping file converted successfully!")
    print("="*80)
    print(f"Use this file for training:")
    print(f"  --mapping_tsv {output_path}")
    print("")
    if species_csv:
        print("Note: Species names loaded from database")
    else:
        print("Note: Using TaxID_* as labels (species names not loaded)")


import pandas as pd
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_subset(
    full_mapping_path: str,
    train_fasta_path: str,
    val_fasta_path: str,
    output_dir: str,
    num_species: int = 50,
    seed: int = 42
):
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading mapping from {full_mapping_path}...")
    # Try reading with different encodings/separators if needed
    try:
        df = pd.read_csv(full_mapping_path, sep='\t')
    except:
        df = pd.read_csv(full_mapping_path)
        
    # Get all unique class_ids
    all_classes = df['class_id'].unique().tolist()
    print(f"Found {len(all_classes)} total species.")
    
    # Randomly select subset
    selected_classes = random.sample(all_classes, min(num_species, len(all_classes)))
    selected_classes.sort()
    print(f"Selected {len(selected_classes)} species for diagnosis.")
    
    # Filter mapping
    subset_df = df[df['class_id'].isin(selected_classes)].copy()
    
    # Save subset mapping
    subset_mapping_path = output_dir / "subset_mapping.tsv"
    subset_df.to_csv(subset_mapping_path, sep='\t', index=False)
    print(f"Saved subset mapping to {subset_mapping_path}")
    
    # Create a set for fast lookup
    selected_class_set = set(selected_classes)
    
    def filter_fasta(input_path, output_path, max_reads=None):
        print(f"Filtering {input_path} -> {output_path}...")
        if max_reads:
            print(f"  Limit: {max_reads} reads per species")
            
        count = 0
        class_counts = {cid: 0 for cid in selected_class_set}
        
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            while True:
                header = f_in.readline()
                if not header:
                    break
                
                # Check if header matches selected classes
                # Format: >lbl|class_id|...
                if header.startswith('>'):
                    try:
                        parts = header[1:].split('|')
                        if len(parts) >= 2:
                            class_id = int(parts[1])
                            if class_id in selected_class_set:
                                # Check limit
                                if max_reads is None or class_counts[class_id] < max_reads:
                                    seq = f_in.readline()
                                    f_out.write(header)
                                    f_out.write(seq)
                                    
                                    class_counts[class_id] += 1
                                    count += 1
                                else:
                                    # Skip sequence (limit reached for this class)
                                    f_in.readline()
                            else:
                                # Skip sequence (not in selected classes)
                                f_in.readline()
                        else:
                            f_in.readline()
                    except ValueError:
                        f_in.readline()
                else:
                    # Should not happen if formatted correctly
                    pass
                    
        print(f"Extracted {count} reads.")

    # Filter Train
    filter_fasta(train_fasta_path, output_dir / "subset_train.fa", args.max_reads_train)
    
    # Filter Val
    filter_fasta(val_fasta_path, output_dir / "subset_val.fa", args.max_reads_val)
    
    print("\nSubset preparation complete!")
    print(f"Training data: {output_dir / 'subset_train.fa'}")
    print(f"Validation data: {output_dir / 'subset_val.fa'}")
    print(f"Mapping file: {subset_mapping_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare subset data for diagnosis")
    parser.add_argument("--mapping", required=True, help="Path to full mapping file")
    parser.add_argument("--train_fasta", required=True, help="Path to full training FASTA")
    parser.add_argument("--val_fasta", required=True, help="Path to full validation FASTA")
    parser.add_argument("--output_dir", default="data_diagnosis_50", help="Output directory")
    parser.add_argument("--num_species", type=int, default=50, help="Number of species to select")
    parser.add_argument("--max_reads_train", type=int, default=None, help="Max reads per species for training")
    parser.add_argument("--max_reads_val", type=int, default=None, help="Max reads per species for validation")
    
    args = parser.parse_args()
    
    prepare_subset(
        args.mapping,
        args.train_fasta,
        args.val_fasta,
        args.output_dir,
        args.num_species
    )

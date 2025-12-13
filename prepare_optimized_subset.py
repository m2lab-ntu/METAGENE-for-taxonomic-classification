
import os
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

def create_subset(
    source_dir: str,
    output_dir: str,
    mapping_file: str,
    num_species: int = 100,
    max_reads_per_species: int = 1000
):
    print(f"Creating subset of {num_species} species from {source_dir}...")
    
    # Paths
    source_path = Path(source_dir)
    train_reads_file = source_path / "training_reads" / "training_reads.fa"
    val_reads_file = source_path / "val_reads" / "val_reads.fa"
    output_path = Path(output_dir)
    
    # Clean output dir
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load mapping
    print("Loading species mapping...")
    mapping_df = pd.read_csv(mapping_file, sep='\t')
    
    # Get all class IDs
    all_class_ids = mapping_df['Class'].unique().tolist()
    print(f"Found {len(all_class_ids)} total species/classes.")
    
    # Randomly select subset
    if len(all_class_ids) <= num_species:
        selected_class_ids = set(all_class_ids)
    else:
        selected_class_ids = set(random.sample(all_class_ids, num_species))
        
    print(f"Selected {len(selected_class_ids)} species for diagnosis.")
    
    # Create subset mapping
    subset_mapping_df = mapping_df[mapping_df['Class'].isin(selected_class_ids)]
    # Rename for compatibility with pipeline: Class -> class_id, Species -> label_name
    subset_mapping_df = subset_mapping_df.rename(columns={'Class': 'class_id', 'Species': 'label_name'})
    
    mapping_out_file = output_path / "subset_mapping.tsv"
    subset_mapping_df.to_csv(mapping_out_file, sep='\t', index=False)
    print(f"Saved subset mapping to {mapping_out_file}")
    
    # Extraction Function
    def extract_reads(input_file, output_file, target_class_ids, max_reads=None):
        print(f"Extracting from {input_file}...")
        
        # Track reads per class to limit
        class_counts = {cid: 0 for cid in target_class_ids}
        
        with open(output_file, 'w') as f_out:
            with open(input_file, 'r') as f_in:
                
                # Simple state machine for reading FASTA
                # Given large file, we process line by line
                keep_current_read = False
                current_header = ""
                
                # Check file size for progress bar
                file_size = os.path.getsize(input_file)
                pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=output_file.name)
                
                for line in f_in:
                    pbar.update(len(line))
                    
                    if line.startswith('>'):
                        # Process previous read (already written if keep_current_read was True)
                        # Actually, we should write line-by-line if keep_current_read is True.
                        
                        # Parse new header
                        # >lbl|class_id|tax_id|...
                        try:
                            parts = line.split('|')
                            if len(parts) > 1:
                                class_id = int(parts[1])
                                
                                if class_id in target_class_ids:
                                    if max_reads is None or class_counts[class_id] < max_reads:
                                        keep_current_read = True
                                        class_counts[class_id] += 1
                                        f_out.write(line)
                                    else:
                                        keep_current_read = False
                                else:
                                    keep_current_read = False
                            else:
                                keep_current_read = False
                        except ValueError:
                            keep_current_read = False
                            
                    else:
                        if keep_current_read:
                            f_out.write(line)
                            
                pbar.close()
                
        # Report
        total_extracted = sum(class_counts.values())
        print(f"Extracted {total_extracted} reads.")
        return class_counts

    # Extract Train
    train_out_file = output_path / "subset_train.fa"
    train_counts = extract_reads(train_reads_file, train_out_file, selected_class_ids, max_reads=max_reads_per_species)
    
    # Extract Val
    # For val, we might want fewer reads, e.g. 200 max
    val_out_file = output_path / "subset_val.fa"
    val_counts = extract_reads(val_reads_file, val_out_file, selected_class_ids, max_reads=200)
    
    print(f"✓ Created Dataset in {output_dir}")

if __name__ == "__main__":
    create_subset(
        source_dir="/media/user/disk2/MetaTransformer_original_dataset",
        output_dir="data_subset_100",
        mapping_file="/media/user/disk2/MetaTransformer_original_dataset/species_mapping.tab",
        num_species=100
    )

import os
import random
from pathlib import Path
from Bio import SeqIO
import pandas as pd

def generate_reads(
    input_dir: str,
    output_fasta: str,
    mapping_output: str,
    num_reads_per_species: int = 100,
    read_length: int = 150
):
    """
    Generate synthetic reads from reference genomes.
    Header format: >lbl|class|tax_id|genus|species_name/pair_end
    """
    print(f"Scanning {input_dir} for genomes...")
    
    # Find all fasta files
    genome_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.fasta', '.fa', '.fna')) and 'MACOSX' not in root:
                genome_files.append(os.path.join(root, file))
    
    # Sort for deterministic order
    genome_files.sort()
    
    print(f"Found {len(genome_files)} genomes")
    
    # Create mapping
    mapping_data = []
    
    with open(output_fasta, 'w') as f_out:
        for class_id, genome_path in enumerate(genome_files):
            # Parse species name from filename or header
            # Zymo filenames are usually like "Lactobacillus_fermentum.fasta"
            filename = Path(genome_path).stem
            species_name = filename.replace('_', ' ').replace('ZymoBIOMICS STD refseq v3 ', '')
            
            # Mock tax_id (just use class_id + 1000 for testing)
            tax_id = 1000 + class_id
            
            # Genus is first part
            genus = species_name.split()[0]
            
            mapping_data.append({
                'class_id': class_id,
                'label_name': species_name,
                'tax_id': tax_id
            })
            
            print(f"Processing {species_name}...")
            
            # Read genome
            genome_seqs = []
            for record in SeqIO.parse(genome_path, 'fasta'):
                genome_seqs.append(str(record.seq))
            
            full_genome = "".join(genome_seqs)
            genome_len = len(full_genome)
            
            if genome_len < read_length:
                print(f"Warning: Genome too short for {species_name}, skipping")
                continue
                
            # Generate reads
            for i in range(num_reads_per_species):
                # Random start position
                start = random.randint(0, genome_len - read_length)
                seq = full_genome[start : start + read_length]
                
                # Write header
                # >lbl|class|tax_id|genus|species_name/pair_end
                header = f">lbl|{class_id}|{tax_id}|{genus}|{species_name}/1"
                f_out.write(f"{header}\n{seq}\n")
                
    # Save mapping
    df = pd.DataFrame(mapping_data)
    df.to_csv(mapping_output, sep='\t', index=False)
    print(f"Saved mapping to {mapping_output}")
    print(f"Saved reads to {output_fasta}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic Zymo reads")
    parser.add_argument("--input_dir", default="zymo_refs", help="Input directory containing reference genomes")
    parser.add_argument("--output_fasta", required=True, help="Output FASTA file")
    parser.add_argument("--mapping_output", required=True, help="Output mapping TSV file")
    parser.add_argument("--num_reads", type=int, default=200, help="Number of reads per species")
    parser.add_argument("--read_length", type=int, default=150, help="Read length")
    
    args = parser.parse_args()
    
    generate_reads(
        input_dir=args.input_dir,
        output_fasta=args.output_fasta,
        mapping_output=args.mapping_output,
        num_reads_per_species=args.num_reads,
        read_length=args.read_length
    )

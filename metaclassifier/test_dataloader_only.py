import sys
import pandas as pd
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Mock Bio
import sys
from unittest.mock import MagicMock
mock_bio = MagicMock()
sys.modules['Bio'] = mock_bio
sys.modules['Bio.SeqIO'] = mock_bio.SeqIO

# Mock SeqIO.parse
def mock_parse(path, format):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    records = []
    current_header = None
    current_seq = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_header:
                record = MagicMock()
                record.id = current_header[1:]
                record.seq = "".join(current_seq)
                records.append(record)
            current_header = line
            current_seq = []
        else:
            current_seq.append(line)
            
    if current_header:
        record = MagicMock()
        record.id = current_header[1:]
        record.seq = "".join(current_seq)
        records.append(record)
        
    return records

mock_bio.SeqIO.parse = mock_parse

from dataset_optimized import InMemoryDataset, IndexedFastaDataset

class MockTokenizer:
    def encode(self, text):
        return [1, 2, 3]
    def pad_and_truncate(self, tokens):
        return tokens
    def create_attention_mask(self, tokens):
        return [1, 1, 1]

def create_mock_fasta(path):
    with open(path, 'w') as f:
        # >lbl|class|tax_id|genus|species_name/pair_end
        f.write(">lbl|0|7|Azorhizobium|Azorhizobium caulinodans/1\n")
        f.write("ACGT\n")
        f.write(">lbl|1|9|Buchnera|Buchnera aphidicola/1\n")
        f.write("TGCA\n")
        f.write(">lbl|0|7|Azorhizobium|Azorhizobium caulinodans/2\n")
        f.write("ACGT\n")

def create_mock_mapping(path):
    df = pd.DataFrame({
        'class_id': [0, 1],
        'label_name': ['Azorhizobium caulinodans', 'Buchnera aphidicola'],
        'tax_id': [7, 9]
    })
    df.to_csv(path, sep='\t', index=False)
    return df

def test_dataloader():
    print("Testing dataloader...")
    
    # Create mock files
    fasta_path = "test_data.fa"
    mapping_path = "test_mapping.tsv"
    
    create_mock_fasta(fasta_path)
    mapping_df = create_mock_mapping(mapping_path)
    
    tokenizer = MockTokenizer()
    
    # Test InMemoryDataset
    print("\nTesting InMemoryDataset:")
    dataset = InMemoryDataset(fasta_path, mapping_df, tokenizer, 100)
    print(f"Loaded {len(dataset)} sequences")
    
    if len(dataset) == 3:
        print("✓ Count correct")
    else:
        print(f"✗ Count incorrect: expected 3, got {len(dataset)}")
        
    # Check labels
    item0 = dataset[0]
    print(f"Item 0 label: {item0['labels']}")
    
    # Test IndexedFastaDataset
    print("\nTesting IndexedFastaDataset:")
    dataset_idx = IndexedFastaDataset(fasta_path, mapping_df, tokenizer, 100, index_cache_dir=".")
    print(f"Loaded {len(dataset_idx)} sequences")
    
    if len(dataset_idx) == 3:
        print("✓ Count correct")
    else:
        print(f"✗ Count incorrect: expected 3, got {len(dataset_idx)}")

    # Cleanup
    Path(fasta_path).unlink()
    Path(mapping_path).unlink()
    Path("test_data_index.pkl").unlink(missing_ok=True)

if __name__ == "__main__":
    test_dataloader()

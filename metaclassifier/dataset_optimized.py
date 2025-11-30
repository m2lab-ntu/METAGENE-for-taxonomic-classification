"""
Optimized dataset loader for large FASTA files.
Uses lazy loading to avoid loading entire file into memory.
"""

import torch
from torch.utils.data import IterableDataset, Dataset
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import pickle


class IndexedFastaDataset(Dataset):
    """
    Memory-efficient dataset that creates an index file and loads sequences on-demand.
    Only suitable for random access (not streaming).
    """
    
    def __init__(self, fasta_path: str, mapping_df: pd.DataFrame, tokenizer, max_length: int, 
                 index_cache_dir: str = None):
        """
        Args:
            fasta_path: Path to FASTA file
            mapping_df: DataFrame with 'class_id' and 'label_name' columns
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            index_cache_dir: Directory to cache index file (default: same as FASTA)
        """
        self.fasta_path = fasta_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df['label_name'].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Build or load index
        if index_cache_dir is None:
            index_cache_dir = Path(fasta_path).parent
        self.index_path = Path(index_cache_dir) / f"{Path(fasta_path).stem}_index.pkl"
        
        if self.index_path.exists():
            print(f"Loading existing index from {self.index_path}")
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            print(f"Loaded index with {len(self.index)} sequences")
        else:
            print(f"Building index for {fasta_path}...")
            self.index = self._build_index(mapping_df)
            print(f"Saving index to {self.index_path}")
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.index, f)
            print(f"Index saved with {len(self.index)} sequences")
    
    def _build_index(self, mapping_df):
        """Build an index of sequence positions in the FASTA file."""
        from tqdm import tqdm
        
        index = []
        with open(self.fasta_path, 'r') as f:
            while True:
                pos = f.tell()
                header_line = f.readline()
                if not header_line:
                    break
                
                if header_line.startswith('>'):
                    # Extract class_id from header
                    # Format: >lbl|class|tax_id|genus|species_name/pair_end
                    record_id = header_line[1:].strip().split()[0]
                    parts = record_id.split('|')
                    
                    if len(parts) >= 5:
                        try:
                            # class is at index 1
                            class_id = int(parts[1])
                            
                            # We can also extract other metadata if needed
                            # tax_id = parts[2]
                            # species_name = parts[4].split('/')[0]
                            
                            label_row = mapping_df[mapping_df['class_id'] == class_id]
                            
                            if not label_row.empty:
                                label = label_row.iloc[0]['label_name']
                                label_id = self.label_to_id[label]
                                
                                # Read sequence line
                                seq_pos = f.tell()
                                seq_line = f.readline()
                                seq_length = len(seq_line.strip())
                                
                                index.append({
                                    'header_pos': pos,
                                    'seq_pos': seq_pos,
                                    'seq_length': seq_length,
                                    'label_id': label_id,
                                    'class_id': class_id
                                })
                            else:
                                # Skip sequence line
                                f.readline()
                        except (ValueError, IndexError):
                            # Skip sequence line
                            f.readline()
                    else:
                        # Skip sequence line
                        f.readline()
                
                # Show progress every 10M sequences
                if len(index) % 10000000 == 0 and len(index) > 0:
                    print(f"  Indexed {len(index):,} sequences...")
        
        return index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """Load sequence on-demand from file."""
        entry = self.index[idx]
        
        # Read sequence from file
        with open(self.fasta_path, 'r') as f:
            f.seek(entry['seq_pos'])
            sequence = f.readline().strip()
        
        label = entry['label_id']
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        tokens = self.tokenizer.pad_and_truncate(tokens)
        attention_mask = self.tokenizer.create_attention_mask(tokens)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class StreamingFastaDataset(IterableDataset):
    """
    Streaming dataset that reads FASTA file sequentially.
    Memory efficient but only supports sequential access (no shuffling).
    """
    
    def __init__(self, fasta_path: str, mapping_df: pd.DataFrame, tokenizer, max_length: int):
        """
        Args:
            fasta_path: Path to FASTA file
            mapping_df: DataFrame with 'class_id' and 'label_name' columns
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.fasta_path = fasta_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mapping_df = mapping_df
        
        # Create label mapping
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df['label_name'].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Count total sequences (for progress tracking)
        self.total_sequences = self._count_sequences()
        print(f"Dataset contains {self.total_sequences:,} sequences")
    
    def _count_sequences(self):
        """Quick count of sequences in file."""
        count = 0
        with open(self.fasta_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    # Format: >lbl|class|tax_id|genus|species_name/pair_end
                    parts = line[1:].strip().split()[0].split('|')
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[1])
                            label_row = self.mapping_df[self.mapping_df['class_id'] == class_id]
                            if not label_row.empty:
                                count += 1
                        except (ValueError, IndexError):
                            pass
        return count
    
    def __iter__(self):
        """Iterate through sequences in the file."""
        for record in SeqIO.parse(self.fasta_path, 'fasta'):
            # Format: lbl|class|tax_id|genus|species_name/pair_end
            parts = record.id.split('|')
            if len(parts) >= 5:
                try:
                    class_id = int(parts[1])
                    label_row = self.mapping_df[self.mapping_df['class_id'] == class_id]
                    
                    if not label_row.empty:
                        label = label_row.iloc[0]['label_name']
                        label_id = self.label_to_id[label]
                        
                        sequence = str(record.seq)
                        
                        # Tokenize
                        tokens = self.tokenizer.encode(sequence)
                        tokens = self.tokenizer.pad_and_truncate(tokens)
                        attention_mask = self.tokenizer.create_attention_mask(tokens)
                        
                        yield {
                            'input_ids': torch.tensor(tokens, dtype=torch.long),
                            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                            'labels': torch.tensor(label_id, dtype=torch.long)
                        }
                except (ValueError, IndexError):
                    continue
    
    def __len__(self):
        return self.total_sequences


class InMemoryDataset(Dataset):
    """
    Dataset that loads all sequences into memory for maximum speed.
    Suitable when dataset fits in RAM (e.g. subsets).
    """
    
    def __init__(self, fasta_path: str, mapping_df: pd.DataFrame, tokenizer, max_length: int):
        """
        Args:
            fasta_path: Path to FASTA file
            mapping_df: DataFrame with 'class_id' and 'label_name' columns
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df['label_name'].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Load all sequences
        print(f"Loading {fasta_path} into memory...")
        self.sequences = []
        self.labels = []
        
        with open(fasta_path, 'r') as f:
            current_header = None
            
            for line in f:
                if line.startswith('>'):
                    current_header = line
                else:
                    if current_header:
                        # Extract class_id
                        # Format: >lbl|class|tax_id|genus|species_name/pair_end
                        parts = current_header[1:].strip().split()[0].split('|')
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[1])
                                label_row = mapping_df[mapping_df['class_id'] == class_id]
                                
                                if not label_row.empty:
                                    label = label_row.iloc[0]['label_name']
                                    label_id = self.label_to_id[label]
                                    
                                    sequence = line.strip()
                                    self.sequences.append(sequence)
                                    self.labels.append(label_id)
                            except (ValueError, IndexError):
                                pass
                        current_header = None
                        
        print(f"Loaded {len(self.sequences):,} sequences into memory")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        tokens = self.tokenizer.pad_and_truncate(tokens)
        attention_mask = self.tokenizer.create_attention_mask(tokens)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


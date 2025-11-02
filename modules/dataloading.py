"""
Data loading utilities for DNA/RNA sequence classification.
Handles FASTA/FASTQ parsing, header extraction, and tokenization.
"""

import gzip
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import tokenizer from minbpe
import sys
import os
sys.path.append('/media/user/disk2/METAGENE/metagene-pretrain/train')
from minbpe.minbpe import RegexTokenizer


class SequenceDataset(Dataset):
    """Dataset for DNA/RNA sequences with header parsing and label mapping."""
    
    def __init__(
        self,
        fasta_path: Union[str, Path],
        mapping_df: pd.DataFrame,
        tokenizer: Any,
        header_regex: str,
        max_length: int = 512,
        class_column: str = "class_id",
        label_column: str = "label_name",
        strict_classes: bool = True
    ):
        self.fasta_path = Path(fasta_path)
        self.mapping_df = mapping_df
        self.tokenizer = tokenizer
        self.header_regex = re.compile(header_regex)
        self.max_length = max_length
        self.class_column = class_column
        self.label_column = label_column
        self.strict_classes = strict_classes
        
        # Build mappings
        self.class_to_label = dict(zip(mapping_df[class_column], mapping_df[label_column]))
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df[label_column].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.num_classes = len(self.label_to_id)
        
        # Load sequences
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences with {self.num_classes} classes")
        print(f"Classes: {list(self.label_to_id.keys())}")
    
    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load sequences from FASTA/FASTQ file."""
        sequences = []
        
        # Detect file format and compression
        is_gzipped = self._is_gzipped()
        file_format = self._detect_format()
        
        print(f"Loading sequences from {self.fasta_path}")
        print(f"Format: {file_format}, Gzipped: {is_gzipped}")
        
        # Open file
        if is_gzipped:
            file_handle = gzip.open(self.fasta_path, 'rt')
        else:
            file_handle = open(self.fasta_path, 'r')
        
        try:
            for record in tqdm(SeqIO.parse(file_handle, file_format), desc="Loading sequences"):
                # Parse header (BioPython record.id doesn't include the > symbol)
                header_match = self.header_regex.match(record.id)
                if not header_match:
                    print(f"Warning: Could not parse header: {record.id}")
                    continue
                
                class_id = int(header_match.group('class_id'))
                
                # Check if class exists in mapping
                if class_id not in self.class_to_label:
                    if self.strict_classes:
                        raise ValueError(f"Class ID {class_id} not found in mapping")
                    else:
                        print(f"Warning: Class ID {class_id} not found in mapping, skipping")
                        continue
                
                label = self.class_to_label[class_id]
                label_id = self.label_to_id[label]
                
                sequences.append({
                    'id': record.id,
                    'sequence': str(record.seq),
                    'class_id': class_id,
                    'label': label,
                    'label_id': label_id,
                    'length': len(record.seq)
                })
        
        finally:
            file_handle.close()
        
        return sequences
    
    def _is_gzipped(self) -> bool:
        """Check if file is gzipped."""
        with open(self.fasta_path, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    
    def _detect_format(self) -> str:
        """Detect file format (fasta or fastq)."""
        with open(self.fasta_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('@'):
                return 'fastq'
            elif first_line.startswith('>'):
                return 'fasta'
            else:
                raise ValueError(f"Unknown file format. First line: {first_line}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sequence and its label."""
        seq_data = self.sequences[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(seq_data['sequence'])
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(seq_data['label_id'], dtype=torch.long),
            'sequence_id': seq_data['id'],
            'class_id': seq_data['class_id'],
            'label': seq_data['label'],
            'length': seq_data['length']
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        from collections import Counter
        return dict(Counter(seq['label'] for seq in self.sequences))


class MetaGeneTokenizer:
    """Wrapper for METAGENE BPE tokenizer to be compatible with HuggingFace format."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512, use_hf_tokenizer: bool = False):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_path: Path to minbpe tokenizer OR HuggingFace model name
            max_length: Maximum sequence length
            use_hf_tokenizer: If True, try to use HuggingFace AutoTokenizer instead
        """
        self.max_length = max_length
        self.use_hf = False
        
        if use_hf_tokenizer or tokenizer_path.startswith("metagene-ai"):
            # Try to use HuggingFace official tokenizer
            try:
                from transformers import AutoTokenizer
                print(f"Attempting to load HuggingFace tokenizer from {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True
                )
                self.use_hf = True
                print("✓ Using HuggingFace official tokenizer")
            except Exception as e:
                print(f"Warning: Could not load HuggingFace tokenizer: {e}")
                print(f"Falling back to minbpe tokenizer")
                self.use_hf = False
        
        if not self.use_hf:
            # Use minbpe tokenizer
            self.tokenizer = RegexTokenizer()
            self.tokenizer.load(tokenizer_path)
            
            # Set special tokens
            self.pad_token_id = self.tokenizer.vocab_size - 1
            self.bos_token_id = self.tokenizer.vocab_size - 1
            self.eos_token_id = self.tokenizer.vocab_size - 1
            
            # Add pad token to vocab
            if not hasattr(self.tokenizer, 'pad_token_id'):
                self.tokenizer.pad_token_id = self.pad_token_id
        else:
            # HuggingFace tokenizer
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else 0
            self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.use_hf:
            # Don't add special tokens for DNA sequences
            return self.tokenizer(text, add_special_tokens=False)['input_ids']
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        if self.use_hf:
            return len(self.tokenizer)
        else:
            return self.tokenizer.vocab_size


def load_mapping_tsv(mapping_path: Union[str, Path]) -> pd.DataFrame:
    """Load mapping TSV file."""
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    # Try different separators
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(mapping_path, sep=sep)
            if len(df.columns) >= 2:  # At least class_id and label_name
                print(f"Loaded mapping with {len(df)} entries using separator '{sep}'")
                return df
        except Exception as e:
            continue
    
    raise ValueError(f"Could not parse mapping file: {mapping_path}")


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with proper collation."""
    
    def collate_fn(batch):
        """Collate function for batching."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'metadata': {
                'sequence_ids': [item['sequence_id'] for item in batch],
                'class_ids': [item['class_id'] for item in batch],
                'labels': [item['label'] for item in batch],
                'lengths': [item['length'] for item in batch]
            }
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def get_sequence_statistics(dataset: SequenceDataset) -> Dict[str, Any]:
    """Get statistics about the dataset."""
    lengths = [seq['length'] for seq in dataset.sequences]
    class_counts = dataset.get_class_distribution()
    
    return {
        'num_sequences': len(dataset),
        'num_classes': dataset.num_classes,
        'sequence_lengths': {
            'min': min(lengths),
            'max': max(lengths),
            'mean': sum(lengths) / len(lengths),
            'median': sorted(lengths)[len(lengths) // 2]
        },
        'class_distribution': class_counts,
        'avg_sequences_per_class': len(dataset) / dataset.num_classes
    }


def save_class_distribution(dataset: SequenceDataset, output_path: Union[str, Path]) -> None:
    """Save class distribution to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    class_counts = dataset.get_class_distribution()
    df = pd.DataFrame([
        {'class': cls, 'count': count, 'percentage': count / len(dataset) * 100}
        for cls, count in class_counts.items()
    ])
    
    df.to_csv(output_path, index=False)
    print(f"Saved class distribution to {output_path}")

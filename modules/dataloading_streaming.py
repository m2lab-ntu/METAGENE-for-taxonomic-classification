"""
Streaming data loading utilities for large DNA/RNA sequence datasets.
This version does not load all sequences into memory, enabling training on datasets >100GB.
"""

import gzip
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset
from tqdm import tqdm


class StreamingSequenceDataset(Dataset):
    """
    Memory-efficient streaming dataset for DNA/RNA sequences.
    
    Instead of loading all sequences into memory, this dataset:
    1. Creates an index of sequence positions in the file
    2. Loads sequences on-demand during training
    
    This enables training on datasets that don't fit in RAM.
    """
    
    def __init__(
        self,
        fasta_path: Union[str, Path],
        mapping_df: pd.DataFrame,
        tokenizer: Any,
        header_regex: str,
        max_length: int = 512,
        class_column: str = "class_id",
        label_column: str = "label_name",
        strict_classes: bool = True,
        cache_index: bool = True
    ):
        self.fasta_path = Path(fasta_path)
        self.mapping_df = mapping_df
        self.tokenizer = tokenizer
        self.header_regex = re.compile(header_regex)
        self.max_length = max_length
        self.class_column = class_column
        self.label_column = label_column
        self.strict_classes = strict_classes
        self.cache_index = cache_index
        
        # Build mappings
        self.class_to_label = dict(zip(mapping_df[class_column], mapping_df[label_column]))
        self.label_to_id = {label: idx for idx, label in enumerate(sorted(mapping_df[label_column].unique()))}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.num_classes = len(self.label_to_id)
        
        # Detect file properties
        self.is_gzipped = self._is_gzipped()
        self.file_format = self._detect_format()
        
        # Build or load sequence index
        self.index = self._build_or_load_index()
        
        print(f"Streaming dataset ready: {len(self.index)} sequences with {self.num_classes} classes")
    
    def _is_gzipped(self) -> bool:
        """Detect if file is gzipped."""
        return self.fasta_path.suffix == '.gz'
    
    def _detect_format(self) -> str:
        """Detect if FASTA or FASTQ."""
        suffix = self.fasta_path.suffixes[-2] if self.is_gzipped else self.fasta_path.suffix
        return "fastq" if suffix in ['.fastq', '.fq'] else "fasta"
    
    def _get_index_path(self) -> Path:
        """Get path for cached index file."""
        return self.fasta_path.with_suffix(self.fasta_path.suffix + '.index.pkl')
    
    def _build_or_load_index(self) -> List[Dict[str, Any]]:
        """Build sequence index or load from cache."""
        index_path = self._get_index_path()
        
        # Try to load cached index
        if self.cache_index and index_path.exists():
            print(f"Loading cached index from {index_path}")
            try:
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)
                print(f"✓ Loaded index with {len(index)} sequences")
                return index
            except Exception as e:
                print(f"Warning: Failed to load cached index: {e}")
                print("Building new index...")
        
        # Build new index
        index = self._build_index()
        
        # Cache index if enabled
        if self.cache_index:
            print(f"Caching index to {index_path}")
            try:
                with open(index_path, 'wb') as f:
                    pickle.dump(index, f)
                print("✓ Index cached")
            except Exception as e:
                print(f"Warning: Failed to cache index: {e}")
        
        return index
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """Build index of sequence positions in file."""
        index = []
        
        print(f"Building sequence index from {self.fasta_path}")
        print(f"Format: {self.file_format}, Gzipped: {self.is_gzipped}")
        print("This may take a few minutes for large files...")
        
        # For uncompressed files, we can store file positions
        # For gzipped files, we must store sequence IDs only (random access is expensive)
        if not self.is_gzipped:
            index = self._build_seekable_index()
        else:
            index = self._build_sequential_index()
        
        return index
    
    def _build_seekable_index(self) -> List[Dict[str, Any]]:
        """Build index with file positions for uncompressed files (fast random access)."""
        index = []
        
        with open(self.fasta_path, 'r') as f:
            position = f.tell()
            
            for record in tqdm(SeqIO.parse(f, self.file_format), desc="Indexing sequences"):
                header_match = self.header_regex.match(record.id)
                if not header_match:
                    print(f"Warning: Could not parse header: {record.id}")
                    position = f.tell()
                    continue
                
                class_id = int(header_match.group('class_id'))
                
                if class_id not in self.class_to_label:
                    if self.strict_classes:
                        raise ValueError(f"Class ID {class_id} not found in mapping")
                    position = f.tell()
                    continue
                
                label = self.class_to_label[class_id]
                label_id = self.label_to_id[label]
                
                index.append({
                    'file_position': position,
                    'id': record.id,
                    'class_id': class_id,
                    'label_id': label_id,
                    'label': label,
                    'metadata': {k: v for k, v in header_match.groupdict().items()}
                })
                
                position = f.tell()
        
        return index
    
    def _build_sequential_index(self) -> List[Dict[str, Any]]:
        """Build index for compressed files (sequential access only)."""
        index = []
        
        with gzip.open(self.fasta_path, 'rt') as f:
            seq_idx = 0
            
            for record in tqdm(SeqIO.parse(f, self.file_format), desc="Indexing sequences"):
                header_match = self.header_regex.match(record.id)
                if not header_match:
                    print(f"Warning: Could not parse header: {record.id}")
                    continue
                
                class_id = int(header_match.group('class_id'))
                
                if class_id not in self.class_to_label:
                    if self.strict_classes:
                        raise ValueError(f"Class ID {class_id} not found in mapping")
                    continue
                
                label = self.class_to_label[class_id]
                label_id = self.label_to_id[label]
                
                index.append({
                    'sequence_index': seq_idx,
                    'id': record.id,
                    'class_id': class_id,
                    'label_id': label_id,
                    'label': label,
                    'metadata': {k: v for k, v in header_match.groupdict().items()}
                })
                
                seq_idx += 1
        
        # For gzipped files, we'll need to cache sequences on first pass
        print("Note: Gzipped files require sequential reading. Consider decompressing for faster training.")
        
        return index
    
    def _load_sequence_by_position(self, file_position: int) -> str:
        """Load sequence by file position (for uncompressed files)."""
        with open(self.fasta_path, 'r') as f:
            f.seek(file_position)
            record = next(SeqIO.parse(f, self.file_format))
            return str(record.seq)
    
    def _load_sequence_by_index(self, sequence_index: int) -> str:
        """Load sequence by sequential index (for compressed files)."""
        # This is slow - for production, consider decompressing first
        if self.is_gzipped:
            with gzip.open(self.fasta_path, 'rt') as f:
                for i, record in enumerate(SeqIO.parse(f, self.file_format)):
                    if i == sequence_index:
                        return str(record.seq)
        else:
            with open(self.fasta_path, 'r') as f:
                for i, record in enumerate(SeqIO.parse(f, self.file_format)):
                    if i == sequence_index:
                        return str(record.seq)
        
        raise IndexError(f"Sequence index {sequence_index} not found")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and tokenize sequence on-demand."""
        entry = self.index[idx]
        
        # Load sequence from file
        if 'file_position' in entry:
            sequence = self._load_sequence_by_position(entry['file_position'])
        else:
            sequence = self._load_sequence_by_index(entry['sequence_index'])
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(entry['label_id'], dtype=torch.long),
            'metadata': entry['metadata']
        }


class CachedStreamingDataset(StreamingSequenceDataset):
    """
    Streaming dataset with in-memory caching of recently accessed sequences.
    
    This provides a middle ground between full memory loading and pure streaming:
    - Recently accessed sequences are cached in memory
    - Cache size is configurable
    - Uses LRU (Least Recently Used) eviction
    """
    
    def __init__(
        self,
        fasta_path: Union[str, Path],
        mapping_df: pd.DataFrame,
        tokenizer: Any,
        header_regex: str,
        max_length: int = 512,
        class_column: str = "class_id",
        label_column: str = "label_name",
        strict_classes: bool = True,
        cache_index: bool = True,
        cache_size: int = 10000  # Number of sequences to cache
    ):
        super().__init__(
            fasta_path=fasta_path,
            mapping_df=mapping_df,
            tokenizer=tokenizer,
            header_regex=header_regex,
            max_length=max_length,
            class_column=class_column,
            label_column=label_column,
            strict_classes=strict_classes,
            cache_index=cache_index
        )
        
        self.cache_size = cache_size
        self.sequence_cache = {}
        self.cache_order = []
        
        print(f"LRU cache enabled with size: {cache_size} sequences")
    
    def _get_sequence(self, idx: int) -> str:
        """Get sequence with LRU caching."""
        if idx in self.sequence_cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.sequence_cache[idx]
        
        # Load sequence
        entry = self.index[idx]
        if 'file_position' in entry:
            sequence = self._load_sequence_by_position(entry['file_position'])
        else:
            sequence = self._load_sequence_by_index(entry['sequence_index'])
        
        # Add to cache
        self.sequence_cache[idx] = sequence
        self.cache_order.append(idx)
        
        # Evict if cache is full
        if len(self.sequence_cache) > self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.sequence_cache[oldest_idx]
        
        return sequence
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and tokenize sequence with caching."""
        entry = self.index[idx]
        
        # Get sequence (from cache or file)
        sequence = self._get_sequence(idx)
        
        # Tokenize
        tokens = self.tokenizer.encode(sequence)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(entry['label_id'], dtype=torch.long),
            'metadata': entry['metadata']
        }


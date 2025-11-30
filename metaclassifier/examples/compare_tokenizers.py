"""
Compare different tokenizers on the same sequence.

Demonstrates:
- BPE tokenization
- K-mer tokenization
- Single-nucleotide tokenization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenization import BPETokenizer, KmerTokenizer, Evo2Tokenizer


def main():
    print("=" * 80)
    print("Tokenizer Comparison")
    print("=" * 80)
    
    # Test sequence
    sequence = "ATCGATCGATCGATCGATCG"
    print(f"\nTest sequence ({len(sequence)} bp):")
    print(f"  {sequence}")
    
    # 1. BPE Tokenizer
    print("\n" + "-" * 80)
    print("1. BPE Tokenizer (METAGENE-1 style)")
    print("-" * 80)
    
    bpe_tokenizer = BPETokenizer(
        tokenizer_path="metagene-ai/METAGENE-1",
        max_length=192,
        use_hf_tokenizer=True
    )
    
    bpe_tokens = bpe_tokenizer.tokenize(sequence)
    bpe_token_ids = bpe_tokenizer.encode(sequence)
    
    print(f"Vocab size: {bpe_tokenizer.get_vocab_size()}")
    print(f"Token count: {len(bpe_token_ids)}")
    print(f"Tokens (first 10): {bpe_tokens[:10]}")
    print(f"Token IDs (first 10): {bpe_token_ids[:10]}")
    print(f"Compression ratio: {len(sequence) / len(bpe_token_ids):.2f}x")
    
    # 2. K-mer Tokenizer (6-mer, overlapping)
    print("\n" + "-" * 80)
    print("2. K-mer Tokenizer (6-mer, overlapping)")
    print("-" * 80)
    
    kmer_tokenizer = KmerTokenizer(
        k=6,
        max_length=512,
        overlap=True,
        stride=1
    )
    
    kmer_tokens = kmer_tokenizer.tokenize(sequence)
    kmer_token_ids = kmer_tokenizer.encode(sequence)
    
    print(f"Vocab size: {kmer_tokenizer.get_vocab_size()}")
    print(f"Token count: {len(kmer_token_ids)}")
    print(f"Tokens: {kmer_tokens}")
    print(f"Token IDs (first 10): {kmer_token_ids[:10]}")
    print(f"Compression ratio: {len(sequence) / len(kmer_token_ids):.2f}x")
    
    # 3. K-mer Tokenizer (6-mer, non-overlapping)
    print("\n" + "-" * 80)
    print("3. K-mer Tokenizer (6-mer, non-overlapping)")
    print("-" * 80)
    
    kmer_nonoverlap = KmerTokenizer(
        k=6,
        max_length=512,
        overlap=False
    )
    
    kmer_no_tokens = kmer_nonoverlap.tokenize(sequence)
    kmer_no_token_ids = kmer_nonoverlap.encode(sequence)
    
    print(f"Vocab size: {kmer_nonoverlap.get_vocab_size()}")
    print(f"Token count: {len(kmer_no_token_ids)}")
    print(f"Tokens: {kmer_no_tokens}")
    print(f"Token IDs: {kmer_no_token_ids}")
    print(f"Compression ratio: {len(sequence) / len(kmer_no_token_ids):.2f}x")
    
    # 4. Evo2 Tokenizer (single-nucleotide)
    print("\n" + "-" * 80)
    print("4. Evo2 Tokenizer (single-nucleotide)")
    print("-" * 80)
    
    evo2_tokenizer = Evo2Tokenizer(max_length=8192)
    
    evo2_tokens = evo2_tokenizer.tokenize(sequence)
    evo2_token_ids = evo2_tokenizer.encode(sequence)
    
    print(f"Vocab size: {evo2_tokenizer.get_vocab_size()}")
    print(f"Token count: {len(evo2_token_ids)}")
    print(f"Tokens: {evo2_tokens}")
    print(f"Token IDs: {evo2_token_ids}")
    print(f"Compression ratio: {len(sequence) / len(evo2_token_ids):.2f}x")
    
    # Decode test
    decoded = evo2_tokenizer.decode(evo2_token_ids)
    print(f"Decoded: {decoded}")
    print(f"Match original: {decoded == sequence}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n{'Tokenizer':<30} {'Tokens':<10} {'Compression':<15} {'Best For'}")
    print("-" * 80)
    print(f"{'BPE (METAGENE-1)':<30} {len(bpe_token_ids):<10} {len(sequence)/len(bpe_token_ids):.2f}x{'':>11} Short reads, fast")
    print(f"{'K-mer (6, overlap)':<30} {len(kmer_token_ids):<10} {len(sequence)/len(kmer_token_ids):.2f}x{'':>11} Context-aware")
    print(f"{'K-mer (6, no overlap)':<30} {len(kmer_no_token_ids):<10} {len(sequence)/len(kmer_no_token_ids):.2f}x{'':>11} Non-overlapping")
    print(f"{'Evo2 (single-nuc)':<30} {len(evo2_token_ids):<10} {len(sequence)/len(evo2_token_ids):.2f}x{'':>11} Long reads, exact")
    
    print("\n" + "=" * 80)
    print("✓ Tokenizer comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


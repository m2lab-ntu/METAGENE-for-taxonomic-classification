"""
Download METAGENE-1 model using HuggingFace mirror or alternative methods.
"""

import os
import sys
from transformers import AutoModel, AutoTokenizer
import torch

def download_with_mirror():
    """Download using HuggingFace mirror."""
    
    print("=" * 80)
    print("Downloading METAGENE-1 via Mirror/Alternative Methods")
    print("=" * 80)
    
    # Set HuggingFace mirror (for regions with slow access)
    # Uncomment one of these mirror options:
    
    # Option 1: HF Mirror (China)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # Option 2: ModelScope (Alternative)
    # os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'
    
    print(f"\n✓ Using HF endpoint: {os.environ.get('HF_ENDPOINT', 'default')}")
    
    model_name = "metagene-ai/METAGENE-1"
    
    try:
        print("\n[1/2] Downloading tokenizer...")
        print("-" * 80)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,  # Resume if interrupted
            force_download=False   # Use cache if available
        )
        print(f"✓ Tokenizer downloaded: vocab size = {len(tokenizer)}")
        
        print("\n[2/2] Downloading model (16GB)...")
        print("-" * 80)
        print("This may take 30-60 minutes depending on your connection...")
        
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            resume_download=True,
            force_download=False,
            device_map="auto",
            low_cpu_mem_usage=True  # Reduce memory usage during load
        )
        print(f"✓ Model downloaded successfully!")
        
        # Quick test
        print("\n[3/3] Testing model...")
        test_seq = "ACGTACGTACGT"
        inputs = tokenizer(test_seq, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Model works! Output shape: {outputs.last_hidden_state.shape}")
        
        print("\n" + "=" * 80)
        print("✓ SUCCESS! Model ready for training.")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cache():
    """Check what's already in cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, "models--metagene-ai--METAGENE-1")
    
    print("\n" + "=" * 80)
    print("Checking HuggingFace Cache")
    print("=" * 80)
    
    if os.path.exists(model_cache):
        print(f"✓ Model cache found at: {model_cache}")
        
        # Check blobs
        blobs_dir = os.path.join(model_cache, "blobs")
        if os.path.exists(blobs_dir):
            blobs = os.listdir(blobs_dir)
            complete = [b for b in blobs if not b.endswith('.incomplete')]
            incomplete = [b for b in blobs if b.endswith('.incomplete')]
            
            print(f"\n  Complete files: {len(complete)}")
            print(f"  Incomplete files: {len(incomplete)}")
            
            if incomplete:
                print(f"\n  ⚠️ Found {len(incomplete)} incomplete downloads.")
                print(f"  These will be cleaned up and resumed.")
    else:
        print("  No cache found - will download from scratch.")
    
    print("=" * 80)

if __name__ == "__main__":
    check_cache()
    
    print("\n" + "=" * 80)
    print("Download Options:")
    print("=" * 80)
    print("1. Press ENTER to start download with mirror")
    print("2. Press Ctrl+C to cancel")
    print("=" * 80)
    
    try:
        input("\nPress ENTER to continue...")
    except KeyboardInterrupt:
        print("\n\nCanceled by user.")
        sys.exit(1)
    
    success = download_with_mirror()
    sys.exit(0 if success else 1)



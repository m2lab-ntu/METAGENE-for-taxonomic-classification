"""
Pre-download METAGENE-1 model and tokenizer from HuggingFace.
"""

import sys
from transformers import AutoModel, AutoTokenizer
import torch

def download_model():
    """Download METAGENE-1 model and tokenizer."""
    
    print("=" * 80)
    print("Downloading METAGENE-1 Model and Tokenizer")
    print("=" * 80)
    print("\nThis will download approximately 16GB of model files.")
    print("Please ensure you have sufficient disk space and stable internet connection.\n")
    
    model_name = "metagene-ai/METAGENE-1"
    
    try:
        # Download tokenizer (small, ~200MB)
        print("\n[1/2] Downloading tokenizer...")
        print("-" * 80)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer downloaded successfully")
        print(f"  - Vocab size: {len(tokenizer)}")
        
        # Download model (large, ~16GB)
        print("\n[2/2] Downloading model...")
        print("-" * 80)
        print("This may take a while (16GB download)...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        print(f"✓ Model downloaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Hidden size: {model.config.hidden_size}")
        
        # Test inference
        print("\n[3/3] Testing model inference...")
        print("-" * 80)
        test_sequence = "ACGTACGTACGTACGT"
        inputs = tokenizer(test_sequence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✓ Model inference successful")
        print(f"  - Input sequence: {test_sequence}")
        print(f"  - Output shape: {outputs.last_hidden_state.shape}")
        
        print("\n" + "=" * 80)
        print("✓ ALL DOWNLOADS COMPLETE!")
        print("=" * 80)
        print("\nModel is cached at: ~/.cache/huggingface/hub/")
        print("You can now run training with:")
        print("  python train.py --config configs/default.yaml ...")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)



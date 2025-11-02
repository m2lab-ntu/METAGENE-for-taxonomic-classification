"""
Verify that METAGENE-1 model is downloaded and working.
"""

import os
os.environ['HF_HOME'] = '/media/user/disk2/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/media/user/disk2/.cache/huggingface'

from transformers import AutoModel, AutoTokenizer
import torch

print("=" * 80)
print("Verifying METAGENE-1 Model")
print("=" * 80)

try:
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "metagene-ai/METAGENE-1",
        trust_remote_code=True
    )
    print(f"✓ Tokenizer OK: vocab_size={len(tokenizer)}")
    
    print("\n[2/3] Loading model...")
    model = AutoModel.from_pretrained(
        "metagene-ai/METAGENE-1",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu"  # Use CPU for verification
    )
    print(f"✓ Model OK: hidden_size={model.config.hidden_size}")
    
    print("\n[3/3] Testing inference...")
    test_seq = "ACGTACGTACGT"
    inputs = tokenizer(test_seq, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Inference OK: output shape = {outputs.last_hidden_state.shape}")
    
    print("\n" + "=" * 80)
    print("✅ METAGENE-1 MODEL IS READY!")
    print("=" * 80)
    print(f"\nModel location: {os.environ['HF_HOME']}")
    print("\nYou can now train classification models with:")
    print("  export HF_HOME=/media/user/disk2/.cache/huggingface")
    print("  python train.py --config configs/default.yaml ...")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()


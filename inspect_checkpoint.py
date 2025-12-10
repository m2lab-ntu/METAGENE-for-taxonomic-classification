import torch
import sys

def inspect_checkpoint(path):
    print(f"Inspecting {path}...")
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        val_acc = checkpoint.get('val_acc', 'Unknown')
        print(f"Validation Accuracy: {val_acc}")
        if isinstance(val_acc, float):
             print(f"Validation Accuracy (%): {val_acc * 100:.2f}%")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        print("Usage: python inspect_checkpoint.py <path_to_checkpoint>")

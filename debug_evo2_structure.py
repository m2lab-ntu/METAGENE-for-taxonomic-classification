
import sys
import os
import torch

# Add local Evo2 paths
sys.path.append('/home/user/Metagenomics/evo2')
sys.path.append('/home/user/Metagenomics/evo2/vortex')

from evo2.models import Evo2

def inspect_model():
    print("Loading Evo2 model...")
    # Load the model (using the same name as in config)
    model_wrapper = Evo2("evo2_1b_base")
    model = model_wrapper.model
    
    print("\nModel structure (top level):")
    for name, module in model.named_children():
        print(f"  {name}: {type(module)}")
        
    # Check 'blocks' specifically if it exists
    if hasattr(model, 'blocks'):
        print(f"\n'blocks' type: {type(model.blocks)}")
        print(f"Number of blocks: {len(model.blocks)}")
        
        # Check the first block's structure
        print("\nStructure of block 0:")
        for name, module in model.blocks[0].named_children():
            print(f"  {name}: {type(module)}")
            
    # Try to find the layer we want
    target_layer = "blocks.28.mlp.l3"
    print(f"\nTrying to access '{target_layer}'...")
    try:
        submodule = model.get_submodule(target_layer)
        print(f"Success! Found module: {submodule}")
    except Exception as e:
        print(f"Failed: {e}")
        
    # Check config
    if hasattr(model, 'config'):
        print(f"\nModel config type: {type(model.config)}")
        # Try to find hidden size in config
        for key in ['d_model', 'hidden_size', 'n_embd']:
            if hasattr(model.config, key):
                print(f"Config.{key}: {getattr(model.config, key)}")
            elif isinstance(model.config, dict) and key in model.config:
                print(f"Config['{key}']: {model.config[key]}")
                
    # List all named modules to find the right path
    print("\nListing first 50 named modules to find patterns:")
    for i, (name, _) in enumerate(model.named_modules()):
        if i > 50: break
        print(f"  {name}")

if __name__ == "__main__":
    inspect_model()

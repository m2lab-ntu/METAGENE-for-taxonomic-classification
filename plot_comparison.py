import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def plot_comparison(output_dirs, save_path):
    plt.figure(figsize=(12, 8))
    
    colors = {
        'METAGENE-1': 'green',
        'GENERanno (100e)': 'blue',
        'Evo2': 'red'
    }
    
    for label, dir_path in output_dirs.items():
        csv_path = os.path.join(dir_path, 'log_history.csv')
        if os.path.exists(csv_path):
            print(f"Loading {label} from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Plot line
            plt.plot(df['epoch'], df['val_acc'], label=label, color=colors.get(label, 'gray'), linewidth=2, alpha=0.8)
            
            # Plot points for every epoch
            plt.scatter(df['epoch'], df['val_acc'], color=colors.get(label, 'gray'), s=20, alpha=0.6)
            
            # Annotate final point
            final_epoch = df['epoch'].iloc[-1]
            final_acc = df['val_acc'].iloc[-1]
            plt.annotate(f"{final_acc*100:.1f}%", xy=(final_epoch, final_acc), xytext=(5, 5), textcoords='offset points')
            
    plt.title('Model Training Comparison on Zymo Dataset', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot to {save_path}")

# Defines experiments to plot
experiments = {
    'METAGENE-1': 'outputs/zymo_metagene1',
    'GENERanno (100e)': 'outputs/zymo_generanno_100e',
    'Evo2': 'outputs/zymo_evo2'
}

plot_comparison(experiments, 'outputs/model_comparison.png')

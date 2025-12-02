import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

def evaluate(predictions_path, mapping_path="zymo_mapping.tsv"):
    print(f"Evaluating {predictions_path} using mapping {mapping_path}...")
    df = pd.read_csv(predictions_path)
    
    # Load mapping
    mapping_df = pd.read_csv(mapping_path, sep='\t')
    # Create class_id to label_name mapping
    class_to_label = dict(zip(mapping_df['class_id'], mapping_df['label_name']))
    
    def get_true_label(seq_id):
        # Format: lbl|class|tax_id|genus|species_name/pair_end
        try:
            parts = seq_id.split('|')
            if len(parts) >= 2:
                class_id = int(parts[1])
                return class_to_label.get(class_id, "Unknown")
        except:
            pass
        return "Unknown"

    df['true_label'] = df['sequence_id'].apply(get_true_label)
    
    # Calculate metrics
    accuracy = accuracy_score(df['true_label'], df['predicted_class'])
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(df['true_label'], df['predicted_class'], zero_division=0))
    
    # Save comparison
    output_file = predictions_path.replace('.csv', '_evaluated.csv')
    df[['sequence_id', 'true_label', 'predicted_class', 'confidence']].to_csv(
        output_file, index=False
    )
    print(f"\nSaved evaluation details to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path", nargs='?', default="outputs/zymo_test_full/gpu_predictions.csv")
    parser.add_argument("--mapping", default="zymo_mapping.tsv")
    args = parser.parse_args()
    
    evaluate(args.predictions_path, args.mapping)

# %% Load required libraries
import joblib
import numpy as np
import json
import pandas as pd
import glob
import os
import datetime
import argparse

# %% Define function to get the latest file


def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getctime)


def get_unique_path(filepath):
    """Return a unique file path by appending a timestamp if needed."""
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load trained MLP model and make predictions on new embeddings.")
    parser.add_argument('--model_folder', type=str, default='models/', required=False,
                        help="Folder containing trained model and encoder (default: 'models/').")
    parser.add_argument('--embeddings_folder', type=str, default='data/new_data/', required=False,
                        help="Folder containing embeddings.npy and timestamps.json (default: 'data/new_data/').")
    parser.add_argument('--output_csv', type=str, default='results/predictions_with_timestamps_and_scores.csv', required=False,
                        help="Path to output CSV file (default: 'results/predictions_with_timestamps_and_scores.csv').")
    parser.add_argument('--embedding_min', type=str, default='models/embedding_min.npy', required=False,
                        help="Path to embedding_min.npy (default: 'models/embedding_min.npy').")
    parser.add_argument('--embedding_max', type=str, default='models/embedding_max.npy', required=False,
                        help="Path to embedding_max.npy (default: 'models/embedding_max.npy').")
    parser.add_argument('--threshold', type=float, default=0.5, required=False,
                        help="Threshold for multilabel prediction (default: 0.5).")
    args = parser.parse_args()

    # Find latest model and encoder
    model_path = get_latest_file(os.path.join(
        args.model_folder, 'mlp_model_multilabel*.pkl'))
    encoder_path = get_latest_file(os.path.join(
        args.model_folder, 'mlb_encoder*.pkl'))
    mlp = joblib.load(model_path)
    mlb = joblib.load(encoder_path)
    print(f"Loaded model: {model_path}")
    print(f"Loaded encoder: {encoder_path}")

    # Load new embeddings
    embeddings_path = os.path.join(args.embeddings_folder, 'embeddings.npy')
    new_embeddings = np.load(embeddings_path)

    # Load normalization min/max
    emb_min = np.load(args.embedding_min)
    emb_max = np.load(args.embedding_max)
    new_embeddings = (new_embeddings - emb_min) / (emb_max - emb_min)

    # Load timestamps
    timestamps_path = os.path.join(args.embeddings_folder, 'timestamps.json')
    with open(timestamps_path) as f:
        timestamps = json.load(f)

    # Predict probabilities for each class
    proba = mlp.predict_proba(new_embeddings)
    proba_matrix = np.column_stack([
        p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel() for p in proba
    ])

    # Ensure correct shape
    if proba_matrix.shape[1] != len(mlb.classes_):
        proba_matrix = proba_matrix.T
    print("proba_matrix shape:", proba_matrix.shape)

    # Use threshold to get predictions
    threshold = args.threshold
    predictions = (proba_matrix >= threshold).astype(int)
    print("predictions shape:", predictions.shape)

    decoded = mlb.inverse_transform(predictions)
    print("decoded length:", len(decoded))
    print("timestamps length:", len(timestamps))

    # Prepare results for CSV
    results = []
    class_names = mlb.classes_
    for i, (labels, ts, probs) in enumerate(zip(decoded, timestamps, proba_matrix)):
        row = {
            "sample_index": i,
            "audio_file": ts.get("file", ""),
            "start": ts.get("start", ""),
            "end": ts.get("end", ""),
            "predicted_species": ", ".join(labels) if labels else ""
        }
        for cname, score in zip(class_names, probs):
            row[f"score_{cname}"] = score
        results.append(row)

    results_df = pd.DataFrame(results)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_path = get_unique_path(args.output_csv)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

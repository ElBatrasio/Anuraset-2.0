# %% Load required libraries
import joblib
import numpy as np
import json
import pandas as pd
import glob
import os
import datetime

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


# %% Load the trained model and encoder
model_path = get_latest_file(
    r"models\mlp_model_multilabel*.pkl")
encoder_path = get_latest_file(
    r"models\mlb_encoder*.pkl")

mlp = joblib.load(model_path)
mlb = joblib.load(encoder_path)
print(f"Loaded model: {model_path}")
print(f"Loaded encoder: {encoder_path}")

# %% Load new embeddings
new_embeddings = np.load(
    r"C:\Users\gbida\Projects\anurabird\data\extra\embeddings\embeddings.npy")

emb_min = np.load('embedding_min.npy')
emb_max = np.load('embedding_max.npy')
new_embeddings = (new_embeddings - emb_min) / (emb_max - emb_min)

# %% Load timestamps
with open(r"C:\Users\gbida\Projects\anurabird\data\extra\embeddings\timestamps.json") as f:
    timestamps = json.load(f)

# %% Predict probabilities for each class
proba = mlp.predict_proba(new_embeddings)
proba_matrix = np.column_stack(
    [p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel() for p in proba]
)

# %% Ensure correct shape
if proba_matrix.shape[1] != len(mlb.classes_):
    proba_matrix = proba_matrix.T

# Should be (n_samples, n_classes)
print("proba_matrix shape:", proba_matrix.shape)

# %% Use threshold to get predictions
threshold = 0.5
predictions = (proba_matrix >= threshold).astype(int)
# Should be (n_samples, n_classes)
print("predictions shape:", predictions.shape)

decoded = mlb.inverse_transform(predictions)
print("decoded length:", len(decoded))
print("timestamps length:", len(timestamps))

# %% Prepare results for CSV
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

# Add this line to create the DataFrame
results_df = pd.DataFrame(results)

# %% Save to CSV
csv_path = r"C:\Users\gbida\Projects\anurabird\results\predictions_with_timestamps_and_scores.csv"
csv_path = get_unique_path(csv_path)
results_df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# %%

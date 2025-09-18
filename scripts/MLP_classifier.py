# %% Import Libraries
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import numpy as np
import json
from random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import os  # Import os to handle folder paths
import datetime
import argparse


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
        description="Train and evaluate MLP classifier for multilabel audio classification.")
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to the CSV file with audio file labels.')
    parser.add_argument('--timestamps_json', type=str,
                        required=True, help='Path to the timestamps JSON file.')
    parser.add_argument('--embeddings_npy', type=str,
                        required=True, help='Path to the embeddings NPY file.')
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Folder to save trained models and reports.')
    args = parser.parse_args()

    # Set working directory to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    os.chdir(project_dir)
    print(f"Working directory set to: {os.getcwd()}")

    # Load the CSV file with labels
    labels_df = pd.read_csv(args.labels_csv, delimiter=',')

    # Dynamically extract species columns (all columns except 'audio_file')
    species_columns = [col for col in labels_df.columns if col != 'audio_file']

    # Create a mapping of audio files to their corresponding species labels
    file_to_label = {
        row['audio_file']: [
            species for species in species_columns if row[species] == 1]
        for _, row in labels_df.iterrows()
    }

    # Load the JSON file
    with open(args.timestamps_json, 'r') as f:
        timestamps = json.load(f)

    # Extract labels for each audio file in the JSON file
    labels = []
    for entry in timestamps:
        audio_file = entry['file']
        label = file_to_label.get(audio_file, None)
        if label is not None:
            labels.append(label)

    # Convert labels to a binary matrix using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    # Load embeddings
    embeddings = np.load(args.embeddings_npy)

    # Normalize embeddings
    embeddings = (embeddings - np.min(embeddings)) / \
        (np.max(embeddings) - np.min(embeddings))

    # Shuffle data and labels
    embeddings, labels = shuffle(embeddings, labels, random_state=42)

    # Extract audio file names for each sample in the same order as embeddings/labels
    audio_files = [entry['file'] for entry in timestamps]

    # Get unique audio files and split them
    unique_files = sorted(set(audio_files))
    train_files, temp_files = train_test_split(
        unique_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=42)

    # Assign each sample to a split based on its audio file
    train_idx = [i for i, f in enumerate(audio_files) if f in train_files]
    val_idx = [i for i, f in enumerate(audio_files) if f in val_files]
    test_idx = [i for i, f in enumerate(audio_files) if f in test_files]

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_val, y_val = embeddings[val_idx], labels[val_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    # Check the shape of the data
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Build and Train the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                        learning_rate_init=0.0005, max_iter=500, random_state=42)

    # Train the model
    mlp.fit(X_train, y_train)

    # Specify the folder path where the best model will be saved
    model_save_folder = args.model_folder
    os.makedirs(model_save_folder, exist_ok=True)

    # Initialize variables to track the best model
    best_val_f1 = 0.0

    # Evaluate on validation set
    val_predictions = mlp.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions, average='micro')
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation F1 Score: {val_f1}")

    # Save the model if it has the best validation F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_path = os.path.join(
            model_save_folder, 'best_mlp_model_multilabel.pkl')
        joblib.dump(mlp, best_model_path)
        print(f"New best model saved to '{best_model_path}'")

    # Evaluate the Model on the test set
    test_predictions = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions, average='micro')
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test F1 Score: {test_f1}")

    # Generate a classification report for the validation set
    val_report = classification_report(
        y_val, val_predictions, target_names=mlb.classes_, output_dict=True)
    val_report_df = pd.DataFrame(val_report).transpose()
    val_report_path = get_unique_path(os.path.join(
        model_save_folder, 'validation_classification_report.csv'))
    val_report_df.to_csv(val_report_path, index=True)
    print(f"Validation classification report saved to '{val_report_path}'")

    # Generate a classification report for the test set
    test_report = classification_report(
        y_test, test_predictions, target_names=mlb.classes_, output_dict=True)
    test_report_df = pd.DataFrame(test_report).transpose()
    test_report_path = get_unique_path(os.path.join(
        model_save_folder, 'test_classification_report.csv'))
    test_report_df.to_csv(test_report_path, index=True)
    print(f"Test classification report saved to '{test_report_path}'")

    # Save the model and MultiLabelBinarizer to the same folder, avoiding overwrite
    model_path = os.path.join(model_save_folder, 'mlp_model_multilabel.pkl')
    encoder_path = os.path.join(model_save_folder, 'mlb_encoder.pkl')

    model_path = get_unique_path(model_path)
    encoder_path = get_unique_path(encoder_path)

    joblib.dump(mlp, model_path)
    joblib.dump(mlb, encoder_path)
    print(
        f"Model and encoder saved to '{model_save_folder}' as '{os.path.basename(model_path)}' and '{os.path.basename(encoder_path)}'")

    emb_min = np.min(embeddings)
    emb_max = np.max(embeddings)
    np.save(os.path.join(model_save_folder, 'embedding_min.npy'), emb_min)
    np.save(os.path.join(model_save_folder, 'embedding_max.npy'), emb_max)

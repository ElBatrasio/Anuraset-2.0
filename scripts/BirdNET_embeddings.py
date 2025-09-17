from birdnet_analyzer.analyze.utils import get_raw_audio_from_file
from birdnet_analyzer.audio import get_audio_file_length
import birdnet_analyzer.model as model
import birdnet_analyzer.config as cfg
import os
import sys
import numpy as np
import json


def extract_embeddings_for_keras(input_folder, output_folder, segment_duration=3, overlap=0.0):
    """
    Extracts embeddings from audio files (any length) and saves them in a format compatible with Keras.

    Args:
        input_folder (str): Path to the folder containing audio files.
        output_folder (str): Path to save the extracted embeddings.
        segment_duration (int): Duration (in seconds) of each segment.
        overlap (float): Overlap between segments (in seconds).
    """
    os.makedirs(output_folder, exist_ok=True)

    cfg.set_config({
        "SIG_LENGTH": segment_duration,
        "SIG_OVERLAP": overlap,
        "BATCH_SIZE": 1
    })

    audio_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    embeddings_list = []
    timestamps_list = []

    for audio_file in audio_files:
        try:
            file_length = int(get_audio_file_length(audio_file))
            step = segment_duration - overlap
            start = 0
            while start + segment_duration <= file_length:
                end = start + segment_duration
                audio_data = get_raw_audio_from_file(
                    audio_file, start, segment_duration)[0]
                data = np.array([audio_data], dtype="float32")
                embedding = model.embeddings(data)[0]
                embeddings_list.append(embedding.tolist())
                timestamps_list.append(
                    {"file": os.path.basename(
                        audio_file), "start": start, "end": end}
                )
                start += step

        except Exception as ex:
            print(f"Error processing {audio_file}: {ex}")
            continue

    np.save(os.path.join(output_folder, "embeddings.npy"),
            np.array(embeddings_list))
    with open(os.path.join(output_folder, "timestamps.json"), "w") as f:
        json.dump(timestamps_list, f, indent=4)

    print(f"Embeddings and timestamps saved to {output_folder}")


# Example usage
if __name__ == "__main__":
    input_folder = r"C:\Users\gbida\Projects\anuraset\datasets\anuraset\audio"
    output_folder = r"C:\Users\gbida\Projects\anurabird\data_test\embeddings"
    segment_duration = 3  # seconds
    overlap = 1         # seconds, for 50% overlap

    extract_embeddings_for_keras(
        input_folder, output_folder, segment_duration, overlap)

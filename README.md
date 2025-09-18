# User-friendly workflow for automated multi-label classification of Neotropical anuran calls using BirdNET embeddings and MLP models: 

This repository hosts a comprehensive and user-friendly workflow for automated monitoring of anuran species in complex tropical soundscapes using state-of-the-art machine learning techniques. Developed as part of a broader conservation initiative, this project targets the pressing challenge of global amphibian declines, particularly in biodiverse and acoustically complex Neotropical environments.
The workflow leverages the publicly available pre-trained convolutional neural network (CNN) of BirdNET  as embedding extractor for audio recordings. These embeddings are then used to train a multilayer perceptron (MLP) classifier, configured specifically for multi-label classification to handle overlapping calls from multiple species within the same audio segment. The approach is designed to be accessible to conservation practitioners and researchers with minimal machine learning expertise by utilizing widely adopted Python libraries such as scikit-learn and the BirdNET Analyzer.
Key contributions include:
•	Implementation of an automated pipeline for extracting high-dimensional embeddings using [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer.git) from audio data segmented using sliding windows, preserving fine temporal resolution.
•	Integration of precise and robust multi-label classification models trained on the expert-annotated [AnuraSet](https://github.com/soundclim/anuraset/) dataset , a benchmark collection of over 1600 annotated recordings covering 42 Neotropical anuran species across diverse Brazilian biomes.
•	Calibration of model prediction scores into well-interpretable probabilities using logistic modeling, enabling users to set confidence thresholds balancing prediction certainty and recall.

## Installation instruction and reproduction of results

1. Install [Conda](https://docs.conda.io/en/latest/)
2. Clone this repository
```bash
git clone https://github.com/ElBatrasio/Frog-party_detector.git
```
3. Create an environment and install requirements
```bash
cd Frog-party_detector
conda create -n Frog-party_detector python=3.11 -y
conda activate Frog-party_detector
pip install -r requirements.txt
```
4. Extract embeddings
Extract embeddings using the BirdNET's V2.4 model. You need to specify the path to your audio files, the output path for the embeddings and the timestamp and the window duration (and overlap if needed)
 
```bash
python scripts/BirdNET_embeddings.py --input_folder "path/to/input" --output_folder "path/to/output" --segment_duration 3 --overlap 1 
```
* IMPORTANT
The names of the folders must NOT contain spaces between words, use a low bar "_" instead.

5. Train the model
Train the MLP using embeddings as input. You have to specify both the path to the embeddings.

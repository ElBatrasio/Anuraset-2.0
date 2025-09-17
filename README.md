# User-friendly workflow for automated multi-label classification of Neotropical anuran calls using BirdNET embeddings and MLP models: 

This repository hosts a comprehensive and user-friendly workflow for automated monitoring of anuran species in complex tropical soundscapes using state-of-the-art machine learning techniques. Developed as part of a broader conservation initiative, this project targets the pressing challenge of global amphibian declines, particularly in biodiverse and acoustically complex Neotropical environments.
The workflow leverages publicly available pre-trained convolutional neural networks (CNNs) — notably BirdNET — as embedding extractors for audio recordings. These embeddings are then used to train a multilayer perceptron (MLP) classifier, configured specifically for multi-label classification to handle overlapping calls from multiple species within the same audio segment. The approach is designed to be accessible to conservation practitioners and researchers with minimal machine learning expertise by utilizing widely adopted Python libraries such as scikit-learn and the BirdNET Analyzer.
Key contributions include:
•	Implementation of an automated pipeline for extracting high-dimensional embeddings from audio data segmented using sliding windows, preserving fine temporal resolution.
•	Integration of precise and robust multi-label classification models trained on the expert-annotated [AnuraSet](https://github.com/soundclim/anuraset/) dataset , a benchmark collection of over 1600 annotated recordings covering 42 Neotropical anuran species across diverse Brazilian biomes.
•	Calibration of model prediction scores into well-interpretable probabilities using logistic modeling, enabling users to set confidence thresholds balancing prediction certainty and recall.

## Installation instruction and reproduction of results

1. Install [Conda](https://docs.conda.io/en/latest/)
2. Clone this repository
```bash
git clone https://github.com/ElBatrasio/Anuraset-2.0.git
```
3.Create an environment and install requirements
```bash
cd Anuraset-2.0
conda create -n Anuraset-2.0 python=3.11 -y
conda activate Anuraset-2.0
  

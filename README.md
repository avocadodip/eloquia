# Eloquia Project Repository

## Overview
This repository contains the machine learning model and supporting scripts for the Eloqia venture, designed to improve speech fluency through AI-powered analysis. Eloquia leverages audio data to detect and analyze speech patterns, aiding in the development of personalized speech therapy tools.

## Contents
- `model_utils.py`: Utility functions for model operations.
- `data_utils.py`: Functions for handling and preparing data.
- `train.py`: Entry point for training the speech analysis model.
- `inference.py`: Script to perform inference using the trained model.
- `long_audio_inference.py`: Extends inference capabilities to handle long-duration audio files.

### Data
- `data/`: Folder containing necessary datasets and label files for model training and testing.
- `mel_filters.npz`: Mel filter bank for feature extraction from audio files.

### Model
- `model.safetensors`: Pre-trained machine learning model ready for deployment and further training.

### Additional Files
- `long_test.mp3`: Test audio file to demonstrate the inference process.
- `.gitignore`: File specifying untracked files like logs and local configuration.

## Setup
Ensure Python 3.8+ is installed. Install required dependencies via:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, run:
```bash
python train.py
```

For running inference:
```bash
python inference.py [AUDIO_FILE]
```

To process long audio files:
```bash
python long_audio_inference.py [AUDIO_FILE]
```

## Setup
For further inquiries, contact akondep1@jh.edu
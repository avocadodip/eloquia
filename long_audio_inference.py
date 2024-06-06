import torch
import torchaudio
import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from pathlib import Path
import sys

# Set up device for model processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and feature extractor
def load_model(model_path, feature_extractor_path):
    model = WhisperForAudioClassification.from_pretrained(model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)
    return model, feature_extractor

# Utility function to process and load audio
def load_audio(file_path, sample_rate=16000):
    waveform, _ = torchaudio.load(file_path)
    resampled_waveform = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)(waveform)
    return resampled_waveform.squeeze(0).numpy()

# Process the audio file into segments and predict
def process_audio(audio_path, model, feature_extractor, segment_length=3):
    audio = load_audio(audio_path)
    sample_rate = feature_extractor.sampling_rate
    segment_length_samples = segment_length * sample_rate
    
    # Split audio into chunks
    segments = [audio[i:i + segment_length_samples] for i in range(0, len(audio), segment_length_samples) if len(audio[i:i + segment_length_samples]) == segment_length_samples]
    
    results = []
    for segment in segments:
        inputs = feature_extractor(segment, return_tensors="pt", sampling_rate=sample_rate, padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        results.append(predicted_ids.item())

    # Aggregate results
    counts = pd.Series(results).value_counts().to_dict()
    return counts

def main(audio_file_path, model_path, feature_extractor_path):
    model, feature_extractor = load_model(model_path, feature_extractor_path)
    classification_results = process_audio(audio_file_path, model, feature_extractor)
    print("Classification counts for each label:", classification_results)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script_name.py <audio_file_path> <model_path> <feature_extractor_path>")
        sys.exit(1)
    audio_file_path = sys.argv[1]
    model_path = sys.argv[2]
    feature_extractor_path = sys.argv[3]
    main(audio_file_path, model_path, feature_extractor_path)

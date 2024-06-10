import torch
from transformers import WhisperForAudioClassification, AutoFeatureExtractor
import numpy as np
import os
from safetensors.torch import load_model

# Load the model and feature extractor
token = os.getenv('HF_TOKEN')
model_id = "openai/whisper-tiny"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
model = WhisperForAudioClassification.from_pretrained(model_id, token=token, num_labels=7, use_safetensors=True)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model = load_model(model, "results/checkpoint-515/model.safetensors")

model.eval()

class AudioUtil:
    # Assuming you have defined this class elsewhere in your script with necessary methods like load_audio, pad_or_trim, etc.
    pass

def predict(audio_file):
    audio = AudioUtil.load_audio(audio_file)
    audio = AudioUtil.pad_or_trim(audio)

    # Convert audio to log-Mel spectrogram
    mel = AudioUtil.log_mel_spectrogram(audio).to(device)

    # Prepare input for the model
    with torch.no_grad():
        inputs = feature_extractor(mel, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits

    # Optionally, apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    return predicted_class

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_audio_file>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    prediction = predict(audio_file_path)
    print("Predicted class:", prediction)

if __name__ == "__main__":
    main()

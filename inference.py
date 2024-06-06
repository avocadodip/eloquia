import torch
from transformers import WhisperForAudioClassification, AutoFeatureExtractor
import numpy as np

# Load the model and feature extractor
model_path = "./whisper_fine_tuned"
feature_extractor_path = "./whisper_feature_extractor"
model = WhisperForAudioClassification.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
        print("Usage: python script_name.py <path_to_audio_file>")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    prediction = predict(audio_file_path)
    print("Predicted class:", prediction)

if __name__ == "__main__":
    main()

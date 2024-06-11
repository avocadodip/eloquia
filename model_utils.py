from transformers import AutoFeatureExtractor, WhisperForAudioClassification
import os

def get_model(device):
    token = os.getenv('HF_TOKEN')
    model_id = "openai/whisper-medium"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
    model = WhisperForAudioClassification.from_pretrained(model_id, token=token, num_labels=7)

    feature_extractor
    model.to(device)

    return model, feature_extractor
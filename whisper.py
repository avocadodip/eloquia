import transformers

from transformers import AutoFeatureExtractor, WhisperForAudioClassification

model_id = "openai/whisper-tiny"
token = "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"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
model = WhisperForAudioClassification.from_pretrained(model_id, token=token)
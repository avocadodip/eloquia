import pandas as pd
import torchaudio
from torchaudio.transforms import Resample
import os

class DataPreprocessor:
    def __init__(self, annotations_file, audio_dir, target_sample_rate):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate

    def process_audio_file(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform

    def create_processed_dataset(self):
        processed_data = []
        for _, row in self.annotations.iterrows():
            file_path = os.path.join(self.audio_dir, row['audio_filename'])
            label = row['label']
            waveform = self.process_audio_file(file_path)
            processed_data.append((waveform, label))
        return processed_data

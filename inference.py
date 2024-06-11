import torch
from transformers import WhisperForAudioClassification, AutoFeatureExtractor
import numpy as np
import os
from safetensors.torch import load_model

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from subprocess import CalledProcessError, run

import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union


# Load the model and feature extractor
token = os.getenv('HF_TOKEN')
model_id = "openai/whisper-tiny"

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=token)
model = WhisperForAudioClassification.from_pretrained(model_id, token=token, num_labels=7, use_safetensors=True)


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

file_path = "results/checkpoint-515/model.safetensors"
# with open(file_path, "rb") as f:
#     data = f.read()
load_model(model, file_path)

model.eval()

def exact_div(x, y):
    assert x % y == 0
    return x // y

class AudioUtil():
  # hard-coded audio hyperparameters
  SAMPLE_RATE = 16000
  N_FFT = 400
  HOP_LENGTH = 160
  CHUNK_LENGTH = 30
  N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
  N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

  N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
  FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
  TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


  def load_audio(file: str, sr: int = SAMPLE_RATE):
      """
      Open an audio file and read as mono waveform, resampling as necessary

      Parameters
      ----------
      file: str
          The audio file to open

      sr: int
          The sample rate to resample the audio if necessary

      Returns
      -------
      A NumPy array containing the audio waveform, in float32 dtype.
      """

      # This launches a subprocess to decode audio while down-mixing
      # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
      # fmt: off
      cmd = [
          "ffmpeg",
          "-nostdin",
          "-threads", "0",
          "-i", file,
          "-f", "s16le",
          "-ac", "1",
          "-acodec", "pcm_s16le",
          "-ar", str(sr),
          "-"
      ]
      # fmt: on
      try:
          out = run(cmd, capture_output=True, check=True).stdout
      except CalledProcessError as e:
          raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

      return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


  def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
      """
      Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
      """
      if torch.is_tensor(array):
          if array.shape[axis] > length:
              array = array.index_select(
                  dim=axis, index=torch.arange(length, device=array.device)
              )

          if array.shape[axis] < length:
              pad_widths = [(0, 0)] * array.ndim
              pad_widths[axis] = (0, length - array.shape[axis])
              array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
      else:
          if array.shape[axis] > length:
              array = array.take(indices=range(length), axis=axis)

          if array.shape[axis] < length:
              pad_widths = [(0, 0)] * array.ndim
              pad_widths[axis] = (0, length - array.shape[axis])
              array = np.pad(array, pad_widths)

      return array
  
  @lru_cache(maxsize=None)
  def mel_filters(device, n_mels: int) -> torch.Tensor:
        """
        load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
        Allows decoupling librosa dependency; saved using:

            np.savez_compressed(
                "mel_filters.npz",
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
            )
        """
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

        with np.load("data/mel_filters.npz", allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

  def log_mel_spectrogram(
        audio: Union[str, np.ndarray, torch.Tensor],
        n_mels: int = 80,
        padding: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                audio = AudioUtil.load_audio(audio)
            audio = torch.from_numpy(audio)

        if device is not None:
            audio = audio.to(device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(AudioUtil.N_FFT).to(audio.device)
        stft = torch.stft(audio, AudioUtil.N_FFT, AudioUtil.HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = AudioUtil.mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

def predict(audio_file):
    audio = AudioUtil.load_audio(audio_file)
    audio = AudioUtil.pad_or_trim(audio)

    # Convert audio to log-Mel spectrogram
    mel = AudioUtil.log_mel_spectrogram(audio).to(device)
    mel = mel.to(device)

    inputs = {"input_features": mel.unsqueeze(0)}  # Add batch dimension

    # Prepare input for the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

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

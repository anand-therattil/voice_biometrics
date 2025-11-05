import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


import numpy as np
import pandas as pd 
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

class FeatureExtractor:
    def __init__(self, 
                 feature_param: dict = {"sample_rate": 16000}
                 ):
        self.feature_param = feature_param 
        self.target_sr = 16000
        
    def extract_mfcc(self, audio_path, sample_rate=16000, n_mfcc=13, nfft=400, hop_length=160, n_mels=23):
        audio ,sr = self.load_audio(audio_path, sample_rate=sample_rate)
        # Define transform
        mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": nfft,
                        "n_mels": n_mels,
                        "hop_length": hop_length,
                        "mel_scale": "htk",
                    },
        )

        # Perform transform 
        mfcc = mfcc_transform(audio)
        return mfcc

    def extract_mel_spectrogram(self, audio_path, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        audio, sr = self.load_audio(audio_path, sample_rate=sample_rate)
        # Define transform
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Perform transform
        mel_spec = mel_spectrogram(audio)
        return mel_spec
    

    def extrat_spectogram(self, audio_path, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        audio, sr = self.load_audio(audio_path, sample_rate=sample_rate)
        # Define transform
        spectrogram = T.Spectrogram(n_fft=512)

        # Perform transform
        spec = spectrogram(audio)
        return spec

    def load_audio(self, audio_path, sample_rate=16000):
        audio, sr = torchaudio.load(audio_path)
        if sr != self.target_sr:
            audio = self.resample_audio(audio, sr, target_sr=sample_rate)
            sr = sample_rate

        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
        return audio, sr
    
    def resample_audio(self, audio, orig_sr, target_sr=16000):
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resampler(audio)
    
    

feature = FeatureExtractor()
data = pd.read_csv(os.path.join(PROJECT_PATH, "data.csv"),sep=',')

audio_file = data.iloc[0]['file_path']
print(audio_file)

feature_mfcc = feature.extract_mfcc(audio_file)
print("MFCC shape:", feature_mfcc.shape)

feature_mel = feature.extract_mel_spectrogram(audio_file)
print("Mel Spectrogram shape:", feature_mel.shape)

feature_spec = feature.extrat_spectogram(audio_file)
print("Spectrogram shape:", feature_spec.shape)
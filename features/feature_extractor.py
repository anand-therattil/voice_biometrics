import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


import numpy as np
import pandas as pd 
import sys, os
import matplotlib.pyplot as plt

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
    
    def save_feature_as_image(self, feature_tensor, output_path, title="", cmap='viridis', apply_log=True):
        """
        Save a feature tensor as an image.
        
        Args:
            feature_tensor: torch.Tensor of shape (1, freq_bins, time_frames)
            output_path: path to save the image
            title: title for the plot
            cmap: colormap to use
            apply_log: whether to apply log scaling (recommended for spectrograms)
        """
        # Remove batch dimension and convert to numpy
        feature_np = feature_tensor.squeeze(0).numpy()
        
        # Apply log scaling if requested (common for audio features)
        if apply_log:
            feature_np = np.log(feature_np + 1e-9)  # Add small epsilon to avoid log(0)
        
        # Create figure
        plt.figure(figsize=(10, 4))
        plt.imshow(feature_np, aspect='auto', origin='lower', cmap=cmap)
        plt.colorbar(format='%+2.0f dB' if apply_log else None)
        plt.title(title)
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {title} to {output_path}")

    def save_feature_as_array(self, feature_tensor, output_path):
        """
        Save feature tensor as a numpy array (.npy file) for later use.
        
        Args:
            feature_tensor: torch.Tensor to save
            output_path: path to save the .npy file
        """
        feature_np = feature_tensor.numpy()
        np.save(output_path, feature_np)
        print(f"Saved feature array to {output_path}")
    

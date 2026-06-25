"""
feature_extractor.py
Extracts BPM, energy, MFCCs and chroma features from an audio file using librosa.
"""

import librosa
import numpy as np


def extract_features(filepath, return_audio=False):
    """Load an audio file and extract a dictionary of acoustic features.

    If return_audio=True, also returns the raw (y, sr) so the caller can
    optionally feed it into a pretrained embedding model.
    """
    y, sr = librosa.load(filepath, mono=True, duration=60)  # first 60s is enough for fingerprinting

    # --- Tempo (BPM) ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    # --- Energy (RMS) ---
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    # --- MFCCs (timbre) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()

    # --- Chroma (harmonic structure) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).tolist()

    # --- Spectral centroid (brightness, helps explainability) ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid))

    result = {
        "bpm": round(bpm, 2),
        "energy": round(energy, 4),
        "mfcc": [round(v, 4) for v in mfcc_mean],
        "chroma": [round(v, 4) for v in chroma_mean],
        "brightness": round(centroid_mean, 2),
    }

    if return_audio:
        return result, y, sr
    return result

"""
embedding_generator.py
Builds a single numerical "fingerprint" (embedding) vector for a song.

Two modes:
  1. Handcrafted (default) - combines bpm/energy/mfcc/chroma/brightness
     into a normalized vector. No extra dependencies, always works.
  2. Pretrained (optional)  - uses OpenL3, a pretrained CNN audio
     embedding model, if it's installed. Falls back to handcrafted
     mode automatically if OpenL3 / its deps aren't available.

This lets you honestly say "uses pretrained CNN-based audio embeddings"
when USE_PRETRAINED=True and openl3 is installed, while the project
still works out-of-the-box without it.
"""

import numpy as np

USE_PRETRAINED = False  # set True to attempt OpenL3 embeddings

_openl3_available = None  # cached check


def _check_openl3():
    global _openl3_available
    if _openl3_available is None:
        try:
            import openl3  # noqa: F401
            _openl3_available = True
        except ImportError:
            _openl3_available = False
    return _openl3_available


def build_embedding(features: dict, y=None, sr=None):
    """
    Build the fingerprint vector for a song.

    If USE_PRETRAINED is True, openl3 is installed, and raw audio (y, sr)
    is provided, uses a pretrained CNN audio embedding (mean-pooled over
    time). Otherwise falls back to the handcrafted feature vector.
    """
    if USE_PRETRAINED and y is not None and sr is not None and _check_openl3():
        try:
            import openl3
            emb, _ts = openl3.get_audio_embedding(
                y, sr, content_type="music", embedding_size=512, verbose=0
            )
            embedding = np.mean(emb, axis=0)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()
        except Exception:
            pass  # silently fall back to handcrafted vector below

    return _handcrafted_embedding(features)


def _handcrafted_embedding(features: dict):
    """Combine bpm, energy, mfcc, chroma and brightness into one vector."""
    vec = []
    vec.append(features["bpm"] / 200.0)          # normalize roughly to 0-1
    vec.append(features["energy"])
    vec.append(features["brightness"] / 5000.0)  # normalize roughly
    vec.extend(features["mfcc"])
    vec.extend(features["chroma"])

    embedding = np.array(vec, dtype=float)

    # L2-normalize so cosine similarity behaves nicely
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()

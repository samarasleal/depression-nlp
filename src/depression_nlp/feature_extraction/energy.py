"""RMS energy feature extraction."""

import numpy as np


def extract_energy(audio_chunks: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate one RMS energy value for each audio chunk."""
    return np.sqrt(np.mean(audio_chunks**2, axis=1) + eps)

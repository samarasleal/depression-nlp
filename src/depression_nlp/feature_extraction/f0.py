"""Fundamental-frequency feature extraction."""

import librosa
import numpy as np


def extract_f0_segments(
    audio_chunks: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> list[np.ndarray]:
    """Extract frame-level F0 independently from each speaker chunk."""
    segments = []
    for chunk in audio_chunks:
        f0 = librosa.yin(
            y=chunk,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        segments.append(np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0))
    return segments


def summarize_f0(f0_values: np.ndarray) -> tuple[float, float]:
    """Return mean and standard deviation for one F0 segment."""
    return float(np.mean(f0_values)), float(np.std(f0_values))

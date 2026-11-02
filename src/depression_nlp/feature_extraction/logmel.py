"""Three-dimensional LogMel feature extraction for the future CNN."""

import librosa
import numpy as np


def extract_logmel_3d(
    audio_chunks: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    max_time_frames: int,
) -> np.ndarray:
    """Return LogMel with shape [segments, time, mel_bins]."""
    segments = []
    for chunk in audio_chunks:
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
        )
        logmel = librosa.power_to_db(mel, ref=max(float(np.max(mel)), 1e-10)).T
        logmel = logmel[:max_time_frames]
        if len(logmel) < max_time_frames:
            logmel = np.pad(logmel, ((0, max_time_frames - len(logmel)), (0, 0)))
        segments.append(np.nan_to_num(logmel, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(segments).astype(np.float32)

# Audio loading and preprocessing

from datetime import time
import librosa
import numpy as np
from scipy.signal import lfilter

def pre_emphasis(y: np.ndarray, coeff: float) -> np.ndarray:
    return lfilter([1.0, -coeff], [1.0], y)

def rms_normalize(y: np.ndarray, target_rms: float, eps: float = 1e-8) -> np.ndarray:
    # Normalize an audio signal to a RMS level
    rms = np.sqrt(np.mean(y**2) + eps)
    return y if rms < eps else np.clip(y * (target_rms / rms), -1.0, 1.0)

def calculate_frame_length(sr: int, frame_length_ms: float) -> int:
    # Convert a frame duration in milliseconds to samples
    return int(round(sr * frame_length_ms / 1000.0))

def calculate_hop_length(frame_length: int, frame_overlap: float) -> int:
    return max(1, int(round(frame_length * (1.0 - frame_overlap))))

def load_and_preprocess_audio(
    audio_path: str,
    sample_rate: int,
    pre_emphasis_coeff: float,
    target_rms: float,
) -> tuple[np.ndarray, int]:
    # Load mono audio at a fixed sample rate and preserve the notebook preprocessing
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return rms_normalize(pre_emphasis(y, pre_emphasis_coeff), target_rms), sr

def get_seconds(time_value: str | time) -> float:
    # Convert MM:SS, HH:MM:SS, or datetime.time to seconds
    if isinstance(time_value, time):
        return time_value.hour * 3600 + time_value.minute * 60 + time_value.second
    parts = [float(part) for part in str(time_value).strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported time value: {time_value}")

def extract_speaker_audio(y: np.ndarray, sr: int, annotation, speaker: str) -> np.ndarray:
    # Concatenate all exclusive speech turns assigned to one speaker
    segments = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        if label != speaker:
            continue
        start, end = max(0, int(turn.start * sr)), min(len(y), int(turn.end * sr))
        if end > start:
            segments.append(y[start:end])
    if not segments:
        raise ValueError(f"No audio was assigned to speaker {speaker}.")
    return np.concatenate(segments)


def split_audio_chunks(y: np.ndarray, sr: int, duration_seconds: float) -> np.ndarray:
    # Split audio into fixed-size chunks and zero-pad the final chunk
    chunk_length = int(round(duration_seconds * sr))
    if chunk_length <= 0:
        raise ValueError("Chunk duration must be greater than zero.")
    chunks = []
    for start in range(0, len(y), chunk_length):
        chunk = y[start : start + chunk_length]
        chunks.append(np.pad(chunk, (0, chunk_length - len(chunk))))
    if not chunks:
        raise ValueError("Cannot create chunks from empty audio.")
    return np.stack(chunks)

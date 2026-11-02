# Speaker diarization and patient identification 

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from pyannote.audio import Pipeline

@dataclass
class DiarizationResult:
    # Store regular annotations and aligned speaker embeddings
    annotation: object
    exclusive_annotation: object
    labels: list[str]
    embeddings: np.ndarray

def create_diarization_pipeline(model: str, token: str, device: str) -> Pipeline:
    """Load the local pyannote pipeline and move it to the selected device."""
    if not token:
        raise ValueError("HF_TOKEN is required to download the gated pyannote model.")
    pipeline = Pipeline.from_pretrained(model, token=token)
    if pipeline is None:
        raise RuntimeError(f"Could not load diarization model: {model}")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    pipeline.to(torch.device("cpu" if selected_device == "auto" else selected_device))
    return pipeline


def diarize_audio(pipeline: Pipeline, audio_path: str | Path, num_speakers: int) -> DiarizationResult:
    output = pipeline(str(audio_path), num_speakers=num_speakers)
    labels = list(output.speaker_diarization.labels())
    embeddings = np.asarray(output.speaker_embeddings, dtype=np.float32)
    if len(labels) != len(embeddings):
        raise ValueError("Speaker labels and embeddings are not aligned.")
    return DiarizationResult(
        annotation=output.speaker_diarization,
        exclusive_annotation=output.exclusive_speaker_diarization,
        labels=labels,
        embeddings=embeddings)


def identify_reference_speaker(annotation, start_seconds: float, end_seconds: float) -> str:
    """Identify the speaker with the greatest overlap with the annotated patient interval."""
    overlap_by_speaker: dict[str, float] = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        overlap = max(0.0, min(turn.end, end_seconds) - max(turn.start, start_seconds))
        overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap
    if not overlap_by_speaker or max(overlap_by_speaker.values()) == 0:
        raise ValueError("The patient reference interval does not overlap diarized speech.")
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def get_speaker_embedding(result: DiarizationResult, speaker: str) -> np.ndarray:
    # Return the embedding aligned with a diarized speaker label
    if speaker not in result.labels:
        raise ValueError(f"Speaker label not found: {speaker}")
    return result.embeddings[result.labels.index(speaker)]


def identify_patient_speaker(result: DiarizationResult, reference_embedding: np.ndarray) -> str:
    # Match session speakers to the patient's real reference using cosine similarity
    reference = np.asarray(reference_embedding, dtype=np.float32)
    reference /= max(np.linalg.norm(reference), 1e-12)
    embeddings = result.embeddings / np.maximum(
        np.linalg.norm(result.embeddings, axis=1, keepdims=True), 1e-12
    )
    return result.labels[int(np.argmax(embeddings @ reference))]


def assign_speaker_roles(result: DiarizationResult, patient_speaker: str) -> dict[str, str]:
    other_speakers = [speaker for speaker in result.labels if speaker != patient_speaker]
    if len(other_speakers) != 1:
        raise ValueError(f"Expected one therapist speaker, found {len(other_speakers)}.")
    return {"Patient": patient_speaker, "Therapist": other_speakers[0]}

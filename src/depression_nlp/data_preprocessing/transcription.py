# Transcribe therapist speech intervals with faster-whisper
import json
from pathlib import Path
import librosa
from faster_whisper import WhisperModel
from collections import defaultdict

def create_transcription_model(config: dict) -> WhisperModel:
    return WhisperModel(config["model"], device=config["device"], compute_type=config["compute_type"],)

def transcribe_therapist(
    model: WhisperModel,
    audio_path: str | Path,
    annotation,
    therapist_speaker: str,
    therapist_id: str,
    patient_id: str,
    session_number: int,
    output_root: str | Path,
    config: dict,
    segmentation_config: dict,
) -> str:
    """Transcribe therapist turns and preserve absolute session timestamps."""
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    transcript_segments = []
    for turn_id, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        if speaker != therapist_speaker or turn.duration < config["min_turn_seconds"]:
            continue
        start_sample, end_sample = int(turn.start * sr), int(turn.end * sr)
        audio_segment = y[start_sample:end_sample]
        segments, _ = model.transcribe(audio_segment, language=config["language"], task="transcribe", beam_size=config["beam_size"], condition_on_previous_text=False, vad_filter=False)
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_segments.append({
                    "segment_id": len(transcript_segments),
                    "turn_id": turn_id,
                    "start": round(turn.start + segment.start, 3),
                    "end": round(turn.start + segment.end, 3),
                    "text": text,
                })
    semantic_utterances = create_semantic_utterances(transcript_segments, segmentation_config["min_words"])
    payload = {
        "therapist_id": str(therapist_id),
        "patient_id": str(patient_id),
        "session": int(session_number),
        "audio_file": Path(audio_path).name,
        "language": config["language"],
        "whisper_model": config["model"],
        "utterances": semantic_utterances
    }
    output_path = (
        Path(output_root)
        / str(therapist_id)
        / f"{patient_id}_session_{session_number}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path.resolve())

def create_semantic_utterances(segments: list[dict], min_words: int = 3) -> list[dict]:
    """Merge short Whisper fragments into cleaner utterances."""
    utterances, pending = [], None
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        if pending and segment["turn_id"] == pending["turn_id"] and len(pending["text"].split()) < min_words:
            pending["text"] = f'{pending["text"]} {text}'.strip()
            pending["end"] = segment["end"]
        else:
            if pending and len(pending["text"].split()) >= min_words:
                pending["utterance_id"] = len(utterances)
                utterances.append(pending)
            pending = {
                "turn_id": segment["turn_id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
            }
    if pending and len(pending["text"].split()) >= min_words:
        pending["utterance_id"] = len(utterances)
        utterances.append(pending)
    return utterances
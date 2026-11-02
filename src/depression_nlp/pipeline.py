# Pyannote diarization followed by F0, energy, and LogMel extraction.

# Activate .venv
""" cd /workspace/app/depression-nlp
source .venv/bin/activate """

import argparse
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from depression_nlp.config import load_config
from depression_nlp.data_preprocessing.audio import (
    calculate_frame_length,
    calculate_hop_length,
    extract_speaker_audio,
    get_seconds,
    load_and_preprocess_audio,
    split_audio_chunks,
)
from depression_nlp.data_preprocessing.diarization import (
    assign_speaker_roles,
    create_diarization_pipeline,
    diarize_audio,
    get_speaker_embedding,
    identify_patient_speaker,
    identify_reference_speaker,
)
from depression_nlp.data_preprocessing.sessions import (
    find_patient_reference_audio,
    get_patient_row,
    is_session_processed,
    iter_patient_sessions,
    load_sessions,
    mark_session_processed,
)
from depression_nlp.data_preprocessing.transcription import (
    create_transcription_model,
    transcribe_therapist,
)

from depression_nlp.feature_extraction.energy import extract_energy
from depression_nlp.feature_extraction.f0 import extract_f0_segments, summarize_f0
from depression_nlp.feature_extraction.logmel import extract_logmel_3d
from depression_nlp.storage import save_array, save_dataframe

def create_patient_reference(pipeline, df, line_index: int, patient_folder: Path, config: dict):
    """Create a patient embedding from the real annotated reference interval."""
    audio_path = find_patient_reference_audio(df, line_index, patient_folder)
    result = diarize_audio(pipeline, audio_path, config["diarization"]["num_speakers"])
    start = get_seconds(df.at[line_index, "Patient_Speech_Interval_From"])
    end = get_seconds(df.at[line_index, "Patient_Speech_Interval_To"])
    speaker = identify_reference_speaker(result.annotation, start, end)
    return get_speaker_embedding(result, speaker)


def extract_role_features(role_audio, sr: int, role: str, patient_id: str, session_number: int, feature_root: Path, config: dict) -> dict[str, object]:
    """Extract enabled acoustic features from one speaker waveform."""
    audio_cfg, feature_cfg = config["audio"], config["features"]
    chunks = split_audio_chunks(role_audio, sr, audio_cfg["segment_duration_seconds"])
    chunks = chunks[: feature_cfg["max_segments"]]
    frame_length = calculate_frame_length(sr, audio_cfg["frame_length_ms"])
    hop_length = calculate_hop_length(frame_length, audio_cfg["frame_overlap"])
    values: dict[str, object] = {}
    if feature_cfg["f0"]["enabled"]:
        f0_segments = extract_f0_segments(
            chunks,
            sr,
            frame_length,
            hop_length,
            feature_cfg["f0"]["min_hz"],
            feature_cfg["f0"]["max_hz"],
        )
        for index, segment in enumerate(f0_segments):
            mean_f0, std_f0 = summarize_f0(segment)
            values[f"F0_{role}_Session{session_number}_Segment{index}_mean"] = mean_f0
            values[f"F0_{role}_Session{session_number}_Segment{index}_std"] = std_f0
    if feature_cfg["energy"]["enabled"]:
        for index, energy in enumerate(extract_energy(chunks)):
            values[f"Energy_{role}_Session{session_number}_Segment{index}"] = float(energy)
    if feature_cfg["logmel"]["enabled"]:
        logmel_cfg = feature_cfg["logmel"]
        logmel = extract_logmel_3d(
            chunks,
            sr,
            logmel_cfg["n_mels"],
            logmel_cfg["n_fft"],
            logmel_cfg["hop_length"],
            logmel_cfg["max_time_frames"],
        )
        for index, segment in enumerate(logmel):
            path = feature_root / patient_id / f"Session{session_number}" / role.lower() / "logmel"
            values[f"LogMel_{role}_Session{session_number}_Segment{index}"] = save_array(
                segment, path / f"seg{index}.npy"
            )
    return values


def process_session(
    pipeline,
    whisper_model,
    audio_path: Path,
    patient_reference,
    therapist_id: str,
    patient_id: str,
    session_number: int,
    feature_root: Path,
    transcription_root: Path,
    config: dict,
) -> dict[str, object]:
    """Diarize a session, assign roles, and extract all enabled features."""
    result = diarize_audio(pipeline, audio_path, config["diarization"]["num_speakers"])
    patient_speaker = identify_patient_speaker(result, patient_reference)
    roles = assign_speaker_roles(result, patient_speaker)

    # Find the therapist speaker 
    therapist_speaker = next((speaker for role, speaker in roles.items() if role.lower() == "therapist"), None)
    if therapist_speaker is None:
        raise ValueError(f"Therapist speaker was not identified. Available roles: {roles}")

    audio_cfg = config["audio"]
    y, sr = load_and_preprocess_audio(
        str(audio_path),
        audio_cfg["sample_rate"],
        audio_cfg["pre_emphasis"],
        audio_cfg["target_rms"],
    )
    values = {}
    for role, speaker in roles.items():
        role_audio = extract_speaker_audio(y, sr, result.exclusive_annotation, speaker)
        values.update(
            extract_role_features(
                role_audio, sr, role, patient_id, session_number, feature_root, config
            )
        )
        transcription_cfg = config["transcription"]
    if transcription_cfg["enabled"]:
        transcript_path = transcribe_therapist(
            model=whisper_model,
            audio_path=audio_path,
            annotation=result.exclusive_annotation,
            therapist_speaker=therapist_speaker,
            therapist_id=therapist_id,
            patient_id=patient_id,
            session_number=session_number,
            output_root=transcription_root,
            config=transcription_cfg,
            segmentation_config=config["utterance_processing"],
        )
        values[f"Transcript_Therapist_Session{session_number}"] = transcript_path
    return values

def run_pipeline(
    spreadsheet: str,
    audio_root: str,
    output: str,
    feature_root: str,
    transcription_root: str,
    config_path: str,
    hf_token: str,
) -> None:
    """Run diarization and feature extraction for all available patient sessions."""
    config, df = load_config(config_path), load_sessions(spreadsheet)
    
    diarization_cfg = config["diarization"]
    pipeline = create_diarization_pipeline(diarization_cfg["model"], hf_token, diarization_cfg["device"])
    
    transcription_cfg = config["transcription"]
    whisper_model = (create_transcription_model(transcription_cfg) if transcription_cfg["enabled"] else None)
    
    maximum_sessions = config["sessions"]["maximum_per_patient"]

    included_patients = set(config.get("patients", {}).get("include", []))
    included_sessions = {int(session) for session in config.get("sessions", {}).get("include", [])}

    for patient_folder in sorted(Path(audio_root).iterdir()):
        if not patient_folder.is_dir():
            continue
        patient_id = patient_folder.name
        if included_patients and patient_id not in included_patients:
            continue

        line_index = get_patient_row(df, patient_id)
        if line_index is None:
            print(f"Patient {patient_id} was not found in the spreadsheet.")
            continue
        therapist_id = df.at[line_index, "Therapist_ID"]
        if pd.isna(therapist_id):
            print(f"Patient {patient_id}: Therapist_ID was not found.")
            continue
        therapist_id = str(therapist_id)

        patient_reference = create_patient_reference(pipeline, df, line_index, patient_folder, config)      
        for session_number, audio_path in iter_patient_sessions(df, line_index, patient_folder, maximum_sessions):
            if included_sessions and int(session_number) not in included_sessions:
                continue
            if is_session_processed(df, line_index, session_number):
                print(f"Patient {patient_id}, session {session_number}: already processed.")
                continue
            print(f"Patient {patient_id}, session {session_number}: processing.")
            values = process_session(
                pipeline=pipeline,
                whisper_model=whisper_model,
                audio_path=audio_path,
                patient_reference=patient_reference,
                therapist_id=therapist_id,
                patient_id=patient_id,
                session_number=session_number,
                feature_root=Path(feature_root),
                transcription_root=Path(transcription_root),
                config=config,
            )
            for column, value in values.items():
                df.at[line_index, column] = value
            mark_session_processed(df, line_index, session_number)
        save_dataframe(df, output)
        print(f"Partial results saved after patient {patient_id}.")
    save_dataframe(df, output)
    print(f"Pipeline finished. Results saved to {output}.")

def main() -> None:
    """Load command-line/environment settings and run the pipeline."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Diarize sessions and extract acoustic features.")
    parser.add_argument("--spreadsheet", default=os.getenv("INPUT_SPREADSHEET"))
    parser.add_argument("--audio-root", default=os.getenv("AUDIO_ROOT"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_DATAFRAME"))
    parser.add_argument("--feature-root", default=os.getenv("FEATURE_ROOT"))
    parser.add_argument("--config", default=os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml"))
    parser.add_argument("--transcription-root", default=os.getenv("TRANSCRIPTION_ROOT"))
    args = parser.parse_args()
    required = ("spreadsheet", "audio_root", "output", "feature_root", "transcription_root")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing values for: {', '.join(missing)}")
    run_pipeline(
        args.spreadsheet,
        args.audio_root,
        args.output,
        args.feature_root,
        args.transcription_root,
        args.config,
        os.getenv("HF_TOKEN", ""),
    )

if __name__ == "__main__":
    main()

"""Validate diarization for selected patient sessions."""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pyannote.metrics.diarization import DiarizationErrorRate
from depression_nlp.config import load_config
from depression_nlp.data_preprocessing.audio import get_seconds
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
    iter_patient_sessions,
    load_sessions,
)
from depression_nlp.diarization_validation.calculate_der import (
    accumulated_results,
    calculate_session_der,
)
from depression_nlp.storage import save_diarization_csv

def create_patient_reference(
    pipeline,
    df,
    line_index: int,
    patient_folder: Path,
    config: dict,
):
    """Create the patient reference embedding."""
    reference_audio = find_patient_reference_audio(df, line_index, patient_folder)
    result = diarize_audio(pipeline, reference_audio, config["diarization"]["num_speakers"])
    
    start = get_seconds(df.at[line_index, "Patient_Speech_Interval_From"])
    end = get_seconds(df.at[line_index, "Patient_Speech_Interval_To"])

    patient_speaker = identify_reference_speaker(result.annotation, start, end)

    return get_speaker_embedding(result, patient_speaker)


def find_selected_session(
    df,
    line_index: int,
    patient_folder: Path,
    session_number: int,
) -> Path:
    """Find the audio for one selected session."""
    available_sessions = dict(iter_patient_sessions(df, line_index, patient_folder, session_number))

    if session_number not in available_sessions:
        raise FileNotFoundError(
            f"Session {session_number} was not found "
            f"for patient {patient_folder.name}."
        )
    return available_sessions[session_number]


def run_validation(
    spreadsheet: str,
    audio_root: str,
    validation_root: str,
    config_path: str,
    hf_token: str,
) -> None:
    """Run diarization and DER for selected sessions."""

    config = load_config(config_path)
    df = load_sessions(spreadsheet)

    validation_config = config.get("validation",{})
    selected_sessions = validation_config.get("save_diarization_for",{})
    if not selected_sessions:
        raise ValueError(
            "No sessions were configured under "
            "validation.save_diarization_for."
        )

    diarization_config = config["diarization"]
    diarization_pipeline = create_diarization_pipeline(
        diarization_config["model"],
        hf_token,
        diarization_config["device"],
    )
    metric = DiarizationErrorRate(
        collar=float(
            validation_config.get("collar", 0.0)
        ),
        skip_overlap=bool(
            validation_config.get(
                "skip_overlap",
                False,
            )
        ),
    )

    validation_root = Path(validation_root)
    results = []

    for patient_id, session_number in (selected_sessions.items()):
        session_number = int(session_number)
        patient_folder = (Path(audio_root) / patient_id)
        if not patient_folder.is_dir():
            raise FileNotFoundError(
                f"Patient folder not found: "
                f"{patient_folder}"
            )

        line_index = get_patient_row(df, patient_id)
        if line_index is None:
            raise ValueError(
                f"Patient {patient_id} was not found "
                "in the spreadsheet."
            )

        print(
            f"Processing {patient_id}, "
            f"Session{session_number}..."
        )

        patient_reference = create_patient_reference(
            diarization_pipeline,
            df,
            line_index,
            patient_folder,
            config,
        )

        audio_path = find_selected_session(
            df,
            line_index,
            patient_folder,
            session_number,
        )

        diarization_result = diarize_audio(
            diarization_pipeline,
            audio_path,
            diarization_config["num_speakers"],
        )

        patient_speaker = identify_patient_speaker(
            diarization_result,
            patient_reference,
        )

        roles = assign_speaker_roles(
            diarization_result,
            patient_speaker,
        )

        prediction_path = (
            validation_root
            / "predictions"
            / patient_id
            / f"Session{session_number}.csv"
        )

        ground_truth_path = (
            validation_root
            / "ground_truth"
            / patient_id
            / f"Session{session_number}.csv"
        )

        save_diarization_csv(
            annotation=diarization_result.annotation,
            roles=roles,
            output_path=prediction_path,
        )

        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                "Ground truth was not found: "
                f"{ground_truth_path}"
            )

        scores = calculate_session_der(
            ground_truth_path,
            prediction_path,
            metric,
        )

        results.append({
            "patient_id": patient_id,
            "session": session_number,
            **scores,
        })

        print(
            f"{patient_id} Session{session_number}: "
            f"DER = {scores['der_percent']:.2f}%"
        )

    total_scores = accumulated_results(metric)

    results.append({
        "patient_id": "ALL",
        "session": "ALL",
        **total_scores,
    })

    results_path = (
        validation_root / "der_results.csv"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(results).to_csv(
        results_path,
        index=False,
    )

    print(
        f"Overall DER: "
        f"{total_scores['der_percent']:.2f}%"
    )

    print(
        f"Detailed results saved to: "
        f"{results_path}"
    )


def main() -> None:
    """Load settings and run diarization validation."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=("Validate selected diarization sessions."))
    parser.add_argument("--spreadsheet", default=os.getenv("INPUT_SPREADSHEET"))
    parser.add_argument("--audio-root", default=os.getenv("AUDIO_ROOT"))
    parser.add_argument("--validation-root", default=os.getenv( "DIARIZATION_VALIDATION_ROOT", "data/diarization_validation"))
    parser.add_argument("--config", default=os.getenv("PIPELINE_CONFIG", "configs/pipeline.yaml"))
    args = parser.parse_args()
    required_arguments = ("spreadsheet", "audio_root")

    missing_arguments = [
        name
        for name in required_arguments
        if not getattr(args, name)
    ]

    if missing_arguments:
        parser.error(
            "Missing values for: "
            f"{', '.join(missing_arguments)}"
        )

    run_validation(
        spreadsheet=args.spreadsheet,
        audio_root=args.audio_root,
        validation_root=args.validation_root,
        config_path=args.config,
        hf_token=os.getenv("HF_TOKEN", ""),
    )

if __name__ == "__main__":
    main()
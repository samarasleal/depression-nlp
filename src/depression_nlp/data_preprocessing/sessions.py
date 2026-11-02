# Session discovery utilities

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "Patient_ID",
    "Patient_Speech_Audio_Reference",
    "Patient_Speech_Interval_From",
    "Patient_Speech_Interval_To",
}

def load_sessions(spreadsheet_path: str | Path) -> pd.DataFrame:
    """Load and validate the patient-session spreadsheet."""
    df = pd.read_excel(spreadsheet_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not any(column.startswith("FileName_Recording") for column in df.columns):
        raise ValueError("No FileName_Recording columns were found.")
    return df

def get_patient_row(df: pd.DataFrame, patient_id: str) -> int | None:
    """Return the first DataFrame index associated with a patient."""
    rows = df.index[df["Patient_ID"].astype(str) == str(patient_id)]
    return int(rows[0]) if len(rows) else None

def find_patient_reference_audio(
    df: pd.DataFrame, line_index: int, patient_folder: str | Path
) -> Path:
    # Find the audio file containing the annotated patient reference interval
    reference_name = str(df.at[line_index, "Patient_Speech_Audio_Reference"])
    matches = [path for path in Path(patient_folder).glob("*.wav") if path.stem == reference_name]
    if not matches:
        raise FileNotFoundError(f"Patient reference audio was not found: {reference_name}")
    return matches[0]

def iter_patient_sessions(
    df: pd.DataFrame,
    line_index: int,
    patient_folder: str | Path,
    maximum_sessions: int,
):
    """Yield matching session number and audio path pairs."""
    folder = Path(patient_folder)
    for session_number in range(1, maximum_sessions + 1):
        column = f"FileName_Recording{session_number}"
        if column not in df.columns or pd.isna(df.at[line_index, column]):
            continue
        expected_name = str(df.at[line_index, column])
        matches = [path for path in folder.glob("*.wav") if path.stem == expected_name]
        if matches:
            yield session_number, matches[0]

def is_session_processed(df: pd.DataFrame, line_index: int, session_number: int) -> bool:
    column = f"Features_Session{session_number}_Processed"
    if column not in df.columns:
        return False
    value = df.at[line_index, column]
    return pd.notna(value) and bool(value)


def mark_session_processed(df: pd.DataFrame, line_index: int, session_number: int) -> None:
    df.at[line_index, f"Features_Session{session_number}_Processed"] = True

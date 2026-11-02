"""Persistence utilities for tables and feature arrays."""

from pathlib import Path

import numpy as np
import pandas as pd


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a DataFrame and create its parent directory when required."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def save_array(array: np.ndarray, output_path: str | Path, dtype=np.float16) -> str:
    """Save a finite NumPy array and return its absolute path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype)
    np.save(path, clean_array)
    return str(path.resolve())

def save_diarization_csv(annotation, roles: dict, output_path: str | Path) -> str:
    """Save predicted speaker intervals to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    speaker_to_role = {speaker: role.lower() for role, speaker in roles.items()}
    rows = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
            "role": speaker_to_role.get(
                speaker,
                "unknown",
            ),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path.resolve())
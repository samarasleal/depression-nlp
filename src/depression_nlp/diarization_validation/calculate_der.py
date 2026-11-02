# Calculate DER for partially annotated sessions

from pathlib import Path
import pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def parse_time(value) -> float:
    """Convert seconds or an MM:SS/HH:MM:SS value to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(part) for part in str(value).strip().split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported time value: {value}")


def load_annotation(
    csv_path: str | Path,
    label_column: str,
) -> Annotation:
    """Load speaker turns while preserving overlapping speech."""
    df = pd.read_csv(csv_path)
    required_columns = {"start", "end", label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path}: "
            f"{sorted(missing_columns)}"
        )
    annotation = Annotation(uri=Path(csv_path).stem)
    for track, row in df.iterrows():
        start = parse_time(row["start"])
        end = parse_time(row["end"])
        if end <= start:
            raise ValueError(
                f"Invalid interval in {csv_path}: {start}–{end}"
            )
        annotation[
            Segment(start, end),
            int(track),
        ] = str(row[label_column]).lower()

    return annotation


def load_evaluation_windows(
    ground_truth_path: str | Path,
) -> Timeline:
    """Load the three annotated windows from ground truth."""
    df = pd.read_csv(ground_truth_path)
    required_columns = {"window_start", "window_end"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Ground truth must include window_start and "
            f"window_end. Missing: {sorted(missing_columns)}"
        )
    windows = {
        (
            parse_time(row["window_start"]),
            parse_time(row["window_end"]),
        )
        for _, row in df.iterrows()
    }
    segments = []
    for start, end in sorted(windows):
        if end <= start:
            raise ValueError(
                f"Invalid evaluation window: {start}–{end}"
            )
        segments.append(Segment(start, end))
    return Timeline(segments)


def calculate_session_der(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    metric: DiarizationErrorRate,
) -> dict[str, float]:
    """Calculate DER and error components for one session."""
    reference = load_annotation(
        ground_truth_path,
        label_column="speaker",
    )
    hypothesis = load_annotation(
        prediction_path,
        label_column="role",
    )
    evaluation_windows = load_evaluation_windows(
        ground_truth_path
    )
    details = metric(
        reference,
        hypothesis,
        uem=evaluation_windows,
        detailed=True,
    )
    total = details["total"]
    errors = (
        details["false alarm"]
        + details["missed detection"]
        + details["confusion"]
    )
    der = errors / total if total else 0.0
    return {
        "der_percent": der * 100,
        "false_alarm_seconds": details["false alarm"],
        "missed_detection_seconds": details["missed detection"],
        "confusion_seconds": details["confusion"],
        "reference_speech_seconds": total,
    }


def accumulated_results(
    metric: DiarizationErrorRate,
) -> dict[str, float]:
    """Return the duration-weighted DER for all sessions."""
    details = metric[:]
    total = details["total"]
    errors = (
        details["false alarm"]
        + details["missed detection"]
        + details["confusion"]
    )
    der = errors / total if total else 0.0
    return {
        "der_percent": der * 100,
        "false_alarm_seconds": details["false alarm"],
        "missed_detection_seconds": details[
            "missed detection"
        ],
        "confusion_seconds": details["confusion"],
        "reference_speech_seconds": total,
    }
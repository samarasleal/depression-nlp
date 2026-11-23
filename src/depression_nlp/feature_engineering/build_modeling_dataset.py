# Build the session-level feature base and dyad-level outcome table
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from depression_nlp.cbt_taxonomy import INTERVENTION_DOMAINS, INTERVENTION_SUBDOMAINS
from depression_nlp.data_preprocessing.clinical import build_depression_outcomes
from depression_nlp.config import load_config

DYAD_IDS = ["Patient_Code", "Therapist_Code", "Patient_ID", "Therapist_ID"]
SESSION_KEYS = ["Patient_ID", "Session"]
ACOUSTIC_COUNTS = ["n_f0_segments", "n_energy_segments", "n_logmel_segments"]

F0_PATTERN = re.compile(r"F0_Patient_Session(\d+)_Segment(\d+)_(mean|std)$")
ENERGY_PATTERN = re.compile(r"Energy_Patient_Session(\d+)_Segment(\d+)$")
LOGMEL_PATTERN = re.compile(r"LogMel_Patient_Session(\d+)_Segment(\d+)$")

def normalize_ids(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize identifier columns without changing missing values."""
    df = df.copy()
    for column in columns:
        if column in df:
            df[column] = df[column].astype("string").str.strip()
    return df

def numeric_value(value) -> float | None:
    """Return one finite numeric value or None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None

def valid_path(value) -> str | None:
    """Return one non-empty stored path or None."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if not value or value.lower() in {"nan", "none"} else value

def parse_acoustic_columns(columns) -> dict[str, tuple[str, int, int]]:
    """Map wide patient acoustic columns to feature, session, and segment."""
    parsed = {}
    for column in columns:
        if match := F0_PATTERN.fullmatch(str(column)):
            parsed[column] = (f"f0_{match.group(3)}", int(match.group(1)), int(match.group(2)))
        elif match := ENERGY_PATTERN.fullmatch(str(column)):
            parsed[column] = ("energy", int(match.group(1)), int(match.group(2)))
        elif match := LOGMEL_PATTERN.fullmatch(str(column)):
            parsed[column] = ("logmel_path", int(match.group(1)), int(match.group(2)))
    return parsed

def extract_acoustic_segments(features: pd.DataFrame) -> pd.DataFrame:
    """Convert sparse wide acoustic columns into one row per patient segment."""
    if "Patient_ID" not in features:
        raise ValueError("frozen_features.pkl is missing Patient_ID.")
    parsed = parse_acoustic_columns(features.columns)
    if not parsed:
        raise ValueError("No patient F0, energy, or LogMel columns were found.")
    records = {}
    acoustic_columns = list(parsed)
    for _, row in features[["Patient_ID", *acoustic_columns]].iterrows():
        patient_id = str(row["Patient_ID"]).strip()
        for column, value in row[acoustic_columns].dropna().items():
            feature, session, segment = parsed[column]
            item = records.setdefault((patient_id, session, segment), {
                "Patient_ID": patient_id, "Session": session, "Segment": segment,
                "f0_mean": np.nan, "f0_std": np.nan, "energy": np.nan, "logmel_path": None,
            })
            if feature == "logmel_path":
                item[feature] = valid_path(value)
            elif (number := numeric_value(value)) is not None:
                item[feature] = number
    if not records:
        raise ValueError("All patient acoustic values are missing.")
    return pd.DataFrame(records.values()).sort_values(["Patient_ID", "Session", "Segment"])

def aggregate_acoustic_sessions(segments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid segment values while preserving all LogMel paths."""
    rows = []
    for (patient_id, session), group in segments.groupby(SESSION_KEYS, sort=True):
        group = group.sort_values("Segment")
        f0, f0_std, energy = (
            pd.to_numeric(group[column], errors="coerce").dropna()
            for column in ("f0_mean", "f0_std", "energy")
        )
        paths = [path for value in group["logmel_path"] if (path := valid_path(value))]
        rows.append({
            "Patient_ID": patient_id, "Session": int(session),
            "n_f0_segments": len(f0), "f0_mean": f0.mean(),
            "f0_between_segment_std": f0.std(), "f0_within_segment_std_mean": f0_std.mean(),
            "n_energy_segments": len(energy), "energy_mean": energy.mean(), "energy_std": energy.std(),
            "n_logmel_segments": len(paths), "logmel_paths": json.dumps(paths, ensure_ascii=False),
        })
    return pd.DataFrame(rows)

def prepare_profiles(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load unique therapist profiles and retain numeric session features."""
    profiles = pd.read_csv(path)
    required = {*DYAD_IDS, "Session", "intervention_coverage", "intervention_per_speaking_minute"}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Therapist profiles are missing columns: {sorted(missing)}")
    profiles = normalize_ids(profiles, DYAD_IDS)
    profiles["Session"] = pd.to_numeric(profiles["Session"], errors="coerce")
    if profiles[SESSION_KEYS].isna().any().any():
        raise ValueError("Therapist profiles contain missing Patient_ID or Session values.")

    profiles["Session"] = profiles["Session"].astype(int)
    if profiles.duplicated(SESSION_KEYS).any():
        raise ValueError("Therapist profiles contain duplicated Patient_ID/Session rows.")

    audit_columns = [
        "n_eligible_therapist_utterances", "n_intervention_utterances", "n_other_utterances",
        "therapist_speaking_minutes", "intervention_threshold",
    ]
    prevalence = [f"{label}_prevalence" for label in (*INTERVENTION_SUBDOMAINS, *INTERVENTION_DOMAINS)]
    profile_columns = ["intervention_coverage", "intervention_per_speaking_minute", *prevalence]
    missing = set(profile_columns).difference(profiles.columns)
    if missing:
        raise ValueError(f"Therapist profiles are missing modeling columns: {sorted(missing)}")

    numeric_columns = [column for column in [*audit_columns, *profile_columns] if column in profiles]
    profiles[numeric_columns] = profiles[numeric_columns].apply(pd.to_numeric, errors="coerce")
    profiles["_profile_present"] = 1

    metadata = profiles[DYAD_IDS].drop_duplicates()
    if metadata.duplicated("Patient_ID").any():
        raise ValueError("One Patient_ID is associated with multiple dyad identifiers.")

    session_profiles = profiles[[*SESSION_KEYS, *numeric_columns, "_profile_present"]].copy()
    return session_profiles, metadata

def build_session_base(features: pd.DataFrame, profile_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join patient acoustic features and therapist profiles by observed session."""
    if "Patient_ID" not in features:
        raise ValueError("frozen_features.pkl is missing Patient_ID.")

    features = normalize_ids(features, ["Patient_ID"])
    if features["Patient_ID"].isna().any():
        raise ValueError("frozen_features.pkl contains missing Patient_ID values.")

    segments = extract_acoustic_segments(features)
    acoustic = aggregate_acoustic_sessions(segments)
    profiles, metadata = prepare_profiles(profile_path)

    base = profiles.merge(acoustic, on=SESSION_KEYS, how="outer", validate="one_to_one")
    base = base.merge(metadata, on="Patient_ID", how="left", validate="many_to_one")

    count_columns = [*ACOUSTIC_COUNTS, "_profile_present"]
    base[count_columns] = base[count_columns].fillna(0).astype(int)
    base["logmel_paths"] = base["logmel_paths"].fillna("[]")

    first = [*DYAD_IDS, "Session"]
    base = base[first + [column for column in base if column not in first]].sort_values(SESSION_KEYS)
    return base.reset_index(drop=True), metadata

def build_dyad_outcomes(
    features: pd.DataFrame, metadata: pd.DataFrame, session_base: pd.DataFrame, max_hdrs: float, max_cdi: float, min_cdi: float) -> pd.DataFrame:
    """Join clinical outcomes, dyad identifiers, and session counts."""
    outcomes = build_depression_outcomes(features, max_hdrs, max_cdi, min_cdi)
    outcomes = outcomes.merge(metadata, on="Patient_ID", how="left", validate="one_to_one")

    acoustic = session_base[ACOUSTIC_COUNTS].gt(0).any(axis=1).astype(int)
    profile = session_base["_profile_present"].astype(int)
    counts = session_base.assign(_acoustic=acoustic, _profile=profile, _multimodal=acoustic * profile)
    counts = counts.groupby("Patient_ID", as_index=False).agg(
        n_observed_sessions=("Session", "nunique"),
        n_acoustic_sessions=("_acoustic", "sum"),
        n_therapist_profile_sessions=("_profile", "sum"),
        n_multimodal_sessions=("_multimodal", "sum"),
    )
    outcomes = outcomes.merge(counts, on="Patient_ID", how="inner", validate="one_to_one")
    return outcomes[DYAD_IDS + [column for column in outcomes if column not in DYAD_IDS]].sort_values("Patient_ID")

def run(features_path: str, profiles_path: str, output_root: str, config_path: str, *, excel_pt: bool = False) -> None:
    """Build and save the two canonical modeling tables."""
    config = load_config(config_path)
    outcome_config = config["modeling"]["outcome"]
    max_hdrs = float(outcome_config["max_hdrs"])
    max_cdi = float(outcome_config["max_cdi"])
    min_cdi = float(outcome_config["min_cdi"])
    features_path, profiles_path = Path(features_path), Path(profiles_path)

    if not features_path.is_file():
        raise FileNotFoundError(f"Frozen features were not found: {features_path}")
    if not profiles_path.is_file():
        raise FileNotFoundError(f"Therapist profiles were not found: {profiles_path}")
    if max_hdrs <= 0 or max_cdi <= min_cdi:
        raise ValueError("Invalid HDRS/CDI normalization limits.")

    features = pd.read_pickle(features_path)
    session_base, metadata = build_session_base(features, profiles_path)
    outcomes = build_dyad_outcomes(features, metadata, session_base, max_hdrs, max_cdi, min_cdi)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    session_path = output_root / "session_feature_base.csv"
    outcome_path = output_root / "dyad_outcomes.csv"

    session_csv = session_base.drop(columns="_profile_present")
    session_csv.to_csv(session_path, index=False)
    outcomes.to_csv(outcome_path, index=False)

    if excel_pt:
        excel_path = output_root / "session_feature_base_excel_pt.csv"
        precision = {"therapist_speaking_minutes": 2, "intervention_coverage": 4, "intervention_per_speaking_minute": 2}
        session_csv.round(precision).to_csv(excel_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"Excel session feature base saved: {excel_path}")

    print(f"Session feature base saved: {session_path} ({len(session_base)} rows)")
    print(
        f"Dyad outcomes saved: {outcome_path} ({len(outcomes)} rows, "
        f"{outcomes['outcome_available'].sum()} continuous and "
        f"{outcomes['binary_outcome_available'].sum()} binary outcomes)"
    )

def main() -> None:
    """Build modeling datasets from the command line."""
    parser = argparse.ArgumentParser(description="Build session features and dyad outcomes.")
    parser.add_argument("--features", required=True, help="Path to frozen_features.pkl")
    parser.add_argument("--profiles", required=True, help="Path to therapist_session_profiles.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", dest="config_path", required=True, help="Path to the project configuration")
    parser.add_argument("--excel-pt", action="store_true", help="Also export a session CSV with decimal commas")
    args = parser.parse_args()
    run(args.features, args.profiles, args.output_root, args.config_path, excel_pt=args.excel_pt)


if __name__ == "__main__":
    main()

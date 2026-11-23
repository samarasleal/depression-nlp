# Create utterance-level predictions and session-level therapist profiles.
import argparse
import json
from pathlib import Path
import pandas as pd
from depression_nlp.cbt_taxonomy import (INTERVENTION_DOMAINS, INTERVENTION_SUBDOMAINS, SUBDOMAIN_TO_DOMAIN)
from depression_nlp.data_preprocessing.sessions import (load_reference, normalize_integer)
from depression_nlp.config import load_model_evaluation_settings

SUBDOMAINS = INTERVENTION_SUBDOMAINS
DOMAINS = INTERVENTION_DOMAINS

def merge_intervals(intervals: list[tuple[float, float]]) -> float:
    """Return the duration covered by overlapping speech intervals."""
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    total, current_start, current_end = 0.0, valid[0][0], valid[0][1]
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start

def predict_utterance(label_scores: dict, top_k: int, threshold: float) -> tuple[float, list[str], list[str]]:
    """Classify one utterance as OTHER or retain its top intervention labels."""
    missing = set(SUBDOMAINS).difference(label_scores)
    if missing:
        raise ValueError(f"label_scores is missing labels: {sorted(missing)}")
    scores = {label: float(label_scores[label]) for label in SUBDOMAINS}
    intervention_score = max(scores.values())
    if intervention_score < threshold:
        return intervention_score, [], []
    labels = sorted(SUBDOMAINS, key=lambda label: (-scores[label], label))[:top_k]
    domains = sorted({SUBDOMAIN_TO_DOMAIN[label] for label in labels})
    return intervention_score, labels, domains

def extract_utterances(input_root: str | Path, reference: dict[str, dict], top_k: int, threshold: float) -> pd.DataFrame:
    """Extract predictions for every eligible therapist utterance."""
    rows = []
    for json_path in sorted(Path(input_root).rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("output_mode") == "candidates":
            raise ValueError(f"{json_path} contains only annotation candidates. Therapist profiles require all utterances.")

        patient_id = str(data.get("patient_id", "")).strip()
        therapist_id = str(data.get("therapist_id", "")).strip()
        if patient_id not in reference:
            raise ValueError(f"Patient_ID '{patient_id}' from {json_path} was not found in the reference spreadsheet.")

        codes = reference[patient_id]
        expected_therapist = str(codes["Therapist_ID"]).strip()
        if therapist_id and therapist_id != expected_therapist:
            raise ValueError(f"Therapist mismatch for {patient_id}: JSON={therapist_id}, spreadsheet={expected_therapist}")

        session = normalize_integer(data.get("session"), "session")
        for utterance in data.get("utterances", []):
            scores = utterance.get("label_scores")
            if not scores:
                raise ValueError(f"Utterance without label_scores in {json_path}.")

            start, end = float(utterance["start"]), float(utterance["end"])
            if end <= start:
                raise ValueError(f"Invalid interval in {json_path}: {start}–{end}")

            intervention_score, labels, domains = predict_utterance(scores, top_k, threshold)
            predicted_class = "INTERVENTION" if labels else "OTHER"

            row = {
                "Patient_Code": str(codes["Patient_Code"]).strip(),
                "Therapist_Code": str(codes["Therapist_Code"]).strip(),
                "Patient_ID": patient_id,
                "Therapist_ID": expected_therapist,
                "Session": session,
                "Utterance_id": normalize_integer(utterance.get("utterance_id"), "utterance_id"),
                "Utterance_uid": utterance.get("utterance_uid"),
                "Start": start,
                "End": end,
                "Duration_seconds": round(end - start, 3),
                "Utterance": utterance.get("text", "").strip(),
                "Intervention_score": round(intervention_score, 6),
                "Intervention_threshold": threshold,
                "Predicted_class": predicted_class,
                "_predicted_labels": labels,
                "_predicted_domains": domains,
                "Source_json": str(json_path),
            }

            displayed_labels = labels if labels else ["OTHER"]
            for index in range(3):
                row[f"Predicted_label_{index + 1}"] = (
                    displayed_labels[index] if index < len(displayed_labels) else None
                )
                row[f"Predicted_domain_{index + 1}"] = (
                    domains[index] if index < len(domains) else None
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No therapist utterances were found in {input_root}.")

    utterances = pd.DataFrame(rows)
    keys = ["Patient_Code", "Therapist_Code", "Session", "Utterance_id"]
    if utterances.duplicated(keys).any():
        duplicates = utterances.loc[utterances.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicated utterance keys were found:\n{duplicates.to_string(index=False)}")
    return utterances

def create_session_profiles(utterances: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate utterance predictions into one profile per dyad and session."""
    identifiers = ["Patient_Code", "Therapist_Code", "Patient_ID", "Therapist_ID", "Session"]
    profiles = []

    for values, session in utterances.groupby(identifiers, sort=True):
        row = dict(zip(identifiers, values))
        interventions = session.loc[session["Predicted_class"].eq("INTERVENTION")]
        n_eligible, n_interventions = len(session), len(interventions)
        speaking_seconds = merge_intervals(list(zip(session["Start"], session["End"])))
        speaking_minutes = speaking_seconds / 60
        row.update({
            "n_eligible_therapist_utterances": n_eligible,
            "n_intervention_utterances": n_interventions,
            "n_other_utterances": n_eligible - n_interventions,
            "intervention_coverage": n_interventions / n_eligible if n_eligible else 0.0,
            "intervention_per_speaking_minute": n_interventions / speaking_minutes if speaking_minutes else 0.0,
            "therapist_speaking_minutes": speaking_minutes,
            "intervention_threshold": threshold})

        subdomain_counts = {
            label: sum(label in labels for labels in interventions["_predicted_labels"])
            for label in SUBDOMAINS
        }
        domain_counts = {
            domain: sum(domain in domains for domains in interventions["_predicted_domains"])
            for domain in DOMAINS
        }

        for label in SUBDOMAINS:
            row[f"{label}_prevalence"] = (subdomain_counts[label] / n_interventions if n_interventions else 0.0)

        ranked = sorted(((label, count) for label, count in subdomain_counts.items() if count > 0), key=lambda item: (-item[1], item[0]))
        for position in range(3):
            if position < len(ranked):
                label, count = ranked[position]
                prevalence = count / n_interventions
            else:
                label, count, prevalence = None, 0, 0.0

            row[f"top_{position + 1}_subdomain"] = label
            row[f"top_{position + 1}_count"] = count
            row[f"top_{position + 1}_prevalence"] = prevalence

        for domain in DOMAINS:
            row[f"{domain}_prevalence"] = (domain_counts[domain] / n_interventions if n_interventions else 0.0)
        profiles.append(row)

    columns = identifiers + [
        "n_eligible_therapist_utterances",
        "n_intervention_utterances",
        "n_other_utterances",
        "intervention_coverage",
        "intervention_per_speaking_minute",
        "therapist_speaking_minutes",
    ]

    for position in range(1, 4):
        columns.extend([
            f"top_{position}_subdomain",
            f"top_{position}_count",
            f"top_{position}_prevalence",
        ])
    columns.extend(f"{label}_prevalence" for label in SUBDOMAINS)
    columns.extend(f"{domain}_prevalence" for domain in DOMAINS)
    columns.append("intervention_threshold")
    return pd.DataFrame(profiles)[columns]

def run_pipeline(input_root: str | Path, reference_spreadsheet: str | Path, output_root: str | Path, config_path: str | Path) -> None:
    """Create utterance predictions and session-level therapist profiles."""
    top_k, threshold = load_model_evaluation_settings(config_path, require_threshold=True)
    if threshold is None:
        raise ValueError("The intervention threshold is required.")
    reference = load_reference(reference_spreadsheet)
    utterances = extract_utterances(input_root, reference, top_k, threshold)
    profiles = create_session_profiles(utterances, threshold)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    utterance_output = utterances.drop(columns=["_predicted_labels", "_predicted_domains"])
    utterance_output.to_csv(output_root / "therapist_utterance_predictions.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(output_root / "therapist_session_profiles.csv", index=False, encoding="utf-8-sig")

    print(f"{len(utterances)} utterances saved to therapist_utterance_predictions.csv.")
    print(f"{len(profiles)} session profiles saved to therapist_session_profiles.csv.")
    print(f"Results saved to {output_root.resolve()}")

def main() -> None:
    """Create therapist profiles from the command line."""
    parser = argparse.ArgumentParser(description="Create therapist intervention profiles.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference-spreadsheet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run_pipeline(input_root=args.input_root, reference_spreadsheet=args.reference_spreadsheet, output_root=args.output_root, config_path=args.config)

if __name__ == "__main__":
    main()
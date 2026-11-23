# Extract selected utterances and their model predictions from JSON files
import argparse
import json
from pathlib import Path
import pandas as pd
from depression_nlp.data_preprocessing.sessions import (load_reference, normalize_integer)
from depression_nlp.config import load_model_evaluation_settings

def get_top_labels(label_scores: dict, top_k: int) -> list[str]:
    """Return the labels with the highest NLI scores."""
    if not label_scores:
        raise ValueError("label_scores is empty.")
    ranked = sorted(label_scores.items(), key=lambda item: (-float(item[1]), item[0]))
    return [label for label, _ in ranked[:top_k]]

def extract_predictions(input_root: str | Path, reference_spreadsheet: str | Path, output_path: str | Path, config_path: str | Path) -> Path:
    """Extract selected utterances into one prediction table."""
    top_k, threshold = load_model_evaluation_settings(config_path)
    reference = load_reference(reference_spreadsheet)
    rows = []
    for json_path in sorted(Path(input_root).rglob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        patient_id = str(data.get("patient_id", "")).strip()
        therapist_id = str(data.get("therapist_id", "")).strip()

        if not patient_id or patient_id not in reference:
            raise ValueError(f"Patient_ID '{patient_id}' from {json_path} was not found in the reference spreadsheet.")

        codes = reference[patient_id]
        expected_therapist = str(codes["Therapist_ID"]).strip()
        if therapist_id and therapist_id != expected_therapist:
            raise ValueError(f"Therapist mismatch for {patient_id}: JSON={therapist_id}, spreadsheet={expected_therapist}")

        session = normalize_integer(data.get("session"), "session")
        for utterance in data.get("utterances", []):
            if not utterance.get("selected_for_annotation", False):
                continue

            scores = utterance.get("label_scores", {})
            top_labels = get_top_labels(scores, top_k)
            intervention_score = max(float(score) for score in scores.values())
            if threshold is None:
                predicted_labels, status = top_labels, "threshold_pending"
            elif intervention_score < threshold:
                predicted_labels, status = ["OTHER"], "ready"
            else:
                predicted_labels, status = top_labels, "ready"
            row = {
                "Patient_Code": str(codes["Patient_Code"]).strip(),
                "Therapist_Code": str(codes["Therapist_Code"]).strip(),
                "Patient_ID": patient_id,
                "Therapist_ID": expected_therapist,
                "Session": session,
                "Utterance_id": normalize_integer(utterance.get("utterance_id"), "utterance_id"),
                "Utterance": utterance.get("text", "").strip(),
                "Utterance_uid": utterance.get("utterance_uid"),
                "Selection_source": utterance.get("selection_source"),
                "Intervention_score": round(intervention_score, 6),
                "Threshold": threshold,
                "Prediction_status": status,
                "Source_json": str(json_path)
            }
            for index in range(top_k):
                row[f"Top_label_{index + 1}"] = top_labels[index]
                row[f"Predicted_label_{index + 1}"] = predicted_labels[index] if index < len(predicted_labels) else None
            rows.append(row)
    if not rows:
        raise ValueError(f"No selected utterances were found in {input_root}.")

    predictions = pd.DataFrame(rows)
    keys = ["Patient_Code", "Therapist_Code", "Session", "Utterance_id"]
    if predictions.duplicated(keys).any():
        duplicates = predictions.loc[predictions.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicated prediction keys were found:\n{duplicates.to_string(index=False)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{len(predictions)} selected utterances saved to {output_path.resolve()}")

    if threshold is None:
        print("Intervention threshold is pending. Top-3 labels were preserved, but final evaluation is not ready.")
    return output_path

def main() -> None:
    """Extract model predictions from the command line."""
    parser = argparse.ArgumentParser(description="Extract selected intervention predictions.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--reference-spreadsheet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    extract_predictions(
        input_root=args.input_root,
        reference_spreadsheet=args.reference_spreadsheet,
        output_path=args.output,
        config_path=args.config,
    )

if __name__ == "__main__":
    main()
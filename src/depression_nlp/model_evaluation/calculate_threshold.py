# Calculate the intervention threshold using random annotated utterances
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from depression_nlp.config import (load_config, load_settings)
from depression_nlp.data_preprocessing.sessions import normalize_integer
from depression_nlp.model_evaluation.evaluate_predictions import (KEYS, read_annotations)

def load_threshold_data(annotation_path: str | Path, prediction_path: str | Path, calibration_source: str) -> pd.DataFrame:
    """Merge clinical labels with intervention scores."""
    annotations = read_annotations(annotation_path)
    predictions = pd.read_csv(prediction_path, dtype={"Patient_code": str, "Therapist_code": str})
    required = set(KEYS + ["Selection_source", "Intervention_score"])
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    predictions["Patient_code"] = predictions["Patient_code"].str.strip()
    predictions["Therapist_code"] = predictions["Therapist_code"].str.strip()
    predictions["Session"] = predictions["Session"].map(lambda value: normalize_integer(value, "Session"))
    predictions["Utterance_id"] = predictions["Utterance_id"].map(lambda value: normalize_integer(value, "Utterance_id"))
    if predictions.duplicated(KEYS).any():
        raise ValueError("The prediction file contains duplicated utterance keys.")

    columns = KEYS + ["Selection_source", "Intervention_score"]
    data = annotations[KEYS + ["Gold_labels"]].merge(predictions[columns], on=KEYS, how="left", validate="one_to_one")
    if data["Intervention_score"].isna().any():
        missing_rows = data.loc[data["Intervention_score"].isna(), KEYS].head(10)
        raise ValueError(f"Annotations without intervention scores:\n{missing_rows.to_string(index=False)}")

    if calibration_source != "all":
        mask = data["Selection_source"].fillna("").str.lower().eq(calibration_source)
        data = data.loc[mask].copy()
    if data.empty:
        raise ValueError(f"No utterances found for calibration_source='{calibration_source}'.")

    data["Gold_intervention"] = data["Gold_labels"].map(lambda labels: int("other" not in labels))
    data["Gold_class"] = data["Gold_intervention"].map({0: "OTHER", 1: "INTERVENTION"})
    data["Dyad"] = data["Patient_code"] + "_" + data["Therapist_code"]
    if data["Gold_intervention"].nunique() != 2:
        raise ValueError("Threshold calculation requires both OTHER and INTERVENTION examples.")
    return data.reset_index(drop=True)

def create_grouped_split(data: pd.DataFrame, n_splits: int, test_fold: int, random_state: int) -> pd.DataFrame:
    """Create development and test sets without splitting dyads."""
    if data["Dyad"].nunique() < n_splits:
        raise ValueError(f"At least {n_splits} dyads are required for {n_splits} grouped folds.")
    if not 0 <= test_fold < n_splits:
        raise ValueError(f"test_fold must be between 0 and {n_splits - 1}.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.full(len(data), -1, dtype=int)

    for fold, (_, indices) in enumerate(splitter.split(data, data["Gold_intervention"], groups=data["Dyad"])):
        folds[indices] = fold

    data = data.copy()
    data["Fold"] = folds
    data["Evaluation_split"] = np.where(data["Fold"].eq(test_fold), "test", "development")

    for split_name, frame in data.groupby("Evaluation_split"):
        if frame["Gold_intervention"].nunique() != 2:
            raise ValueError(
                f"The {split_name} split does not contain both OTHER and INTERVENTION. "
                "Change test_fold, random_state, or reduce n_splits.")
    return data

def calculate_binary_metrics(y_true, y_pred) -> dict:
    """Calculate metrics for INTERVENTION versus OTHER."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_utterances": len(y_true),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "other_precision": precision[0],
        "other_recall": recall[0],
        "other_f1": f1[0],
        "other_support": int(support[0]),
        "intervention_precision": precision[1],
        "intervention_recall": recall[1],
        "intervention_f1": f1[1],
        "intervention_support": int(support[1]),
        "true_other": int(tn),
        "false_intervention": int(fp),
        "missed_intervention": int(fn),
        "true_intervention": int(tp),
    }

def find_best_threshold(development: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Choose the threshold with the highest binary macro-F1."""
    y_true = development["Gold_intervention"].to_numpy()
    scores = development["Intervention_score"].astype(float).to_numpy()
    rows = []

    for threshold in np.linspace(0, 1, 1001):
        y_pred = (scores >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 3),
            **calculate_binary_metrics(y_true, y_pred),
        })

    search = pd.DataFrame(rows)
    best = search.sort_values(
        ["macro_f1", "balanced_accuracy", "other_recall", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]

    return float(best["threshold"]), search

def evaluate_split(frame: pd.DataFrame, threshold: float, split_name: str) -> dict:
    """Evaluate one split using the selected threshold."""
    y_true = frame["Gold_intervention"].to_numpy()
    y_pred = (frame["Intervention_score"].astype(float).to_numpy() >= threshold).astype(int)
    return {"split": split_name, "threshold": threshold, **calculate_binary_metrics(y_true, y_pred)}

def calculate_threshold(annotation_path: str | Path, prediction_path: str | Path, output_root: str | Path, config_path: str | Path) -> float:
    """Select the threshold on development data and evaluate it on held-out dyads."""
    source, n_splits, test_fold, random_state = load_settings(config_path)
    data = load_threshold_data(annotation_path, prediction_path, source)
    data = create_grouped_split(data, n_splits, test_fold, random_state)

    development = data.loc[data["Evaluation_split"].eq("development")].copy()
    test = data.loc[data["Evaluation_split"].eq("test")].copy()
    threshold, search = find_best_threshold(development)

    data["Predicted_intervention"] = data["Intervention_score"].astype(float).ge(threshold)
    data["Predicted_class"] = data["Predicted_intervention"].map({False: "OTHER", True: "INTERVENTION"})

    metrics = pd.DataFrame([evaluate_split(development, threshold, "development"), evaluate_split(test, threshold, "test")])

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    export_columns = KEYS + [
        "Selection_source",
        "Intervention_score",
        "Gold_class",
        "Predicted_class",
        "Dyad",
        "Fold",
        "Evaluation_split",
    ]
    data[export_columns].to_csv(output_root / "threshold_assignments.csv", index=False, encoding="utf-8-sig")
    search.to_csv(output_root / "threshold_search.csv", index=False)
    metrics.to_csv(output_root / "threshold_metrics.csv", index=False)
    threshold_config = {"model_evaluation": {"intervention_threshold": threshold}}
    Path(output_root / "selected_threshold.yaml").write_text(yaml.safe_dump(threshold_config, sort_keys=False), encoding="utf-8")
    print(f"Selected intervention threshold: {threshold:.3f}")
    print(f"Development utterances: {len(development)}")
    print(f"Test utterances: {len(test)}")
    print(f"Results saved to {output_root.resolve()}")
    print("Copy the selected threshold to configs/pipeline.yaml.")
    return threshold

def main() -> None:
    """Calculate the intervention threshold from the command line."""
    parser = argparse.ArgumentParser(description="Calculate the INTERVENTION versus OTHER threshold.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    calculate_threshold(annotation_path=args.annotations, prediction_path=args.predictions, output_root=args.output_root, config_path=args.config)

if __name__ == "__main__":
    main()
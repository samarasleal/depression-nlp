# Compare model predictions with the adjudicated clinical annotations
import argparse
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, jaccard_score, precision_recall_fscore_support, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer
from depression_nlp.cbt_taxonomy import (ALL_DOMAIN_LABELS, ALL_SUBDOMAIN_LABELS, LABEL_ALIASES, SUBDOMAIN_TO_DOMAIN)
from depression_nlp.data_preprocessing.sessions import normalize_integer

KEYS = ["Patient_code", "Therapist_code", "Session", "Utterance_id"]
SUBDOMAIN_LABELS = list(ALL_SUBDOMAIN_LABELS)
DOMAIN_LABELS = list(ALL_DOMAIN_LABELS)

def normalize_label(value) -> str | None:
    """Convert spreadsheet labels to the internal label names."""
    if pd.isna(value) or not str(value).strip():
        return None
    label = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    label = LABEL_ALIASES.get(label, label)
    if label not in SUBDOMAIN_TO_DOMAIN:
        raise ValueError(f"Unknown annotation label: {value}")
    return label

def make_label_set(values) -> frozenset[str]:
    """Create one validated multilabel set."""
    labels = frozenset(label for value in values if (label := normalize_label(value)))
    if "other" in labels and len(labels) > 1:
        raise ValueError("OTHER must not be combined with an intervention label.")
    return labels

def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read and combine every dyad sheet from the annotation workbook."""
    sheets = pd.read_excel(path, sheet_name=None)
    required = set(KEYS + ["Utterance", "Label 1", "Label 2", "Label 3"])
    frames = []
    for sheet_name, frame in sheets.items():
        if frame.dropna(how="all").empty:
            continue
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Sheet '{sheet_name}' is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["Annotation_sheet"] = sheet_name
        frames.append(frame)

    if not frames:
        raise ValueError("No annotation sheets were found.")

    annotations = pd.concat(frames, ignore_index=True)
    annotations = annotations.dropna(subset=["Patient_code", "Therapist_code", "Session", "Utterance_id"])
    annotations["Patient_code"] = annotations["Patient_code"].astype(str).str.strip()
    annotations["Therapist_code"] = annotations["Therapist_code"].astype(str).str.strip()
    annotations["Session"] = annotations["Session"].map(lambda value: normalize_integer(value, "Session"))
    annotations["Utterance_id"] = annotations["Utterance_id"].map(lambda value: normalize_integer(value, "Utterance_id"))
    annotations["Gold_labels"] = annotations.apply(lambda row: make_label_set([row["Label 1"], row["Label 2"], row["Label 3"]]), axis=1)

    unannotated = annotations["Gold_labels"].map(len).eq(0)
    if unannotated.any():
        print(f"Skipping {int(unannotated.sum())} rows without clinical labels.")
        annotations = annotations.loc[~unannotated].copy()

    if annotations.duplicated(KEYS).any():
        raise ValueError("The annotation workbook contains duplicated utterance keys.")
    return annotations

def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read final model predictions produced from the JSON files."""
    predictions = pd.read_csv(path)
    required = set(KEYS + ["Prediction_status", "Selection_source", "Predicted_label_1", "Predicted_label_2", "Predicted_label_3"])
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    if predictions["Prediction_status"].ne("ready").any():
        raise ValueError("Final evaluation cannot run while intervention_threshold is pending.")
    predictions["Patient_code"] = predictions["Patient_code"].astype(str).str.strip()
    predictions["Therapist_code"] = predictions["Therapist_code"].astype(str).str.strip()
    predictions["Session"] = predictions["Session"].map(lambda value: normalize_integer(value, "Session"))
    predictions["Utterance_id"] = predictions["Utterance_id"].map(lambda value: normalize_integer(value, "Utterance_id"))
    predictions["Predicted_labels"] = predictions.apply(lambda row: make_label_set([row["Predicted_label_1"], row["Predicted_label_2"], row["Predicted_label_3"]]), axis=1)
    return predictions

def to_domains(labels: frozenset[str]) -> frozenset[str]:
    return frozenset(SUBDOMAIN_TO_DOMAIN[label] for label in labels)

def calculate_metrics(gold_sets, predicted_sets, classes: list[str]) -> dict:
    """Calculate multilabel classification metrics."""
    encoder = MultiLabelBinarizer(classes=classes)
    encoder.fit([classes])
    y_true = encoder.transform(gold_sets)
    y_pred = encoder.transform(predicted_sets)
    return {
        "n_utterances": len(gold_sets),
        "exact_match": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_samples": precision_score(y_true, y_pred, average="samples", zero_division=0),
        "recall_samples": recall_score(y_true, y_pred, average="samples", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred)}

def add_scope_metrics(rows: list[dict], scope: str, frame: pd.DataFrame) -> None:
    """Add subdomain and domain metrics for one evaluation subset."""
    if frame.empty:
        return
    for level, gold_column, predicted_column, classes in [
        ("subdomain", "Gold_labels", "Predicted_labels", SUBDOMAIN_LABELS),
        ("domain", "Gold_domains", "Predicted_domains", DOMAIN_LABELS),
    ]:
        metrics = calculate_metrics(frame[gold_column], frame[predicted_column], classes)
        rows.append({"scope": scope, "level": level, **metrics})

def calculate_per_label_metrics(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Calculate precision, recall, F1, and support for every label."""
    if level == "subdomain":
        gold_sets, predicted_sets, classes = frame["Gold_labels"], frame["Predicted_labels"], SUBDOMAIN_LABELS
    else:
        gold_sets, predicted_sets, classes = frame["Gold_domains"], frame["Predicted_domains"], DOMAIN_LABELS
    encoder = MultiLabelBinarizer(classes=classes)
    encoder.fit([classes])
    y_true, y_pred = encoder.transform(gold_sets), encoder.transform(predicted_sets)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return pd.DataFrame({
        "level": level,
        "label": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support})

def evaluate(annotation_path: str | Path, prediction_path: str | Path, output_root: str | Path) -> None:
    """Merge annotations with predictions and calculate final metrics."""
    annotations = read_annotations(annotation_path)
    predictions = read_predictions(prediction_path)

    prediction_columns = KEYS + [
        "Selection_source",
        "Intervention_score",
        "Threshold",
        "Predicted_label_1",
        "Predicted_label_2",
        "Predicted_label_3",
        "Predicted_labels",
        "Utterance_uid",
        "Source_json",
    ]
    merged = annotations.merge(predictions[prediction_columns], on=KEYS, how="left", validate="one_to_one")

    missing = merged["Predicted_labels"].isna()
    if missing.any():
        examples = merged.loc[missing, KEYS].head(10)
        raise ValueError(f"Annotations without matching predictions:\n{examples.to_string(index=False)}")

    merged["Gold_domains"] = merged["Gold_labels"].map(to_domains)
    merged["Predicted_domains"] = merged["Predicted_labels"].map(to_domains)
    merged["Correct_labels"] = merged.apply(lambda row: row["Gold_labels"] & row["Predicted_labels"], axis=1)
    merged["False_positive_labels"] = merged.apply(lambda row: row["Predicted_labels"] - row["Gold_labels"], axis=1)
    merged["False_negative_labels"] = merged.apply(lambda row: row["Gold_labels"] - row["Predicted_labels"], axis=1)
    merged["Exact_match"] = merged["Gold_labels"] == merged["Predicted_labels"]
    merged["Jaccard"] = merged.apply(
        lambda row: len(row["Gold_labels"] & row["Predicted_labels"]) / len(row["Gold_labels"] | row["Predicted_labels"]),
        axis=1,
    )

    metrics = []
    add_scope_metrics(metrics, "overall", merged)

    random_mask = merged["Selection_source"].fillna("").str.lower().eq("random")
    add_scope_metrics(metrics, "random", merged.loc[random_mask])
    add_scope_metrics(metrics, "enriched", merged.loc[~random_mask])

    for (patient, therapist), group in merged.groupby(["Patient_code", "Therapist_code"]):
        add_scope_metrics(metrics, f"dyad:{patient}_{therapist}", group)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    display_columns = KEYS + [
        "Annotation_sheet",
        "Utterance",
        "Selection_source",
        "Intervention_score",
        "Threshold",
        "Gold_labels",
        "Predicted_labels",
        "Correct_labels",
        "False_positive_labels",
        "False_negative_labels",
        "Exact_match",
        "Jaccard",
    ]
    details = merged[display_columns].copy()
    for column in ["Gold_labels", "Predicted_labels", "Correct_labels", "False_positive_labels", "False_negative_labels"]:
        details[column] = details[column].map(lambda labels: ";".join(sorted(labels)))

    details.to_csv(output_root / "evaluation_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(output_root / "evaluation_metrics.csv", index=False)

    per_label = pd.concat([
        calculate_per_label_metrics(merged, "subdomain"),
        calculate_per_label_metrics(merged, "domain"),
    ], ignore_index=True)
    per_label.to_csv(output_root / "per_label_metrics.csv", index=False)

    print(f"Evaluation completed for {len(merged)} annotated utterances.")
    print(f"Results saved to {output_root.resolve()}")

def main() -> None:
    """Evaluate model predictions from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate intervention predictions.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    evaluate(annotation_path=args.annotations, prediction_path=args.predictions, output_root=args.output_root)

if __name__ == "__main__":
    main()
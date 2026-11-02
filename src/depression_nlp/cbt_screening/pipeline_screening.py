"""Score CBT relevance and select utterances for human annotation."""

import argparse
import json
import random
from pathlib import Path

import yaml

from depression_nlp.cbt_screening.nli import NLIClassifier


def select_candidates(utterances: list[dict], config: dict, patient_id: str, session: int) -> list[dict]:
    """Select CBT-enriched, random, and reserve utterances."""
    ordered = sorted(utterances, key=lambda item: item["cbt_enrichment_score"], reverse=True)
    enriched = ordered[:config["n_enriched"]]
    remaining = ordered[config["n_enriched"]:]
    rng = random.Random(f'{config["random_seed"]}:{patient_id}:{session}')

    random_items = [rng.choice(remaining)] if remaining else []
    random_ids = {item["utterance_id"] for item in random_items}
    reserve_pool = [item for item in remaining if item["utterance_id"] not in random_ids]
    reserves = reserve_pool[:config["n_reserve"]]

    selected = []
    for source, items in (
        ("cbt_enriched", enriched),
        ("random", random_items),
        ("reserve", reserves),
    ):
        for item in items:
            candidate = dict(item)
            candidate["selection_source"] = source
            candidate["gold_label"] = None
            candidate["exclude_sensitive"] = None
            selected.append(candidate)
    return selected


def process_json(path: Path, output_path: Path, classifier: NLIClassifier, config: dict,) -> None:
    """Score all utterances and tag those selected for annotation."""
    data = json.loads(path.read_text(encoding="utf-8"))
    utterances = data.get("utterances", [])

    if not utterances:
        print(f"No utterances found: {path}")
        return

    scores = classifier.score(
        [item["text"] for item in utterances],
        config["hypotheses"],
        config["batch_size"],
    )

    scored = []
    cbt_labels = config["cbt_enrichment_labels"]
    for utterance, score in zip(utterances, scores):
        item = dict(utterance)
        item.update(score)
        # Use only Domain 1 labels for CBT enrichment
        item["cbt_enrichment_score"] = round(max(item["label_scores"][label] for label in cbt_labels), 6)
        # Create a unique identifier across therapists, patients and sessions
        item["utterance_uid"] = (
            f'{data["therapist_id"]}_{data["patient_id"]}_'
            f'S{int(data["session"]):02d}_U{int(item["utterance_id"]):04d}'
        )
        item["gold_label"] = None
        item["exclude_sensitive"] = None
        scored.append(item)

    # Select 3 CBT-enriched and 1 random utterance
    selected = select_candidates(
        scored,
        config["selection"],
        str(data["patient_id"]),
        int(data["session"]),
    )

    selection_by_id = { item["utterance_id"]: item["selection_source"] for item in selected}
    # Keep all utterances and tag the annotation subset
    for item in scored:
        source = selection_by_id.get(item["utterance_id"])
        item["selection_source"] = source
        item["selected_for_annotation"] = source in {
            "cbt_enriched",
            "random",
        }
    mode = config.get("output_mode", "all")
    if mode == "all":
        output_utterances = scored
    elif mode == "candidates":
        output_utterances = selected
        for item in output_utterances:
            item["selected_for_annotation"] = (
                item["selection_source"] in {"cbt_enriched", "random"}
            )
    else:
        raise ValueError(f"Invalid CBT screening output_mode: {mode}")

    result = {
        "therapist_id": data["therapist_id"],
        "patient_id": data["patient_id"],
        "session": data["session"],
        "language": data.get("language", "it"),
        "output_mode": mode,
        "n_available_utterances": len(utterances),
        "n_selected_for_annotation": sum(
            item.get("selected_for_annotation", False)
            for item in output_utterances),
        "utterances": output_utterances,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"CBT screening saved: {output_path}")

def run_pipeline(input_root: str, output_root: str, config_path: str) -> None:
    """Process all therapist session JSON files."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["cbt_screening"]
    classifier = NLIClassifier(config["model"], config["device"])
    source_root, target_root = Path(input_root), Path(output_root)

    for path in sorted(source_root.rglob("*.json")):
        process_json(path, target_root / path.relative_to(source_root), classifier, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and select CBT-related utterances.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/cbt_screening.yaml")
    args = parser.parse_args()
    if not Path(args.input_root).is_dir():
        raise SystemExit(f"\nERROR: Input directory not found:\n  {args.input_root}")
    if not Path(args.config).is_file():
        raise SystemExit(f"\nERROR: Configuration file not found:\n  {args.config}")
    run_pipeline(args.input_root, args.output_root, args.config)

if __name__ == "__main__":
    main()
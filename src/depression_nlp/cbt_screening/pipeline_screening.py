# Score CBT relevance and select utterances for human annotation
import argparse
import json
import random
from pathlib import Path
import yaml
from depression_nlp.cbt_screening.nli import NLIClassifier

def select_annotation_candidates(utterances: list[dict], domains: dict[str, list[str]], patient_id: str, session: int, random_seed: int = 42) -> list[dict]:
    """Select one utterance per domain and one random remaining utterance."""
    remaining = utterances.copy()
    for utterance in utterances:
        utterance["selected_for_annotation"] = False
        utterance["selection_source"] = None
        utterance["selection_score"] = None
    for domain_name, domain_labels in domains.items():
        if not remaining:
            break
        def domain_score(utterance: dict) -> float:
            return max(utterance["label_scores"].get(label, 0.0) for label in domain_labels)

        selected = max(remaining, key=domain_score)
        selected["selected_for_annotation"] = True
        selected["selection_source"] = domain_name
        selected["selection_score"] = round(domain_score(selected), 6)
        remaining.remove(selected)
    if remaining:
        rng = random.Random(f"{random_seed}:{patient_id}:{session}")
        selected = rng.choice(remaining)
        selected["selected_for_annotation"] = True
        selected["selection_source"] = "random"
        selected["selection_score"] = None
    return utterances

def process_json(path: Path, output_path: Path, classifier: NLIClassifier, config: dict) -> None:
    """Score utterances and select one per domain plus one random candidate."""
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
    for utterance, score in zip(utterances, scores):
        item = dict(utterance)
        item.update(score)
        item["utterance_uid"] = (
            f'{data["therapist_id"]}_{data["patient_id"]}_'
            f'S{int(data["session"]):02d}_U{int(item["utterance_id"]):04d}'
        )
        item["gold_label"] = None
        item["exclude_sensitive"] = None
        scored.append(item)

    # Select one utterance per domain and one random remaining utterance
    scored = select_annotation_candidates(
        utterances=scored,
        domains=config["selection"]["domains"],
        patient_id=str(data["patient_id"]),
        session=int(data["session"]),
        random_seed=config["selection"].get("random_seed", 42),
    )

    mode = config.get("output_mode", "all")
    if mode == "all":
        output_utterances = scored
    elif mode == "candidates":
        output_utterances = [
            item for item in scored
            if item.get("selected_for_annotation", False)
        ]
    else:
        raise ValueError(f"Invalid CBT screening output_mode: {mode}")

    result = {
        "therapist_id": data["therapist_id"],
        "patient_id": data["patient_id"],
        "session": data["session"],
        "language": data.get("language", "it"),
        "output_mode": mode,
        "n_available_utterances": len(utterances),
        "n_selected_for_annotation": sum(item.get("selected_for_annotation", False) for item in scored),
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
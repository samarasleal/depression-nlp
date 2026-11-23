# De-identify screened Italian therapist utterances
import argparse
import json
import re
from pathlib import Path
import spacy

REGEX_ENTITIES = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE),
    "URL": re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE),
    "PHONE": re.compile(r"(?<!\w)(?:\+39[\s.-]?)?(?:\d[\s.-]?){8,11}(?!\w)"),
    "FISCAL_CODE": re.compile(
        r"\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b",
        re.IGNORECASE,
    ),
    "DATE": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
        r"\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|"
        r"luglio|agosto|settembre|ottobre|novembre|dicembre)"
        r"(?:\s+\d{4})?)\b",
        re.IGNORECASE,
    ),
}
SPACY_ENTITIES = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
}

def detect_entities(text: str, nlp) -> list[tuple[int, int, str, int]]:
    """Detect direct identifiers using Italian NER and local regex rules."""
    spans = []
    for label, pattern in REGEX_ENTITIES.items():
        spans.extend((match.start(), match.end(), label, 2) for match in pattern.finditer(text))
    for entity in nlp(text).ents:
        label = SPACY_ENTITIES.get(entity.label_)
        if label:
            spans.append((entity.start_char, entity.end_char, label, 1))
    return remove_overlaps(spans)

def remove_overlaps(spans: list[tuple[int, int, str, int]]) -> list[tuple[int, int, str, int]]:
    """Prefer regex matches and longer spans when detections overlap."""
    ordered = sorted(spans, key=lambda item: (-item[3], -(item[1] - item[0])))
    accepted = []
    for span in ordered:
        start, end = span[:2]
        if not any(start < saved[1] and end > saved[0] for saved in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda item: item[0])

def deidentify_text(text: str, nlp) -> tuple[str, list[str]]:
    """Replace detected identifiers with generic placeholders."""
    spans = detect_entities(text, nlp)
    result = text
    for start, end, label, _ in reversed(spans):
        result = f"{result[:start]}[{label}]{result[end:]}"
    return result, sorted({span[2] for span in spans})

def process_file(source: Path, target: Path, nlp) -> None:
    """De-identify all utterances from one session JSON."""
    data = json.loads(source.read_text(encoding="utf-8"))
    for utterance in data.get("utterances", []):
        text, entity_types = deidentify_text(utterance["text"], nlp)
        utterance["text"] = text
        utterance["detected_entity_types"] = entity_types
        utterance["privacy_reviewed"] = False
    data["deidentified"] = True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def run_pipeline(input_root: str, output_root: str, spacy_model: str) -> None:
    """De-identify all screened utterances and preserve the JSON structure."""
    source_root, target_root = Path(input_root), Path(output_root)
    nlp = spacy.load(spacy_model)
    json_files = sorted(source_root.rglob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {source_root}")
    for source in json_files:
        target = target_root / source.relative_to(source_root)
        process_file(source, target, nlp)
    print(f"{len(json_files)} de-identified JSON files saved to {target_root}.")

def main() -> None:
    """Run local de-identification from the command line."""
    parser = argparse.ArgumentParser(description="De-identify screened utterances.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--spacy-model", default="it_core_news_sm")
    args = parser.parse_args()
    if not Path(args.input_root).is_dir():
        raise SystemExit(f"\nERROR: Input directory not found:\n  {args.input_root}")
    try:
        run_pipeline(args.input_root, args.output_root, args.spacy_model)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"\nERROR: De-identification failed:\n  {error}") from None

if __name__ == "__main__":
    main()
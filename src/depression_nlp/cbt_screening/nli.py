# Local zero-shot NLI classification for CBT utterances.

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIClassifier:
    """Score utterances against CBT hypotheses using entailment probability."""

    def __init__(self, model_name: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        labels = {str(label).lower(): int(index) for index, label in self.model.config.id2label.items()}
        self.entailment_id = next(
            (index for label, index in labels.items() if "entail" in label),
            None,
        )
        if self.entailment_id is None:
            raise ValueError(f"Entailment label not found: {self.model.config.id2label}")

    def score(self, texts: list[str], hypotheses: dict[str, str], batch_size: int = 16) -> list[dict]:
        """Return one entailment score per CBT label and utterance."""
        labels = list(hypotheses)
        pairs = [(text, hypotheses[label]) for text in texts for label in labels]
        probabilities = []

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities.extend(
                    torch.softmax(logits, dim=-1)[:, self.entailment_id].cpu().tolist()
                )

        results, index = [], 0
        for _ in texts:
            scores = {}
            for label in labels:
                scores[label] = round(probabilities[index], 6)
                index += 1
            predicted = max(scores, key=scores.get)
            results.append({
                "predicted_label": predicted,
                "cbt_score": scores[predicted],
                "label_scores": scores,
            })
        return results
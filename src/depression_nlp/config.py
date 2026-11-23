# Load and validate project configuration
from pathlib import Path
import yaml

def load_config(config_path: str | Path) -> dict:
    """Load the YAML pipeline configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The pipeline configuration must be a YAML mapping.")
    return config

def load_model_evaluation_settings(config_path: str | Path, require_threshold: bool = False) -> tuple[int, float | None]:
    """Load the shared top-k and intervention threshold."""
    config = load_config(config_path)
    settings = config.get("model_evaluation", {})
    top_k = int(settings.get("top_k", 3))
    raw_threshold = settings.get("intervention_threshold")
    threshold = None if raw_threshold is None else float(raw_threshold)
    if top_k != 3:
        raise ValueError("model_evaluation.top_k must be 3.")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("intervention_threshold must be between 0 and 1.")
    if require_threshold and threshold is None:
        raise ValueError(
            "intervention_threshold is null. Calculate and validate it "
            "before creating therapist profiles.")
    return top_k, threshold

def load_settings(config_path: str | Path) -> tuple[str, int, int, int]:
    """Load threshold-validation settings."""
    config = load_config(config_path)
    settings = config.get("model_evaluation", {}).get("threshold_validation", {})
    return (
        str(settings.get("calibration_source", "random")).lower(),
        int(settings.get("n_splits", 5)),
        int(settings.get("test_fold", 0)),
        int(settings.get("random_state", 42)),
    )
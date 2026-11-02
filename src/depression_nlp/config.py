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

import yaml
from pathlib import Path
from typing import Any

def load_config(config_name: str) -> dict[str, Any]:
    """Carrega arquivos YAML da pasta configs/."""
    # Ajuste o path se o script rodar de subpastas diferentes
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
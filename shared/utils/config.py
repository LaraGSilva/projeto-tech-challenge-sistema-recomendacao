import yaml
from pathlib import Path

def load_config(config_path="configs/model_params.yaml"):
    """Carrega o YAML de configurações de forma segura."""
    # O caminho é relativo à raiz do projeto quando executado via Docker ou CLI
    path = Path(config_path)
    if not path.exists():
        # Fallback caso o script seja executado de subdiretórios
        path = Path("../configs/model_params.yaml")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)
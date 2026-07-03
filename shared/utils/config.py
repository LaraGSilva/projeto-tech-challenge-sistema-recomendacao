import yaml
from pathlib import Path

def load_config(config_path="configs/model_params.yaml"):
    # Garante que funciona mesmo se chamado de pastas diferentes
    # Ajuste o caminho se necessário conforme o contexto de execução
    path = Path(config_path)
    if not path.exists():
        # Fallback para caso esteja rodando de dentro de 'api/' ou 'training/'
        path = Path("../configs/model_params.yaml")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)
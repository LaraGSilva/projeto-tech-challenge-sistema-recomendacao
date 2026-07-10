import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "configs/model_params.yaml") -> Dict[str, Any]:
    """Carrega o arquivo de configuração YAML de forma segura.

    Tenta localizar o arquivo de configuração no caminho fornecido. Caso não 
    seja encontrado, realiza uma tentativa de busca em um diretório superior 
    (fallback).

    Args:
        config_path (str): Caminho relativo ou absoluto para o arquivo .yaml. 
            O padrão é "configs/model_params.yaml".

    Returns:
        Dict[str, Any]: Dicionário contendo as configurações carregadas do YAML.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado em nenhum dos caminhos tentados.
        yaml.YAMLError: Se houver erro de sintaxe no arquivo YAML.
    """
    path = Path(config_path)
    
    if not path.exists():
        # Fallback caso o script seja executado a partir de subdiretórios
        path = Path("../configs/model_params.yaml")
        
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado em: {config_path} ou ../configs/model_params.yaml")
        
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
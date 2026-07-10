import importlib.util
import sys
from pathlib import Path
from typing import List


def validate() -> None:
    """Valida se o ambiente de desenvolvimento possui as dependências necessárias.

    Verifica a existência do arquivo de configuração de ambiente (.venv) e a
    instalação de todas as bibliotecas críticas definidas no projeto. Caso
    alguma dependência falhe, o script encerra a execução com código de erro 1.

    Returns:
        None
    """
    print("--- Validando ambiente do Tech Challenge ---")

    # 1. Verificar .venv
    if not Path(".venv").exists():
        print("Erro: Arquivo .venv não encontrado!")
        sys.exit(1)
    print("OK: .venv encontrado.")

    # 2. Verificar bibliotecas críticas
    required_libraries: List[str] = [
        "torch",
        "sklearn",
        "mlflow",
        "dvc",
        "pyspark",
        "pandas",
        "numpy",
    ]

    for lib in required_libraries:
        if importlib.util.find_spec(lib) is None:
            print(f"Erro: {lib} não está instalado!")
            sys.exit(1)
        print(f"OK: {lib} instalado.")

    print("--- Ambiente validado com sucesso! ---")


if __name__ == "__main__":
    validate()

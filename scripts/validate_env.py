import sys
import importlib.util
from pathlib import Path

def validate():
    print("--- Validando ambiente do Tech Challenge ---")
    
    # 1. Verificar .env
    if not Path(".env").exists():
        print("Erro: Arquivo .env não encontrado!")
        sys.exit(1)
    print("OK: .env encontrado.")

    # 2. Verificar bibliotecas críticas
    required = ["torch", "sklearn", "mlflow", "dvc", "pyspark", "pandas", "numpy"]
    for lib in required:
        if importlib.util.find_spec(lib) is None:
            print(f"Erro: {lib} não está instalado!")
            sys.exit(1)
        print(f"OK: {lib} instalado.")

    print("--- Ambiente validado com sucesso! ---")

if __name__ == "__main__":
    validate()
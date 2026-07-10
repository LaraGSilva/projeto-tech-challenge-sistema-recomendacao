import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.sparse import csr_matrix, save_npz

MIN_USER_INTERACTIONS = 5


def build_features(input_path: str, output_dir: str = "data/features") -> None:
    """Carrega eventos, processa pesos, filtra usuários e salva artefatos.

    Args:
        input_path (str): Caminho para o arquivo CSV contendo os eventos.
        output_dir (str): Diretório onde a matriz e os mapeamentos serão salvos.
    """
    df: pd.DataFrame = pd.read_csv(input_path)

    # Mapeamento de pesos
    df["peso"] = df["event"].map(
        {
            "view": 1,
            "addtocart": 3,
            "transaction": 5,
        }
    )

    # Agregação por interação
    interacoes: pd.DataFrame = (
        df.groupby(["visitorid", "itemid"])["peso"].sum().reset_index()
    )

    # Filtragem de usuários (mínimo de 5 interações)
    contagem: pd.Series = interacoes.groupby("visitorid")["itemid"].count()
    usuarios_ativos = contagem[contagem >= MIN_USER_INTERACTIONS].index
    interacoes = interacoes[interacoes["visitorid"].isin(usuarios_ativos)]

    # Mapeamento ID -> Índice
    user_to_idx: Dict[int, int] = {
        u: i for i, u in enumerate(interacoes["visitorid"].unique())
    }
    item_to_idx: Dict[int, int] = {
        it: i for i, it in enumerate(interacoes["itemid"].unique())
    }
    idx_to_user: Dict[int, int] = {i: u for u, i in user_to_idx.items()}
    idx_to_item: Dict[int, int] = {i: it for it, i in item_to_idx.items()}

    # Criação da matriz esparsa
    rows: pd.Series = interacoes["visitorid"].map(user_to_idx)
    cols: pd.Series = interacoes["itemid"].map(item_to_idx)
    data: pd.Series = interacoes["peso"]

    matrix: csr_matrix = csr_matrix((data, (rows, cols)))

    # Salvando artefatos
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_npz(output_path / "user_item_matrix.npz", matrix)

    with open(output_path / "mappings.pkl", "wb") as f:
        pickle.dump(
            {
                "idx_to_user": idx_to_user,
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "idx_to_item": idx_to_item,
            },
            f,
        )

    print(
        f"Matriz: {matrix.shape} | Usuários: {len(user_to_idx)} | Itens: {len(item_to_idx)}"
    )


if __name__ == "__main__":
    build_features("shared/data/data_csv/raw/events.csv")

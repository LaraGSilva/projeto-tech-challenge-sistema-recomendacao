import pandas as pd
import pytest

from shared.data.preprocessing import DefaultEventPreprocessor

VAR = 10
VAR_M = 5
VAR_HUN = 100
VAR_DUN = 200
MIN_INTERACOES = 2
EXPECTED_MAP_SIZE = 2


@pytest.fixture
def sample_df():
    """Cria um dataframe de teste com usuários ativos e inativos."""
    # Usuário 1: 5 interações (deve ser mantido)
    # Usuário 2: 2 interações (deve ser descartado)
    data = {
        "visitorid": [1, 1, 1, 1, 1, 2, 2],
        "itemid": [10, 11, 12, 13, 14, 20, 21],
        "event": ["view", "view", "view", "view", "transaction", "view", "view"],
    }
    return pd.DataFrame(data)


def test_filtering_logic(sample_df):
    """Verifica se apenas usuários com >= 5 interações são mantidos."""
    preprocessor = DefaultEventPreprocessor()
    processed = preprocessor.preprocess(sample_df.copy())

    # O Usuário 2 deveria ter sido removido (apenas 2 interações)
    assert MIN_INTERACOES not in processed["visitorid"].unique()
    assert 1 in processed["visitorid"].unique()
    assert len(processed) == VAR_M


def test_mappings_generation():
    """Testa se get_mappings retorna o dicionário correto."""
    preprocessor = DefaultEventPreprocessor()
    df = pd.DataFrame({"visitorid": [100, 200], "itemid": [1, 2]})
    u_map, i_map = preprocessor.get_mappings(df)

    assert VAR_HUN in u_map
    assert VAR_DUN in u_map
    assert len(u_map) == EXPECTED_MAP_SIZE
    assert u_map[VAR_HUN] == 0  # Primeiro ID mapeado para 0
    assert u_map[VAR_DUN] == 1


def test_index_types(sample_df):
    """Garante que os índices gerados são do tipo inteiro correto."""
    preprocessor = DefaultEventPreprocessor()
    processed = preprocessor.preprocess(sample_df.copy())

    assert processed["user_idx"].dtype == "int32"
    assert processed["item_idx"].dtype == "int32"

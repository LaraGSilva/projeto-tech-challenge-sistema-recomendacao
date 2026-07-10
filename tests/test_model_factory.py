from unittest.mock import MagicMock, patch

import pytest

from shared.ml.model_factory import ModelFactory


# Fixture para configurações simuladas
@pytest.fixture
def mock_config():
    return {"n_users": 10, "n_items": 5, "embedding_dim": 8}


# Teste: Garantir que o ValueError é levantado para modelos inexistentes
def test_create_model_invalid_name():
    with pytest.raises(ValueError, match="Modelo 'invalido' não encontrado"):
        ModelFactory.create_model("invalido", {})


# Teste: Verificar instanciação do NeuMF_Light (Caminho especial)
@patch("shared.ml.model_factory.NeuMF_Light")
@patch("shared.ml.model_factory.NeuMFConfig")
def test_create_neumf_light(mock_config_class, mock_model_class, mock_config):
    # Execução
    model = ModelFactory.create_model("neumf_light", mock_config)

    # Verificações
    mock_config_class.assert_called_once_with(**mock_config)
    mock_model_class.assert_called_once()
    assert model == mock_model_class.return_value


# Teste: Verificar instanciação de baselines (ex: PopularityModel)
@patch("shared.ml.model_factory.PopularityModel")
def test_create_popularity_model(mock_model_class, mock_config):
    model = ModelFactory.create_model("popularity", mock_config)

    mock_model_class.assert_called_once_with(**mock_config)
    assert model == mock_model_class.return_value


# Teste: Injeção de mappings
@patch("shared.ml.model_factory.PopularityModel")
def test_mappings_injection(mock_model_class, mock_config):
    mappings = {"user_to_idx": {0: 1}, "idx_to_item": {1: 0}}

    # Cria uma instância mock
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance

    # Execução
    model = ModelFactory.create_model("popularity", mock_config, mappings=mappings)

    # Verifica se os atributos foram injetados corretamente no mock
    assert model.user_to_idx == mappings["user_to_idx"]
    assert model.idx_to_item == mappings["idx_to_item"]

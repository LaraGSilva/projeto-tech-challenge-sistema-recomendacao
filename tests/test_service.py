from unittest.mock import patch

import numpy as np
import pytest
import torch

from api.service_wrapper import RecommendationService


@pytest.fixture
def mock_mappings(tmp_path):
    """Cria um arquivo de mapeamento fake para o teste."""
    mappings = {"user_to_idx": {101: 0, 102: 1}, "item_to_idx": {5001: 0, 5002: 1}}
    file_path = tmp_path / "mappings.npy"
    np.save(file_path, mappings)
    return str(file_path)


@pytest.fixture
def mock_model_file(tmp_path):
    """Cria um arquivo de pesos fake."""
    file_path = tmp_path / "model.pth"
    # Criar um dummy state_dict
    torch.save({}, file_path)
    return str(file_path)


@patch("mlflow.artifacts.download_artifacts")
def test_get_recommendations_call(mock_download, tmp_path):
    """Testa se a função de recomendação chama o método do modelo."""

    # Setup mínimo de arquivos
    mappings = {"user_to_idx": {1: 0}, "item_to_idx": {1: 0}}
    np.save(tmp_path / "mappings.npy", mappings)
    mock_download.return_value = str(tmp_path)

    # Mock do método recommend no modelo
    with patch("api.service_wrapper.NeuMF_Light") as mock_model:
        instance = mock_model.return_value
        instance.recommend.return_value = [5001, 5002]

        service = RecommendationService("test", "v1")
        recs = service.get_recommendations(user_id=1, k=2)

        assert recs == [5001, 5002]
        instance.recommend.assert_called_once_with(1, 2)


def test_init_raises_file_not_found():
    """Testa se erro ocorre se o mapeamento faltar."""
    with patch("mlflow.artifacts.download_artifacts", return_value="/tmp/fake"):
        with pytest.raises(FileNotFoundError):
            RecommendationService("test", "v1")

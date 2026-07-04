import pytest
from unittest.mock import MagicMock, patch
from api.service_wrapper import RecommendationService

@patch("api.service_wrapper.load_config")
@patch("api.service_wrapper.torch.load")
@patch("api.service_wrapper.np.load")
def test_recommendation_service(mock_np, mock_torch, mock_config):
    # Configuração dos mocks
    mock_config.return_value = {'model': {'mf_dim': 16}}
    mock_np.return_value = {
        'user_to_idx': {1: 0},
        'item_to_idx': {10: 0}
    }
    
    # Instancia o serviço
    service = RecommendationService("dummy.pth", "dummy.npy")
    
    # Mock da inferência
    service.model = MagicMock()
    service.model.return_value = MagicMock(numpy=lambda: [0.9])
    
    recs = service.get_recommendations(visitorid=1, k=1)
    
    assert len(recs) > 0
    assert recs[0] == 10
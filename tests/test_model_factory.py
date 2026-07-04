import pytest
from shared.ml.model_factory import ModelFactory
from shared.ml.models import NeuMF_Light

def test_model_factory_creation():
    # Testa se a factory cria o modelo com os params corretos
    model = ModelFactory.get_model("neumf_light", n_users=10, n_items=10, mf_dim=16)
    
    assert isinstance(model, NeuMF_Light)
    assert model.num_users == 10

def test_invalid_model():
    with pytest.raises(ValueError):
        ModelFactory.get_model("modelo_inexistente")
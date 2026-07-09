import pytest
from shared.ml.model_factory import ModelFactory
from shared.ml.models import NeuMF_Light
from shared.ml.baselines import PopularityModel

def test_model_factory_create_neumf_light():
    """Testa se a factory instancia o NeuMF_Light corretamente com a config."""
    config = {"n_users": 10, "n_items": 10, "mf_dim": 16}
    model = ModelFactory.create_model("neumf_light", config)
    
    assert isinstance(model, NeuMF_Light)
    # Verifica se os atributos foram setados (ajuste conforme seu __init__)
    assert model.user_embed.num_embeddings == 10
    assert model.item_embed.embedding_dim == 16

def test_model_factory_create_popularity_with_mappings():
    """Testa se a factory injeta mappings em baselines corretamente."""
    # Mock de matriz e mappings
    dummy_matrix = None # Em um teste real, use scipy.sparse.csr_matrix
    mappings = {
        "user_to_idx": {101: 0},
        "idx_to_item": {0: 500}
    }
    config = {"matrix": dummy_matrix, "mappings": mappings}
    
    model = ModelFactory.create_model("popularity", config, mappings=mappings)
    
    assert isinstance(model, PopularityModel)
    assert model.user_to_idx == mappings["user_to_idx"]
    assert model.idx_to_item == mappings["idx_to_item"]

def test_invalid_model_raises_value_error():
    """Testa se a factory levanta erro para modelos desconhecidos."""
    with pytest.raises(ValueError, match="Modelo modelo_inexistente não encontrado"):
        ModelFactory.create_model("modelo_inexistente", {})
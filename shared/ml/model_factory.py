from shared.ml.models import NeuMF_Light, NeuMF_RetailRocket_CPU

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            "neumf_light": NeuMF_Light,
            "neumf_retail": NeuMF_RetailRocket_CPU
        }
        if model_name not in models:
            raise ValueError(f"Modelo {model_name} não encontrado.")
        return models[model_name](**kwargs)
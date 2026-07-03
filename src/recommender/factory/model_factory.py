from src.recommender.models.architecture import NeuMF_RetailRocket_CPU, NeuMF_Light

class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "neumf_retail":
            return NeuMF_RetailRocket_CPU(**kwargs)
        elif model_type == "neumf_light":
            return NeuMF_Light(kwargs['n_users'], kwargs['n_items'])
        raise ValueError("Tipo de modelo desconhecido")
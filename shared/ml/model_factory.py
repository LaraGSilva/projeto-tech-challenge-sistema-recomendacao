from shared.ml.models import NeuMF_Light, NeuMF_RetailRocket_CPU

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            "neumf_light": NeuMF_Light,
            "neumf_retail": NeuMF_RetailRocket_CPU
        }
        if model_name == "neumf_light":
            valid_keys = ['n_users', 'n_items', 'mf_dim', 'mlp_dim']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            return models[model_name](**filtered_kwargs)
            
        return models[model_name](**kwargs)
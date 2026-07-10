from typing import Any, Dict, Optional, Union

from shared.ml.baselines import KNNModel, PopularityModel, SVDModel
from shared.ml.models import NeuMF_Light, NeuMFConfig


class ModelFactory:
    """Factory centralizada para instanciar modelos neurais e baselines."""

    @staticmethod
    def create_model(
        model_name: str,
        config: Dict[str, Any],
        mappings: Optional[Dict[str, Any]] = None,
    ) -> Union[NeuMFConfig, NeuMF_Light, PopularityModel, KNNModel, SVDModel]:
        models = {
            "popularity": PopularityModel,
            "knn": KNNModel,
            "svd": SVDModel,
            "NeuMFConfig": NeuMFConfig,
            "neumf_light": NeuMF_Light,
        }

        if model_name not in models:
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. "
                f"Opções: {list(models.keys())}"
            )

        # Instanciação específica do NeuMF
        if model_name == "neumf_light":
            neumf_config = NeuMFConfig(**config)

            model = NeuMF_Light(neumf_config)

        else:
            model = models[model_name](**config)

        # Injeta mappings para recomendação
        if mappings:
            model.user_to_idx = mappings.get("user_to_idx")
            model.idx_to_item = mappings.get("idx_to_item")

        return model

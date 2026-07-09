from typing import Any, Dict, Optional, Union

from shared.ml.baselines import KNNModel, PopularityModel, SVDModel
from shared.ml.models import NeuMF_Light, NeuMF_RetailRocket_CPU


class ModelFactory:
    """Factory centralizada para instanciar modelos neurais e baselines.

    Gerencia a criação de diferentes arquiteturas de modelos, filtrando parâmetros
    quando necessário e realizando a injeção de dependência dos mapeamentos (mappings)
    para habilitar a funcionalidade de recomendação.
    """

    @staticmethod
    def create_model(
        model_name: str,
        config: Dict[str, Any],
        mappings: Optional[Dict[str, Any]] = None,
    ) -> Union[
        NeuMF_Light, NeuMF_RetailRocket_CPU, PopularityModel, KNNModel, SVDModel
    ]:
        """Instancia um modelo baseado no nome e injeta dependências necessárias.

        Args:
            model_name (str): Nome do modelo a ser instanciado
                ('popularity', 'knn', 'svd', 'neumf_light', 'neumf_retail').
            config (Dict[str, Any]): Dicionário contendo os parâmetros de
                inicialização específicos do modelo.
            mappings (Optional[Dict[str, Any]]): Dicionário contendo os mapeamentos
                (user_to_idx, idx_to_item) para habilitar o método '.recommend()'.

        Returns:
            Union[NeuMF_Light, NeuMF_RetailRocket_CPU, PopularityModel, KNNModel, SVDModel]:
                O objeto do modelo instanciado e configurado.

        Raises:
            ValueError: Se o model_name fornecido não existir no registro de modelos.
        """
        models = {
            "popularity": PopularityModel,
            "knn": KNNModel,
            "svd": SVDModel,
            "neumf_light": NeuMF_Light,
            "neumf_retail": NeuMF_RetailRocket_CPU,
        }

        if model_name not in models:
            raise ValueError(
                f"Modelo '{model_name}' não encontrado. Opções: {list(models.keys())}"
            )

        # Instanciação específica para modelos com necessidade de filtro
        if model_name == "neumf_light":
            valid_keys = {"n_users", "n_items", "mf_dim"}
            filtered_kwargs = {k: v for k, v in config.items() if k in valid_keys}
            model = models[model_name](**filtered_kwargs)
        else:
            model = models[model_name](**config)

        # Injeção de dependência dos mapeamentos
        if mappings:
            model.user_to_idx = mappings.get("user_to_idx")
            model.idx_to_item = mappings.get("idx_to_item")

        return model

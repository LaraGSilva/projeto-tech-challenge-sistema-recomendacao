# from abc import ABC, abstractmethod
# from typing import Dict, Optional, Tuple

# import pandas as pd

# MIN_USER_INTERACTIONS = 5


# class BaseEventPreprocessor(ABC):
#     """Interface base para garantir que qualquer preprocessor tenha o método preprocess."""

#     @abstractmethod
#     def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Processa o DataFrame de eventos bruto.

#         Args:
#             df (pd.DataFrame): DataFrame contendo os eventos brutos.

#         Returns:
#             pd.DataFrame: DataFrame processado com pesos e índices.
#         """
#         pass


# class DefaultEventPreprocessor(BaseEventPreprocessor):
#     """Implementação padrão de pré-processamento para o RetailRocket.

#     Atributos:
#         event_weights (Dict[str, int]): Dicionário de pesos para cada tipo de evento.
#     """

#     def __init__(self, event_weights: Optional[Dict[str, int]] = None) -> None:
#         """Inicializa o preprocessor com pesos customizados ou padrão.

#         Args:
#             event_weights (Optional[Dict[str, int]]): Pesos para 'view', 'addtocart', 'transaction'.
#         """
#         self.event_weights: Dict[str, int] = event_weights or {
#             "view": 1,
#             "addtocart": 3,
#             "transaction": 5,
#         }

#     def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Executa a limpeza, atribuição de pesos e criação de índices.

#         Args:
#             df (pd.DataFrame): DataFrame contendo os eventos brutos.

#         Returns:
#             pd.DataFrame: DataFrame com interações ponderadas, filtradas e indexadas.
#         """
#         # 1. Mapeamento de pesos
#         processed_df: pd.DataFrame = df
#         processed_df["weight"] = processed_df["event"].map(self.event_weights).fillna(1)

#         # 2. Agregação por interação (usuário/item)
#         interacoes: pd.DataFrame = (
#             processed_df.groupby(["visitorid", "itemid"])["weight"].sum().reset_index()
#         )

#         # 3. Filtragem (usuários com pelo menos 5 interações)
#         contagem = interacoes.groupby("visitorid")["itemid"].count()
#         usuarios_ativos = contagem[contagem >= MIN_USER_INTERACTIONS].index
#         interacoes = interacoes[interacoes["visitorid"].isin(usuarios_ativos)].copy()

#         # 4. Criação de índices para tensores
#         interacoes["user_idx"] = interacoes["visitorid"].astype("category").cat.codes
#         interacoes["item_idx"] = interacoes["itemid"].astype("category").cat.codes

#         return interacoes

#     def get_mappings(self, df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
#         """Retorna os mapeamentos de ID original para índice.

#         Args:
#             df (pd.DataFrame): DataFrame contendo as colunas 'visitorid' e 'itemid'.

#         Returns:
#             Tuple[Dict[int, int], Dict[int, int]]: Tupla contendo (user_map, item_map).
#         """
#         user_map: Dict[int, int] = {
#             id_val: idx for idx, id_val in enumerate(df["visitorid"].unique())
#         }
#         item_map: Dict[int, int] = {
#             id_val: idx for idx, id_val in enumerate(df["itemid"].unique())
#         }
#         return user_map, item_map

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import pandas as pd

MIN_USER_INTERACTIONS = 5


class BaseEventPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DefaultEventPreprocessor(BaseEventPreprocessor):
    def __init__(self, event_weights: Optional[Dict[str, int]] = None) -> None:
        self.event_weights = event_weights or {
            "view": 1,
            "addtocart": 3,
            "transaction": 5,
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-processamento otimizado para reduzir uso de memória.
        """

        # Trabalha sobre o próprio dataframe
        df["weight"] = df["event"].map(self.event_weights).fillna(1).astype("int8")

        # Mantém apenas as colunas necessárias
        interacoes = (
            df[
                [
                    "visitorid",
                    "itemid",
                    "weight",
                ]
            ]
            .groupby(
                [
                    "visitorid",
                    "itemid",
                ],
                sort=False,
                observed=True,
                as_index=False,
            )
            .sum()
        )

        # Filtra usuários ativos (uma única passada)
        counts = interacoes.groupby("visitorid")["itemid"].transform("count")
        interacoes = interacoes[counts >= MIN_USER_INTERACTIONS]

        # Muito mais rápido que astype(category)
        interacoes["user_idx"], _ = pd.factorize(
            interacoes["visitorid"],
            sort=False,
        )

        interacoes["item_idx"], _ = pd.factorize(
            interacoes["itemid"],
            sort=False,
        )

        interacoes["user_idx"] = interacoes["user_idx"].astype("int32")
        interacoes["item_idx"] = interacoes["item_idx"].astype("int32")

        return interacoes

    def get_mappings(self, df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
        user_ids = df["visitorid"].unique()
        item_ids = df["itemid"].unique()

        user_map = {uid: idx for idx, uid in enumerate(user_ids)}

        item_map = {iid: idx for idx, iid in enumerate(item_ids)}

        return user_map, item_map

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

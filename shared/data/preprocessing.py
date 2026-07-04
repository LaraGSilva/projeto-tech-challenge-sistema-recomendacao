import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseEventPreprocessor(ABC):
    """Interface base para garantir que qualquer preprocessor tenha o método preprocess."""
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DefaultEventPreprocessor(BaseEventPreprocessor):
    """Implementação padrão de pré-processamento para o RetailRocket."""
    
    def __init__(self, event_weights=None):
        # Permite customizar os pesos via construtor, mas tem um padrão
        self.event_weights = event_weights or {
            "view": 1,
            "addtocart": 3,
            "transaction": 5,
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executa a limpeza, atribuição de pesos e criação de índices.
        """
        # 1. Mapeamento de pesos
        df = df.copy()
        df['weight'] = df['event'].map(self.event_weights).fillna(1)
        
        # 2. Agregação por interação (usuário/item)
        interacoes = (
            df.groupby(["visitorid", "itemid"])["weight"]
            .sum()
            .reset_index()
        )
        
        # 3. Filtragem (ex: usuários com pelo menos 5 interações)
        contagem = interacoes.groupby("visitorid")["itemid"].count()
        usuarios_ativos = contagem[contagem >= 5].index
        interacoes = interacoes[interacoes["visitorid"].isin(usuarios_ativos)]
        
        # 4. Criação de índices (mapeamento para tensores)
        # É importante que estes índices sejam gerados e salvos para uso na API
        interacoes['user_idx'] = interacoes['visitorid'].astype("category").cat.codes
        interacoes['item_idx'] = interacoes['itemid'].astype("category").cat.codes
        
        return interacoes

    def get_mappings(self, df: pd.DataFrame):
        """Retorna os mapeamentos caso precise salvar."""
        user_map = {id: idx for idx, id in enumerate(df['visitorid'].unique())}
        item_map = {id: idx for idx, id in enumerate(df['itemid'].unique())}
        return user_map, item_map
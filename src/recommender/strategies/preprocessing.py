from abc import ABC, abstractmethod
import pandas as pd

class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class DefaultEventPreprocessor(PreprocessingStrategy):
    """Estratégia padrão de tratamento de eventos do RetailRocket."""
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        peso_map = {'view': 1, 'addtocart': 3, 'transaction': 5}
        df['weight'] = df['event'].map(peso_map).fillna(0).astype('int8')
        return df
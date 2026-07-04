import pytest
import pandas as pd
from shared.data.preprocessing import DefaultEventPreprocessor

def test_preprocess_logic():
    data = {
        'visitorid': [1, 1, 2],
        'itemid': [10, 11, 10],
        'event': ['view', 'transaction', 'view']
    }
    df = pd.DataFrame(data)
    preprocessor = DefaultEventPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    # Validações
    assert 'weight' in df_processed.columns
    assert df_processed[df_processed['visitorid'] == 1]['weight'].sum() == 6 # 1 + 5
    # Como filtro é >= 5 interações, o dataframe resultante deve estar vazio
    assert len(df_processed) == 0
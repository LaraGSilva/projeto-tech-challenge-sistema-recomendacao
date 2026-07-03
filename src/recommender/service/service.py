import torch
import numpy as np
from src.recommender.models.architecture import NeuMF_Light
from src.recommender.factory.model_factory import ModelFactory
from src.recommender.strategies.preprocessing import PreprocessingStrategy

class RecommendationService:
    def __init__(self, model_type: str, preprocessor: PreprocessingStrategy, 
                 model_path: str, mappings_path: str, **model_params):
        
        self.preprocessor = preprocessor
        
        # 1. Carrega mapeamentos
        data = np.load(mappings_path, allow_pickle=True).item()
        self.user_to_idx = data['user_to_idx']
        self.item_to_idx = data['item_to_idx']
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        # 2. Cria o modelo via Factory
        # Passamos os parâmetros necessários para a arquitetura
        model_params.update({
            'n_users': len(self.user_to_idx),
            'n_items': len(self.item_to_idx)
        })
        self.model = ModelFactory.create_model(model_type, **model_params)
        
        # 3. Carrega os pesos salvos no treino
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def get_recommendations(self, visitorid: int, k: int) -> list:
        if visitorid not in self.user_to_idx:
            return []
        
        u_idx = self.user_to_idx[visitorid]
        item_indices = list(self.item_to_idx.values())
        
        with torch.no_grad():
            u_t = torch.full((len(item_indices),), u_idx, dtype=torch.long)
            i_t = torch.tensor(item_indices, dtype=torch.long)
            scores = self.model(u_t, i_t).numpy()
            
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.idx_to_item[idx] for idx in top_indices]
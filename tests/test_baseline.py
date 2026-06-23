import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.recommender.models.train_baselines import train_popularity, train_svd

@pytest.fixture
def dummy_sparse_matrix():
    # Matriz 4 usuários x 5 itens
    data = np.array([1, 3, 5, 2, 4])
    row_ind = np.array([0, 0, 1, 2, 3])
    col_ind = np.array([0, 2, 1, 4, 3])
    return csr_matrix((data, (row_ind, col_ind)), shape=(4, 5))

@pytest.fixture
def dummy_mappings():
    return {
        "user_to_idx": {"u1": 0, "u2": 1, "u3": 2, "u4": 3},
        "idx_to_user": {0: "u1", 1: "u2", 2: "u3", 3: "u4"},
        "item_to_idx": {"i1": 0, "i2": 1, "i3": 2, "i4": 3, "i5": 4},
        "idx_to_item": {0: "i1", 1: "i2", 2: "i3", 3: "i4", 4: "i5"}
    }

def test_train_popularity(dummy_sparse_matrix, dummy_mappings, mocker):
    # Bloqueia as chamadas reais ao MLflow durante o teste
    mocker.patch('mlflow.start_run')
    mocker.patch('mlflow.log_param')
    mocker.patch('mlflow.log_metrics')
    mocker.patch('mlflow.log_artifact')
    
    gt_dict = {"u1": ["i3"], "u2": ["i2"]}
    test_users_original = ["u1", "u2"]
    
    # Não deve levantar exceções
    train_popularity(
        matrix=dummy_sparse_matrix,
        mappings=dummy_mappings,
        test_users_original=test_users_original,
        gt_dict=gt_dict,
        n_items_total=5
    )

def test_train_svd(dummy_sparse_matrix, dummy_mappings, mocker):
    mocker.patch('mlflow.start_run')
    mocker.patch('mlflow.log_param')
    mocker.patch('mlflow.log_metrics')
    mocker.patch('mlflow.sklearn.log_model')
    
    gt_dict = {"u1": ["i3"], "u2": ["i2"]}
    test_users_original = ["u1", "u2"]
    
    # N_components deve ser menor que min(n_users, n_items)
    train_svd(
        matrix=dummy_sparse_matrix,
        mappings=dummy_mappings,
        test_users_original=test_users_original,
        gt_dict=gt_dict,
        n_items_total=5,
        n_components=2 
    )
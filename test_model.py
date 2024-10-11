
import numpy as np
import sklearn
import joblib

from typing import Union
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
import sklearn.ensemble
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from huggingface_hub import hf_hub_download
import pytest



@pytest.fixture
def test_dataset() -> Union[np.array, np.array]:

    X, y = load_wine(return_X_y=True)
    y = y == 2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42)
    
    return X_test, y_test


@pytest.fixture
def model() -> sklearn.ensemble.RandomForestClassifier:
    REPO_ID = 'camzis/rfc_model'
    FILENAME = 'rfc_wine.joblib'
    model = joblib.load(hf_hub_download(REPO_ID, FILENAME))
    return model


def test_model_inference_types(model, test_dataset):
  
  
    assert isinstance(model.predict(test_dataset[0]), np.ndarray)
    assert isinstance(test_dataset[0], np.ndarray)
    assert isinstance(test_dataset[1], np.ndarray)


def test_model_performance(model, test_dataset):

    metrics = classification_report(y_true=test_dataset[1], 
                                    y_pred=model.predict(test_dataset[0]),
                                    output_dict=True)  
    logger.info(f"Metrics {metrics}")  
    assert metrics['0']['f1-score'] > 0.58
    assert metrics['0']['precision'] > 0.9
    assert metrics['1']['f1-score'] > 0.05
    assert metrics['1']['precision'] > 0.05
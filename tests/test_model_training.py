import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model_training import split_data, train_model, evaluate_model, ModelTrainingError

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'f1': np.random.rand(100),
        'f2': np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_split_data(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80

def test_train_model(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    trained_model = train_model(model, X, y)
    
    assert hasattr(trained_model, "classes_")

def test_evaluate_model(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    
    metrics = evaluate_model(model, X, y)
    
    assert "Accuracy" in metrics
    assert "F1 Score" in metrics
    assert 0 <= metrics["Accuracy"] <= 1

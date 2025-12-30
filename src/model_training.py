import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Any, Dict, Optional, Tuple, Union

try:
    import mlflow
    import mlflow.sklearn
    mlflow_available = True
except ImportError:
    mlflow_available = False

class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass

def split_data(
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray], 
    test_size: float = 0.2, 
    random_state: int = 42, 
    stratify: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[Any, Any, Any, Any]:
    """
    Splits data into training and testing sets.
    
    Args:
        X (pd.DataFrame or np.array): Features.
        y (pd.Series or np.array): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        stratify (pd.Series or np.array): Data to use for stratification (usually y).
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    try:
        logging.info("Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        # Return Nones or raise. Since the original swallowed exceptions, we'll raise now for better control or return None as before but cleaner.
        # But per best practices, failing hard is often better than failing silently. 
        # However, to preserve compatibility with existing main pipeline which probably doesn't catch this class-specific error yet, 
        # I will log and re-raise a wrapper exception.
        raise ModelTrainingError(f"Failed to split data: {e}")

def train_model(
    model: Any, 
    X_train: Union[pd.DataFrame, np.ndarray], 
    y_train: Union[pd.Series, np.ndarray], 
    model_name: str = "model"
) -> Optional[Any]:
    """
    Trains a scikit-learn model.
    
    Args:
        model: The scikit-learn model instance.
        X_train: Training features.
        y_train: Training labels.
        model_name (str): Name of the model for logging.
        
    Returns:
        The trained model.
    """
    try:
        logging.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        logging.info(f"{model_name} training completed.")
        
        if mlflow_available:
            try:
                mlflow.sklearn.log_model(model, model_name)
            except Exception as ml_err:
                logging.warning(f"MLflow logging failed: {ml_err}")
            
        return model
    except Exception as e:
        logging.error(f"Error training {model_name}: {e}")
        raise ModelTrainingError(f"Failed to train {model_name}: {e}")

def evaluate_model(
    model: Any, 
    X_test: Union[pd.DataFrame, np.ndarray], 
    y_test: Union[pd.Series, np.ndarray], 
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Evaluates a trained model and returns metrics.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        model_name (str): Name of the model (for display).
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    try:
        logging.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        if y_prob is not None:
             metrics["ROC AUC"] = float(roc_auc_score(y_test, y_prob))
        
        logging.info(f"Metrics for {model_name}: {metrics}")
        
        # Log to MLflow if available
        if mlflow_available:
            try:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
            except Exception as ml_err:
                logging.warning(f"MLflow metric logging failed: {ml_err}")

        return metrics
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}")
        return {}

def print_evaluation_report(metrics: Dict[str, float], model_name: str = "model") -> None:
    print(f"\nModel Performance for: {model_name}")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)


"""
Comprehensive Model Training Pipeline for Fraud Detection
Implements Task 2a: Baseline Logistic Regression with class imbalance handling
Implements Task 2b: Ensemble models, hyperparameter tuning, cross-validation, and model selection
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data, basic_cleaning
from src.preprocessing import assign_countries, feature_engineering_dates, feature_engineering_velocity, get_preprocessor
from src.model_training import split_data, train_model, evaluate_model, print_evaluation_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import mlflow
    import mlflow.sklearn
    mlflow_available = True
except ImportError:
    mlflow_available = False
    logging.warning("MLflow not available. Skipping MLflow logging.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_fraud_data(fraud_df, ip_country_df, target_col='class'):
    """
    Prepares e-commerce fraud dataset with feature engineering.
    
    Returns:
        X, y, numerical_cols, categorical_cols
    """
    logging.info("Preparing E-Commerce Fraud Data...")
    
    # Basic cleaning
    fraud_df = basic_cleaning(fraud_df, date_columns=['signup_time', 'purchase_time'])
    
    # Assign countries
    fraud_df = assign_countries(fraud_df, ip_country_df)
    
    # Feature engineering
    fraud_df = feature_engineering_dates(fraud_df, ['signup_time', 'purchase_time'])
    fraud_df = feature_engineering_velocity(fraud_df)
    
    # Drop non-feature columns
    drop_cols = ['signup_time', 'purchase_time', 'user_id', 'ip_address', 'device_id']
    feature_df = fraud_df.drop(columns=[col for col in drop_cols if col in fraud_df.columns] + [target_col])
    
    # Separate numerical and categorical
    numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    X = feature_df
    y = fraud_df[target_col]
    
    logging.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logging.info(f"Numerical features: {len(numerical_cols)}, Categorical features: {len(categorical_cols)}")
    logging.info(f"Class distribution:\n{y.value_counts()}")
    
    return X, y, numerical_cols, categorical_cols

def prepare_creditcard_data(cc_df, target_col='Class'):
    """
    Prepares credit card fraud dataset.
    
    Returns:
        X, y, numerical_cols, categorical_cols
    """
    logging.info("Preparing Credit Card Fraud Data...")
    
    # Basic cleaning
    cc_df = basic_cleaning(cc_df)
    
    # All features are numerical (V1-V28, Amount, Time)
    X = cc_df.drop(columns=[target_col])
    y = cc_df[target_col]
    
    numerical_cols = X.columns.tolist()
    categorical_cols = []
    
    logging.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logging.info(f"Numerical features: {len(numerical_cols)}")
    logging.info(f"Class distribution:\n{y.value_counts()}")
    
    return X, y, numerical_cols, categorical_cols

def create_baseline_models():
    """
    Creates baseline models with class imbalance handling.
    Task 2a: LogisticRegression as primary interpretable baseline.
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            solver='liblinear'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def create_ensemble_models():
    """
    Creates additional ensemble models for comparison.
    Task 2b: Additional models for ensemble and comparison.
    """
    models = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    return models

def get_hyperparameter_grids():
    """
    Defines hyperparameter grids for tuning.
    Task 2b: Hyperparameter tuning configuration.
    """
    param_grids = {
        'LogisticRegression': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        },
        'RandomForest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },
        'GradientBoosting': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
    }
    return param_grids

def cross_validate_model(pipeline, X_train, y_train, cv=5, model_name='model'):
    """
    Performs cross-validation and reports mean and std for key metrics.
    Task 2b: Cross-validation with comprehensive metrics.
    """
    logging.info(f"Cross-validating {model_name} with {cv}-fold stratified CV...")
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'  # Use string for built-in scorer (compatible with all versions)
    }
    
    # Stratified K-Fold for imbalanced datasets
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_results = cross_validate(pipeline, X_train, y_train, 
                                cv=cv_splitter, 
                                scoring=scoring, 
                                return_train_score=True,
                                n_jobs=-1)
    
    # Calculate mean and std
    results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_key = f'test_{metric}'
        results[f'{metric}_mean'] = cv_results[test_key].mean()
        results[f'{metric}_std'] = cv_results[test_key].std()
    
    logging.info(f"CV Results for {model_name}:")
    for key, value in results.items():
        logging.info(f"  {key}: {value:.4f}")
    
    return results

def tune_hyperparameters(base_pipeline, param_grid, X_train, y_train, model_name='model', use_random=False):
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    Task 2b: Hyperparameter tuning.
    """
    logging.info(f"Tuning hyperparameters for {model_name}...")
    
    # Choose search method
    if use_random:
        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=param_grid,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        search = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
    
    # Fit
    search.fit(X_train, y_train)
    
    logging.info(f"Best parameters for {model_name}: {search.best_params_}")
    logging.info(f"Best CV F1 Score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def evaluate_and_compare_models(models_dict, X_test, y_test):
    """
    Evaluates multiple models and creates a comparison table.
    Task 2b: Model comparison.
    """
    logging.info("Evaluating and comparing models...")
    
    results = []
    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
        metrics['Model'] = model_name
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
    
    print("\n" + "="*80)
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    
    return comparison_df

def select_best_model(comparison_df, cv_results_dict):
    """
    Selects the best model based on performance and interpretability.
    Task 2b: Model selection with justification.
    """
    print("\n" + "="*80)
    print("MODEL SELECTION ANALYSIS")
    print("="*80)
    
    # Find best model by F1 Score
    best_f1_idx = comparison_df['F1 Score'].idxmax()
    best_f1_model = comparison_df.loc[best_f1_idx, 'Model']
    
    # Find best model by ROC AUC
    best_auc_idx = comparison_df['ROC AUC'].idxmax()
    best_auc_model = comparison_df.loc[best_auc_idx, 'Model']
    
    print(f"\nBest F1 Score: {best_f1_model} ({comparison_df.loc[best_f1_idx, 'F1 Score']:.4f})")
    print(f"Best ROC AUC: {best_auc_model} ({comparison_df.loc[best_auc_idx, 'ROC AUC']:.4f})")
    
    # Interpretability considerations
    print("\n--- INTERPRETABILITY ANALYSIS ---")
    print("LogisticRegression: HIGH - Coefficients directly interpretable, feature importance clear")
    print("RandomForest: MEDIUM - Feature importance available, but ensemble of trees less transparent")
    print("GradientBoosting: LOW - Complex boosting process, harder to explain individual predictions")
    
    # Performance vs Interpretability Trade-off
    print("\n--- PERFORMANCE vs INTERPRETABILITY TRADE-OFF ---")
    lr_row = comparison_df[comparison_df['Model'] == 'LogisticRegression']
    rf_row = comparison_df[comparison_df['Model'] == 'RandomForest']
    gb_row = comparison_df[comparison_df['Model'] == 'GradientBoosting']
    
    if not lr_row.empty:
        print(f"LogisticRegression - F1: {lr_row['F1 Score'].values[0]:.4f}, "
              f"Recall: {lr_row['Recall'].values[0]:.4f}, Interpretability: HIGH")
    if not rf_row.empty:
        print(f"RandomForest - F1: {rf_row['F1 Score'].values[0]:.4f}, "
              f"Recall: {rf_row['Recall'].values[0]:.4f}, Interpretability: MEDIUM")
    if not gb_row.empty:
        print(f"GradientBoosting - F1: {gb_row['F1 Score'].values[0]:.4f}, "
              f"Recall: {gb_row['Recall'].values[0]:.4f}, Interpretability: LOW")
    
    # Final Recommendation
    print("\n--- FINAL MODEL SELECTION ---")
    print("For Fraud Detection, we prioritize:")
    print("1. High Recall (minimize false negatives - catch fraudulent transactions)")
    print("2. Reasonable Precision (minimize false positives - avoid blocking legitimate transactions)")
    print("3. Interpretability (explain decisions for regulatory compliance and user trust)")
    
    # Select based on recall and interpretability
    comparison_df_sorted = comparison_df.copy()
    comparison_df_sorted['Interpretability_Score'] = comparison_df_sorted['Model'].map({
        'LogisticRegression': 3,
        'RandomForest': 2,
        'GradientBoosting': 1
    })
    
    # Weighted score: 40% Recall, 30% F1, 20% ROC AUC, 10% Interpretability
    comparison_df_sorted['Composite_Score'] = (
        0.4 * comparison_df_sorted['Recall'] +
        0.3 * comparison_df_sorted['F1 Score'] +
        0.2 * comparison_df_sorted['ROC AUC'] +
        0.1 * (comparison_df_sorted['Interpretability_Score'] / 3)
    )
    
    best_overall_idx = comparison_df_sorted['Composite_Score'].idxmax()
    best_overall_model = comparison_df_sorted.loc[best_overall_idx, 'Model']
    
    print(f"\nRecommended Model: {best_overall_model}")
    print(f"Composite Score: {comparison_df_sorted.loc[best_overall_idx, 'Composite_Score']:.4f}")
    print(f"Justification: Balances high recall for fraud detection with interpretability for compliance.")
    print("="*80 + "\n")
    
    return best_overall_model

def train_fraud_detection_pipeline(dataset_name, X, y, numerical_cols, categorical_cols):
    """
    Complete training pipeline for a fraud dataset.
    Implements Task 2a and Task 2b.
    """
    print("\n" + "#"*80)
    print(f"# TRAINING PIPELINE FOR: {dataset_name}")
    print("#"*80 + "\n")
    
    # Initialize MLflow experiment
    if mlflow_available:
        mlflow.set_experiment(f"Fraud_Detection_{dataset_name}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create preprocessor
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    
    # ==================== TASK 2a: BASELINE MODELS ====================
    print("\n" + "="*80)
    print("TASK 2a: BASELINE MODELS WITH CLASS IMBALANCE HANDLING")
    print("="*80 + "\n")
    
    baseline_models = create_baseline_models()
    trained_baselines = {}
    baseline_cv_results = {}
    
    for model_name, model in baseline_models.items():
        if mlflow_available:
            with mlflow.start_run(run_name=f"{dataset_name}_{model_name}_baseline"):
                # Create pipeline with SMOTE for class imbalance
                pipeline = ImbPipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])
                
                # Cross-validation
                cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5, model_name=model_name)
                baseline_cv_results[model_name] = cv_results
                
                # Train on full training set
                pipeline.fit(X_train, y_train)
                trained_baselines[model_name] = pipeline
                
                # Evaluate on test set
                test_metrics = evaluate_model(pipeline, X_test, y_test, model_name=model_name)
                print_evaluation_report(test_metrics, model_name=model_name)
                
                # Log to MLflow
                mlflow.log_params({f'model_type': model_name, 'dataset': dataset_name})
                for metric, value in cv_results.items():
                    mlflow.log_metric(f'cv_{metric}', value)
        else:
            # Without MLflow
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', model)
            ])
            
            cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5, model_name=model_name)
            baseline_cv_results[model_name] = cv_results
            
            pipeline.fit(X_train, y_train)
            trained_baselines[model_name] = pipeline
            
            test_metrics = evaluate_model(pipeline, X_test, y_test, model_name=model_name)
            print_evaluation_report(test_metrics, model_name=model_name)
    
    # ==================== TASK 2b: ENSEMBLE & HYPERPARAMETER TUNING ====================
    print("\n" + "="*80)
    print("TASK 2b: ENSEMBLE MODELS, HYPERPARAMETER TUNING, AND MODEL SELECTION")
    print("="*80 + "\n")
    
    # Add additional ensemble models
    ensemble_models = create_ensemble_models()
    param_grids = get_hyperparameter_grids()
    
    tuned_models = {}
    tuned_cv_results = {}
    
    # Tune baseline models
    for model_name in ['LogisticRegression', 'RandomForest']:
        if mlflow_available:
            with mlflow.start_run(run_name=f"{dataset_name}_{model_name}_tuned"):
                base_pipeline = ImbPipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42)),
                    ('model', baseline_models[model_name])
                ])
                
                # Hyperparameter tuning
                use_random = (model_name == 'RandomForest')  # Use RandomizedSearch for RF (large grid)
                best_model, best_params, best_score = tune_hyperparameters(
                    base_pipeline, param_grids[model_name], X_train, y_train, 
                    model_name=model_name, use_random=use_random
                )
                
                tuned_models[f'{model_name}_tuned'] = best_model
                
                # Cross-validation with best model
                cv_results = cross_validate_model(best_model, X_train, y_train, cv=5, model_name=f'{model_name}_tuned')
                tuned_cv_results[f'{model_name}_tuned'] = cv_results
                
                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric('best_cv_f1', best_score)
                for metric, value in cv_results.items():
                    mlflow.log_metric(f'cv_{metric}', value)
        else:
            base_pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', baseline_models[model_name])
            ])
            
            use_random = (model_name == 'RandomForest')
            best_model, best_params, best_score = tune_hyperparameters(
                base_pipeline, param_grids[model_name], X_train, y_train, 
                model_name=model_name, use_random=use_random
            )
            
            tuned_models[f'{model_name}_tuned'] = best_model
            cv_results = cross_validate_model(best_model, X_train, y_train, cv=5, model_name=f'{model_name}_tuned')
            tuned_cv_results[f'{model_name}_tuned'] = cv_results
    
    # Train additional ensemble model (GradientBoosting)
    for model_name, model in ensemble_models.items():
        if mlflow_available:
            with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
                pipeline = ImbPipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=42)),
                    ('model', model)
                ])
                
                # Cross-validation
                cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5, model_name=model_name)
                tuned_cv_results[model_name] = cv_results
                
                # Train
                pipeline.fit(X_train, y_train)
                tuned_models[model_name] = pipeline
                
                # Log to MLflow
                mlflow.log_params({f'model_type': model_name, 'dataset': dataset_name})
                for metric, value in cv_results.items():
                    mlflow.log_metric(f'cv_{metric}', value)
        else:
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('model', model)
            ])
            
            cv_results = cross_validate_model(pipeline, X_train, y_train, cv=5, model_name=model_name)
            tuned_cv_results[model_name] = cv_results
            
            pipeline.fit(X_train, y_train)
            tuned_models[model_name] = pipeline
    
    # ==================== MODEL EVALUATION AND COMPARISON ====================
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("="*80 + "\n")
    
    all_cv_results = {**baseline_cv_results, **tuned_cv_results}
    cv_summary = []
    for model_name, cv_res in all_cv_results.items():
        cv_summary.append({
            'Model': model_name,
            'Accuracy': f"{cv_res.get('accuracy_mean', 0):.4f} ± {cv_res.get('accuracy_std', 0):.4f}",
            'Precision': f"{cv_res.get('precision_mean', 0):.4f} ± {cv_res.get('precision_std', 0):.4f}",
            'Recall': f"{cv_res.get('recall_mean', 0):.4f} ± {cv_res.get('recall_std', 0):.4f}",
            'F1 Score': f"{cv_res.get('f1_mean', 0):.4f} ± {cv_res.get('f1_std', 0):.4f}",
            'ROC AUC': f"{cv_res.get('roc_auc_mean', 0):.4f} ± {cv_res.get('roc_auc_std', 0):.4f}"
        })
    
    cv_df = pd.DataFrame(cv_summary)
    print(cv_df.to_string(index=False))
    print("\n")
    
    # Evaluate all models on test set
    all_models = {**trained_baselines, **tuned_models}
    comparison_df = evaluate_and_compare_models(all_models, X_test, y_test)
    
    # Model selection
    best_model_name = select_best_model(comparison_df, all_cv_results)
    best_model = all_models[best_model_name]
    
    # Save best model
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    import joblib
    model_path = os.path.join(model_dir, f'{dataset_name}_best_model.pkl')
    joblib.dump(best_model, model_path)
    logging.info(f"Best model saved to: {model_path}")
    
    # Save comparison results
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    comparison_path = os.path.join(results_dir, f'{dataset_name}_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    cv_path = os.path.join(results_dir, f'{dataset_name}_cv_results.csv')
    cv_df.to_csv(cv_path, index=False)
    
    logging.info(f"Results saved to: {results_dir}")
    
    return best_model, comparison_df, cv_df

def main():
    """
    Main training pipeline for both fraud datasets.
    """
    logging.info("Starting Fraud Detection Model Training Pipeline...")
    
    # ==================== E-COMMERCE FRAUD DATA ====================
    try:
        fraud_df = load_data('data/raw/Fraud_Data.csv')
        ip_country_df = load_data('data/raw/IpAddress_to_Country.csv')
        
        if fraud_df is not None and ip_country_df is not None:
            X_fraud, y_fraud, num_cols_fraud, cat_cols_fraud = prepare_fraud_data(fraud_df, ip_country_df)
            
            best_fraud_model, fraud_comparison, fraud_cv = train_fraud_detection_pipeline(
                'ECommerce_Fraud', 
                X_fraud, 
                y_fraud, 
                num_cols_fraud, 
                cat_cols_fraud
            )
    except Exception as e:
        logging.error(f"Error processing E-Commerce Fraud data: {e}")
    
    # ==================== CREDIT CARD FRAUD DATA ====================
    try:
        cc_df = load_data('data/raw/creditcard.csv')
        
        if cc_df is not None:
            X_cc, y_cc, num_cols_cc, cat_cols_cc = prepare_creditcard_data(cc_df)
            
            best_cc_model, cc_comparison, cc_cv = train_fraud_detection_pipeline(
                'CreditCard_Fraud', 
                X_cc, 
                y_cc, 
                num_cols_cc, 
                cat_cols_cc
            )
    except Exception as e:
        logging.error(f"Error processing Credit Card Fraud data: {e}")
    
    logging.info("\n" + "="*80)
    logging.info("TRAINING PIPELINE COMPLETED!")
    logging.info("="*80)
    logging.info("Check 'models/' directory for saved models")
    logging.info("Check 'results/' directory for performance metrics")
    logging.info("Check MLflow UI for experiment tracking (if available)")

if __name__ == "__main__":
    main()

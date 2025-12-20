import sys
import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data, basic_cleaning
from src.preprocessing import assign_countries, feature_engineering_dates, feature_engineering_velocity, get_preprocessor, handle_imbalance_smote
from src.model_training import split_data, train_model, evaluate_model, print_evaluation_report

# Configure logging for main execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Fraud Detection Pipeline...")

    # 1. Load Data
    # Assuming script is run from project root
    fraud_data_path = os.path.join('data', 'raw', 'Fraud_Data.csv')
    ip_country_path = os.path.join('data', 'raw', 'IpAddress_to_Country.csv')
    
    fraud_df = load_data(fraud_data_path)
    ip_country_df = load_data(ip_country_path)
    
    if fraud_df is None or ip_country_df is None:
        logging.error("Failed to load datasets. Exiting.")
        return

    # 2. Data Cleaning
    fraud_df = basic_cleaning(fraud_df, date_columns=['signup_time', 'purchase_time'])

    # 3. Feature Engineering
    # IP Mapping
    fraud_df = assign_countries(fraud_df, ip_country_df)
    
    # Date Features
    fraud_df = feature_engineering_dates(fraud_df, ['signup_time', 'purchase_time'])
    
    # Velocity Features
    fraud_df = feature_engineering_velocity(fraud_df, user_col='user_id')
    
    # Drop original ID and Date columns for modeling
    drop_cols = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'source', 'browser']
    # Note: keeping country, age, sex, purchase_value, etc.
    # We need to decide what to keep. 'country', 'sex' are categorical. 'age', 'purchase_value', 'time_diff_minutes', 'user_tx_count' are numerical.
    
    # 4. Preparing X and y
    target_col = 'class'
    
    # Define features
    numerical_cols = ['purchase_value', 'age', 'time_diff_minutes', 'user_tx_count']
    categorical_cols = ['country', 'sex'] # source and browser could be added but simpler for now
    
    feature_cols = numerical_cols + categorical_cols
    
    # Ensure all columns exist
    missing_cols = [c for c in feature_cols if c not in fraud_df.columns]
    if missing_cols:
        logging.error(f"Missing columns: {missing_cols}")
        return

    X = fraud_df[feature_cols]
    y = fraud_df[target_col]
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, stratify=y)
    
    # 6. Preprocessing Pipeline
    # We need to fit the preprocessor on X_train first to transform before SMOTE (if we SMOTE numeric vars).
    # However, SMOTE supports categorical if encoded. 
    # Standard approach: Split -> Preprocess (Impute/Scale/Encode) -> SMOTE -> Train
    
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    
    # 7. Model Pipeline
    # For simplicity in this script, we'll use a Random Forest which handles some imbalance well, 
    # but let's build a proper pipeline.
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    # Note: SMOTE inside a pipeline requires imbalanced-learn pipeline, which is not imported here.
    # To keep "Code Best Practices" simple, we'll train the baseline Random Forest first without SMOTE in the pipeline,
    # or apply SMOTE manually on transformed data. For now, let's run a baseline.
    
    logging.info("Training Baseline Random Forest...")
    model = train_model(pipeline, X_train, y_train, model_name="RandomForest_Baseline")
    
    # 8. Evaluation
    metrics = evaluate_model(model, X_test, y_test, model_name="RandomForest_Baseline")
    print_evaluation_report(metrics, model_name="RandomForest_Baseline")
    
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data, basic_cleaning
from src.preprocessing import assign_countries, feature_engineering_dates, feature_engineering_velocity, get_preprocessor, handle_imbalance_smote
from src.model_training import split_data, train_model, evaluate_model, print_evaluation_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("=======================================================")
    print("   Starting Fraud Detection Pipelife with SMOTE")
    print("=======================================================")

    # 1. Load Data
    fraud_data_path = os.path.join('data', 'raw', 'Fraud_Data.csv')
    ip_country_path = os.path.join('data', 'raw', 'IpAddress_to_Country.csv')
    
    fraud_df = load_data(fraud_data_path)
    ip_country_df = load_data(ip_country_path)
    
    if fraud_df is None or ip_country_df is None:
        logging.error("Failed to load datasets. Exiting.")
        return

    # 2. Data Cleaning
    logging.info("Step 2: Cleaning Data...")
    fraud_df = basic_cleaning(fraud_df, date_columns=['signup_time', 'purchase_time'])

    # 3. Feature Engineering
    logging.info("Step 3: Feature Engineering...")
    
    # 3a. IP Mapping
    fraud_df = assign_countries(fraud_df, ip_country_df)
    print(f"   [Info] Country mapped. Sample countries: {fraud_df['country'].unique()[:5]}")
    
    # 3b. Date Features
    fraud_df = feature_engineering_dates(fraud_df, ['signup_time', 'purchase_time'])
    
    # 3c. Velocity Features
    fraud_df = feature_engineering_velocity(fraud_df, user_col='user_id')
    
    # 4. Preparing X and y
    target_col = 'class'
    numerical_cols = ['purchase_value', 'age', 'time_diff_minutes', 'user_tx_count']
    categorical_cols = ['country', 'sex', 'source', 'browser']
    
    feature_cols = numerical_cols + categorical_cols
    X = fraud_df[feature_cols]
    y = fraud_df[target_col]
    
    print(f"   [Info] Features selected: {feature_cols}")

    # 5. Split Data (Before SMOTE!)
    logging.info("Step 5: Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, stratify=y)
    
    print(f"   [Info] Training Set Distribution: {Counter(y_train)}")
    print(f"   [Info] Test Set Distribution: {Counter(y_test)}")

    # 6. Preprocessing (Transformations)
    logging.info("Step 6: Applying Transformations...")
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    
    # Fit on Train, Transform Train and Test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding (optional, for debugging)
    # feature_names = preprocessor.get_feature_names_out()
    
    # 7. Handle Imbalance (SMOTE on Train only)
    logging.info("Step 7: Handling Imbalance with SMOTE...")
    print(f"   [Info] Before SMOTE: {Counter(y_train)}")
    
    X_train_resampled, y_train_resampled = handle_imbalance_smote(X_train_processed, y_train)
    
    print(f"   [Info] After SMOTE: {Counter(y_train_resampled)}")

    # 8. Model Training
    logging.info("Step 8: Training Model (Random Forest)...")
    # Note: We pass the PROCESSED and RESAMPLED data
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model = train_model(clf, X_train_resampled, y_train_resampled, model_name="RandomForest_SMOTE")
    
    # 9. Evaluation
    logging.info("Step 9: Evaluating Model...")
    # Note: We evaluate on the PROCESSED ORIGINAL test set (never resampled)
    metrics = evaluate_model(model, X_test_processed, y_test, model_name="RandomForest_SMOTE")
    print_evaluation_report(metrics, model_name="RandomForest_SMOTE")
    
    print("=======================================================")
    print("   Pipeline Completed Successfully")
    print("=======================================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

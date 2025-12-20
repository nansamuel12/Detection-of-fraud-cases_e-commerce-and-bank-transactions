# Fraud Detection Project

## Project Overview
**Role:** Data Scientist at Adey Innovations Inc.
**Goal:** Build machine learning models to detect fraudulent transactions in both e-commerce and bank credit card data.

This project aims to create accurate and robust fraud detection models capable of handling unique challenges such as class imbalance and the trade-off between security (false positives) and user experience (false negatives).

## Business Need
Adey Innovations Inc. requires solutions to improve fraud detection for e-commerce and banking. Effective fraud detection minimizes financial losses and builds trust with customers. The system must also support real-time monitoring and reporting.

## Key Challenges
- **Class Imbalance:** Fraudulent transactions are rare compared to legitimate ones.
- **Precision-Recall Trade-off:** Balancing the prevention of fraud with the avoidance of blocking legitimate users.
- **Explainability:** Models must be interpretable to understand why a transaction was flagged.

## Project Scope
1.  **Data Analysis & Preprocessing:** Cleaning data, handling missing values, and geolocation analysis.
2.  **Feature Engineering:** Creating features to capture fraud patterns.
3.  **Model Building:** Training ML models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
4.  **Evaluation:** Using metrics suitable for imbalanced data (Precision, Recall, F1-Score, ROC-AUC).
5.  **Explainability:** Using SHAP or LIME to interpret model decisions.

## Data Sources
1.  **Fraud_Data.csv:** E-commerce transaction data (timestamps, device ID, IP address, etc.).
2.  **IpAddress_to_Country.csv:** Mapping of IP ranges to countries.
3.  **creditcard.csv:** Anonymized bank credit card transactions (PCA features V1-V28, Time, Amount).

## Project Structure
- `data/raw/`: Contains the raw CSV datasets.
- `notebooks/`: Jupyter notebooks for EDA and modeling.
    - `eda-fraud-data.ipynb`: Analysis of E-commerce data.
    - `eda-creditcard.ipynb`: Analysis of Credit Card data.
- `requirements.txt`: Python dependencies.

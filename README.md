# Fraud Detection Project

## Project Overview
**Role:** Data Scientist at Adey Innovations Inc.
**Goal:** Build machine learning models to detect fraudulent transactions in both e-commerce and bank credit card data.

This project aims to create accurate and robust fraud detection models capable of handling unique challenges such as class imbalance and the trade-off between security (false positives) and user experience (false negatives).

## Business Need
Adey Innovations Inc. requires solutions to improve fraud detection for e-commerce and banking. Effective fraud detection minimizes financial losses and builds trust with customers. The system must also support real-time monitoring and reporting.

## Key Challenges
- **Class Imbalance:** Fraudulent transactions are rare compared to legitimate ones. This is handled using SMOTE (Synthetic Minority Over-sampling Technique) in the training pipeline.
- **Precision-Recall Trade-off:** Balancing the prevention of fraud with the avoidance of blocking legitimate users.
- **Explainability:** Models must be interpretable to understand why a transaction was flagged. We use SHAP and LIME to interpret predictions.

## Project Structure

```bash
detection_fraud_cases_bank_transaction/
│
├── .github/                # GitHub workflows and templates
│   ├── workflows/          # CI/CD pipelines
│   └── pull_request_template.md
│
├── data/
│   └── raw/                # Contains the raw CSV datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│
├── docs/                   # Documentation
│   └── BRANCHING_STRATEGY.md
│
├── notebooks/              # Jupyter notebooks for interactive analysis
│   ├── eda-fraud-data.ipynb
│   └── eda-creditcard.ipynb
│
├── src/                    # Reusable source code modules
│   ├── data_loader.py      # Data loading and basic cleaning (typed & exception handled)
│   ├── preprocessing.py    # Feature engineering (IP mapping, Date extraction, SMOTE)
│   ├── model_training.py   # Model training and evaluation logic
│   ├── explainability.py   # SHAP and LIME analysis module
│   └── visualization.py    # Plotting functions
│
├── tests/                  # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_model_training.py
│
├── photo/                  # Generated EDA visualizations (saved as .png)
│
├── train_models.py         # End-to-end training pipeline script
├── generate_plots.py       # Script to generate all EDA plots
├── requirements.txt        # Python dependencies
└── README.md
```

## Features Implemented

### 1. Data Analysis & Preprocessing
- **Data Cleaning:** Handling duplicates and converting data types with robust error handling.
- **Geolocation Analysis:** Mapping IP addresses to countries to identify high-risk regions.
- **Exploratory Data Analysis (EDA):** Comprehensive analysis of class distributions and feature relationships associated with fraud.

### 2. Feature Engineering
- **Temporal Features:** Extracted hour, day, and month from timestamps. Calculated `time_diff` between sign-up and purchase.
- **Velocity Features:** Calculated user transaction frequency (`user_tx_count`).
- **IP Mapping:** Efficiently mapped integer IP addresses to countries using `merge_asof`.

### 3. Model Building & Evaluation
- **Pipeline:** Modular pipeline handling data loading, processing, SMOTE, and training.
- **Models:**
    - Logistic Regression (Baseline with interpretable coefficients)
    - Random Forest (Robust ensemble method)
    - Gradient Boosting (High-performance ensemble)
- **Evaluation:** Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **Model Selection:** Automated selection based on a composite score of Recall, F1, ROC-AUC, and Interpretability.

### 4. Model Explainability 🧠
- **SHAP (SHapley Additive exPlanations):** Global and local feature contribution analysis.
- **LIME (Local Interpretable Model-agnostic Explanations):** Explaining individual predictions for transparency.
- **Feature Importance:** Visualizing top contributing factors for fraud detection.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests:**
    Ensure environment integrity by running the test suite:
    ```bash
    pytest
    ```

3.  **Generate EDA Plots:**
    Run the plotting script to generate visualizations in the `photo/` folder:
    ```bash
    python generate_plots.py
    ```

4.  **Run the Training Pipeline:**
    Execute the main pipeline to load data, process features, apply SMOTE, train models, tune hyperparameters, and evaluate results:
    ```bash
    python train_models.py
    ```

## CI/CD & Best Practices
- **Branching Strategy:** We follow a feature-branch workflow (see `docs/BRANCHING_STRATEGY.md`).
- **CI/CD:** GitHub Actions automatically run tests and linting on every push and PR.
- **Code Quality:** All code is type-hinted and follows PEP 8 standards.

## Data Sources
1.  **Fraud_Data.csv:** E-commerce transaction data (timestamps, device ID, IP address, etc.).
2.  **IpAddress_to_Country.csv:** Mapping of IP ranges to countries.
3.  **creditcard.csv:** Anonymized bank credit card transactions (PCA features V1-V28, Time, Amount).

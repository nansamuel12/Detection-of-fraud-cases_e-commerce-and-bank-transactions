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
- **Explainability:** Models must be interpretable to understand why a transaction was flagged.

## Project Structure

```bash
detection_fraud_cases_bank_transaction/
│
├── data/
│   └── raw/                # Contains the raw CSV datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│
├── notebooks/              # Jupyter notebooks for interactive analysis
│   ├── eda-fraud-data.ipynb
│   └── eda-creditcard.ipynb
│
├── src/                    # Reusable source code modules
│   ├── data_loader.py      # Data loading and basic cleaning
│   ├── preprocessing.py    # Feature engineering (IP mapping, Date extraction, SMOTE)
│   ├── visualization.py    # Plotting functions
│   └── model_training.py   # Model training and evaluation logic
│
├── photo/                  # Generated EDA visualizations (saved as .png)
│
├── main.py                 # End-to-end pipeline script
├── generate_plots.py       # Script to generate all EDA plots
├── requirements.txt        # Python dependencies
└── README.md
```

## Features Implemented

### 1. Data Analysis & Preprocessing
- **Data Cleaning:** Handling duplicates and converting data types.
- **Geolocation Analysis:** Mapping IP addresses to countries to identify high-risk regions.
- **Exploratory Data Analysis (EDA):** Comprehensive analysis of class distributions and feature relationships associated with fraud. Visualizations are saved in the `photo/` folder.

### 2. Feature Engineering
- **Temporal Features:** Extracted hour, day, and month from timestamps. Calculated `time_diff` between sign-up and purchase.
- **Velocity Features:** Calculated user transaction frequency (`user_tx_count`).
- **IP Mapping:** Efficiently mapped integer IP addresses to countries using `merge_asof`.

### 3. Model Building & Evaluation
- **Pipeline:** Implemented a modular pipeline in `main.py` that handles data loading, cleaning, feature engineering, splitting, preprocessing, and training.
- **Class Imbalance:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data.
- **Model:** Trained a Random Forest Classifier.
- **Evaluation:** Metrics used include Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate EDA Plots:**
    Run the plotting script to generate visualizations in the `photo/` folder:
    ```bash
    python generate_plots.py
    ```

3.  **Run the Training Pipeline:**
    Execute the main pipeline to load data, process features, apply SMOTE, train the model, and evaluate results:
    ```bash
    python main.py
    ```

## Data Sources
1.  **Fraud_Data.csv:** E-commerce transaction data (timestamps, device ID, IP address, etc.).
2.  **IpAddress_to_Country.csv:** Mapping of IP ranges to countries.
3.  **creditcard.csv:** Anonymized bank credit card transactions (PCA features V1-V28, Time, Amount).

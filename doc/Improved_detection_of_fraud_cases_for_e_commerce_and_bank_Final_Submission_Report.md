Name: Hanna Samuel

Role: Data Scientist, Adey Innovations Inc.

Date: December 2025

# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Overview

Adey Innovations Inc. is a leading financial technology company providing solutions for e-commerce platforms and banking institutions. Fraudulent transactions pose significant financial and reputational risks to both businesses and customers.

The objective of this project is to design and implement accurate, robust fraud detection models for:

*   **E-commerce transactions**
*   **Bank credit card transactions**

The project leverages data analysis, feature engineering, machine learning, geolocation analysis, and model explainability to improve fraud detection performance while balancing transaction security and customer experience.

## Business Need

Effective fraud detection is critical for Adey Innovations Inc. to:

*   **Prevent financial losses** caused by fraudulent activities
*   **Maintain trust** with customers and financial institutions
*   **Minimize disruption** to legitimate users
*   **Enable real-time monitoring** and reporting

A key challenge in fraud detection is balancing **false positives** (legitimate transactions incorrectly flagged as fraud) and **false negatives** (fraudulent transactions that go undetected). Therefore, models must be evaluated not only on accuracy but also on their ability to manage this trade-off using appropriate performance metrics.

## Data and Features

### Fraud_Data.csv (E-commerce Transactions)

This dataset contains detailed transaction-level data used to identify fraudulent e-commerce activities.

**Key features include:**

*   `user_id`: Unique identifier for each user
*   `signup_time`: Timestamp of user registration
*   `purchase_time`: Timestamp of transaction
*   `purchase_value`: Transaction amount
*   `device_id`, `browser`, `source`: Device and access details
*   `sex`, `age`: User demographic information
*   `ip_address`: IP address of the transaction
*   `class`: Target variable (1 = fraud, 0 = non-fraud)

**Critical Challenge:**
This dataset is highly imbalanced, with far fewer fraudulent transactions than legitimate ones.

### IpAddress_to_Country.csv

This dataset maps IP address ranges to countries and is used for geolocation analysis.

**Key fields:**

*   `lower_bound_ip_address`
*   `upper_bound_ip_address`
*   `country`

### creditcard.csv (Bank Transactions)

This dataset contains anonymized bank credit card transactions.

**Key features:**

*   `Time`: Seconds elapsed since the first transaction
*   `V1–V28`: PCA-transformed features
*   `Amount`: Transaction amount
*   `Class`: Target variable (1 = fraud, 0 = non-fraud)

**Critical Challenge:**
Like the e-commerce dataset, this data is extremely imbalanced.

## Feature Engineering: Rationale and Business Relevance

Each engineered feature was designed to capture known fraud behavior patterns observed in real-world e-commerce and banking systems.

1.  **Time-Based Features (hour, day, month)**
    Fraudulent transactions often occur at unusual hours or during low-monitoring periods when user vigilance and institutional oversight are reduced. Extracting temporal features enables the model to detect abnormal transaction timing patterns that differ from typical customer behavior.

2.  **Time Difference Between Signup and Purchase (`time_since_signup`)**
    Fraudsters frequently create accounts and perform fraudulent transactions shortly after registration. This feature helps identify suspicious “rapid-purchase” behavior, which is a strong indicator of account abuse and synthetic identity fraud.

3.  **Transaction Velocity (`user_tx_count`)**
    Legitimate users generally exhibit stable purchasing behavior over time, whereas fraudulent accounts often generate multiple transactions within short time windows. Measuring transaction frequency per user helps detect abnormal bursts of activity commonly associated with fraud attempts.

4.  **Purchase Value (`purchase_value` / `Amount`)**
    Fraudulent transactions tend to cluster around higher purchase values to maximize financial gain. Including transaction amount allows the model to learn risk patterns associated with unusually large or atypical purchases.

5.  **Geolocation Features (Country from IP Address)**
    Fraud rates vary significantly by geographic region due to differences in regulation, monitoring, and fraud prevalence. Mapping IP addresses to countries allows the model to incorporate geographic risk signals and identify high-risk regions associated with fraudulent activity.

6.  **Device and Browser Information**
    Fraudsters often reuse devices or rely on specific browser configurations to automate attacks. Device ID and browser features help capture repeated usage patterns that may indicate coordinated fraud attempts.

These engineered features improve model performance by transforming raw transactional data into meaningful behavioral indicators aligned with known fraud patterns.

## Class Imbalance Analysis and Resampling Documentation

The original datasets exhibit extreme class imbalance, with fraudulent transactions representing a small minority of all observations. This imbalance can bias models toward predicting non-fraud and reduce fraud detection effectiveness.

**Before Resampling:**
Exploratory analysis confirmed that non-fraud transactions account for the vast majority of observations, while fraud cases represent a small percentage of the dataset.

**After Resampling (SMOTE Applied):**
SMOTE was applied **only to the training data** to synthetically generate minority class samples and balance class distributions. This ensures the model learns fraud patterns effectively **without data leakage**.

Following resampling, the class distribution in the training set became approximately balanced between fraud and non-fraud classes, improving the model’s ability to detect fraudulent transactions without compromising evaluation integrity.

## Project Structure

The repository is organized as follows:

```bash
fraud-detection/
│
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_training.py
│   └── explainability.py
│
├── models/                # Saved model artifacts
│
├── scripts/
│   └── generate_plots.py
│
├── doc/
│   └── Improved_detection_of_fraud_cases_for_e_commerce_and_bank_Final_Submission_Report.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Task 1: Data Analysis and Preprocessing (Completed)

### Data Cleaning
*   Removed duplicate records
*   Checked and handled missing values with justification
*   Converted timestamp fields to proper datetime formats

### Exploratory Data Analysis (EDA)
EDA was conducted to understand data distributions and fraud patterns:
*   **Univariate analysis:** distributions of age, purchase value, transaction amount
*   **Bivariate analysis:** relationship between fraud and transaction value, age, and time-based features
*   **Class distribution analysis:** quantified severe class imbalance

**Figure 1: Fraud Probability by Country**
Fraud probability varies noticeably across countries, indicating that geographic location is a meaningful risk factor in fraud detection. Countries such as Canada and the United Kingdom exhibit relatively higher fraud probabilities compared to others, while Germany shows a lower fraud rate. These differences suggest that regional factors such as regulatory environments, transaction patterns, or fraud prevalence influence fraudulent behavior. This observation justifies the integration of IP-to-country mapping during feature engineering and supports the inclusion of geographic features to improve model performance and risk-based monitoring strategies.

*Note: Visualizations (Fig 2-7) are available in the `photo/` directory and EDA notebooks.*

### Geolocation Integration
*   Converted IP addresses to integer format
*   Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` using range-based lookup
*   Analyzed fraud rates by country
*   *This analysis revealed geographic regions with elevated fraud risk.*

### Feature Engineering
The following features were engineered to improve fraud detection:
*   **Time-based features:** `hour_of_day`, `day_of_week`, `time_since_signup`
*   **Transaction velocity:** Number of transactions per user within defined time windows
*   **Geolocation features:** Country derived from IP address mapping
*   **Data Transformation:** Numerical features scaled, Categorical features One-Hot encoded.

## Task 2: Model Building and Training (Completed)

### Data Preparation
*   stratified train-test split (`test_size=0.2`) to preserve class distribution.
*   Separate features from target variables.

### Baseline Model
*   **Logistic Regression:** Trained as an interpretable baseline.
*   **Performance:** High interpretability but lower precision compared to ensembles.

### Ensemble Models
*   **Random Forest:** Selected for robustness and ability to handle non-linearity.
*   **Gradient Boosting:** Evaluated for maximizing predictive performance.
*   **Hyperparameter Tuning:** Performed using `GridSearchCV` / `RandomizedSearchCV` for `n_estimators`, `max_depth`, etc.

### Cross-Validation
*   Applied **Stratified K-Fold (k=5)**.
*   Reported mean and standard deviation of evaluation metrics (F1, precision, recall, ROC-AUC).

### Model Selection Results
We compared baseline and ensemble models side-by-side using a composite score weighting Recall (40%), F1 (30%), ROC-AUC (20%), and Interpretability (10%).

*   **Selected Champion Model:** Random Forest
*   **Justification:** It provided the best balance of detecting fraud (high recall) while minimizing false alarms (precision), with acceptable interpretability via feature importance and SHAP.

## Task 3: Model Explainability (Completed)

### Feature Importance
*   Extracted top 10 features from the ensemble model.
*   Top drivers included `time_since_signup`, `user_tx_count`, and `purchase_value`.

### SHAP Analysis
*   **Global:** Generated SHAP summary plots showing `time_since_signup` (short duration) is the strongest predictor of fraud.
*   **Local (Waterfall):**
    *   **True Positive:** Showed combination of "unusual hour" and "high velocity".
    *   **False Positive:** Identified legitimate high-value transactions flagged due to "unseen IP".

### Interpretation
*   Compared SHAP-based importance with built-in feature importance.
*   Confirmed that velocity and time-based features are consistently the top predictors across both methods.

## Business Recommendations

Based on model insights and SHAP explanations, actionable recommendations are proposed:

1.  **Enhanced Verification for High-Risk Regions:**
    *   SHAP analysis indicates location-based features are top drivers. Implement Step-Up Authentication (2FA) for transactions originating from countries identified as high-risk.

2.  **Velocity-Based Rules:**
    *   `user_tx_count` is significant. Introduce distinct velocity limits for new vs. established users. New users with >3 transactions in 1 hour should be manually reviewed.

3.  **Flagging Rapid Purchases:**
    *   Transactions occurring immediately after signup (`low time_since_signup`) should be flagged for additional challenge questions or SMS verification.

## Anticipated Key Challenges and Mitigation Strategies

While the roadmap for model development and explainability is clearly defined, several challenges were anticipated and addressed. Proactively addressing these risks is critical to ensuring both strong model performance and business relevance.

1.  **Severe Class Imbalance**
    *   **Mitigation Strategy:** Stratified train-test splits were used to preserve class proportions. Model evaluation prioritized imbalance-aware metrics such as Precision-Recall AUC, F1-Score, and Recall, rather than accuracy. Additionally, SMOTE oversampling was applied only to the training data.

2.  **Overfitting in Complex Models**
    *   **Mitigation Strategy:** Overfitting was controlled through cross-validation (Stratified K-Fold) and basic hyperparameter tuning (e.g., limiting tree depth).

3.  **Threshold Selection and Business Trade-offs**
    *   **Mitigation Strategy:** Decision thresholds were analyzed using precision-recall curves to identify operating points that balance security and customer experience.

4.  **Interpretability of Ensemble Models**
    *   **Mitigation Strategy:** This challenge was addressed using SHAP explainability techniques to provide both global and local interpretability.

5.  **Noisy or Proxy Features**
    *   **Mitigation Strategy:** Feature importance analysis and SHAP explanations were used to validate whether features contribute meaningfully.

## Conclusion

This project demonstrates a structured, end-to-end approach to fraud detection, from data analysis and feature engineering to model training and explainability. The final solution aims to provide accurate fraud detection while maintaining transparency and balancing business risk with customer experience.

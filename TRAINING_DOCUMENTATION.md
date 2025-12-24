# Task 2a & 2b: Model Training Documentation

## Overview

This document describes the implementation of **Task 2a** (Data Preparation and Baseline Model) and **Task 2b** (Ensemble Model, Cross-Validation, and Model Selection) for the fraud detection project.

## Task 2a: Data Preparation and Baseline Model

### Implementation Details

#### 1. **Logistic Regression as Primary Baseline**
- Implemented `LogisticRegression` with `class_weight='balanced'` to handle class imbalance
- This is the primary interpretable baseline model for both fraud datasets
- Configured with:
  - `max_iter=1000` for convergence
  - `solver='liblinear'` for flexibility with penalty types
  - `random_state=42` for reproducibility

#### 2. **Class Imbalance Handling**
Two complementary approaches are used:
- **Class Weighting**: `class_weight='balanced'` in model initialization
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied in the pipeline to balance training data

#### 3. **Stratified Split**
- Used `stratify=y` in `train_test_split()` to maintain class distribution
- Test size: 20%, Random state: 42
- Ensures both train and test sets have representative fraud/non-fraud ratios

#### 4. **Feature Engineering**
For **E-Commerce Fraud Data**:
- Date features: hour, day of week, month from `signup_time` and `purchase_time`
- Velocity features: transaction count per user
- Geographical features: IP to country mapping
- Time difference: minutes between signup and purchase

For **Credit Card Fraud Data**:
- All numerical features (V1-V28 from PCA, Amount, Time)
- Minimal feature engineering needed (data already preprocessed)

#### 5. **Pipeline Integration**
Both LogisticRegression and RandomForest are integrated into a comprehensive pipeline:
```python
ImbPipeline:
  1. Preprocessor (ColumnTransformer)
     - Numerical: SimpleImputer → StandardScaler
     - Categorical: SimpleImputer → OneHotEncoder
  2. SMOTE (for class imbalance)
  3. Model (LogisticRegression or RandomForest)
```

### Evaluation Metrics
All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Fraction of predicted frauds that are actual frauds (controls false positives)
- **Recall**: Fraction of actual frauds that are detected (controls false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve (probability calibration quality)

## Task 2b: Ensemble Model, Cross-Validation, and Model Selection

### Implementation Details

#### 1. **Hyperparameter Tuning**

**LogisticRegression Hyperparameter Grid**:
- `C`: [0.001, 0.01, 0.1, 1, 10, 100] (regularization strength)
- `penalty`: ['l1', 'l2'] (regularization type)
- `solver`: ['liblinear'] (compatible with both penalties)

**RandomForest Hyperparameter Grid**:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

**GradientBoosting Hyperparameter Grid**:
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 5, 7]
- `subsample`: [0.8, 1.0]

#### 2. **Search Strategy**
- **GridSearchCV**: For LogisticRegression (smaller grid)
- **RandomizedSearchCV**: For RandomForest (larger grid, more efficient)
- **GridSearchCV**: For GradientBoosting
- All use **Stratified 3-Fold CV** for tuning
- Optimization metric: **F1 Score** (balances precision and recall)

#### 3. **Cross-Validation**

**Configuration**:
- **Stratified K-Fold CV** with 5 folds
- Stratification ensures class distribution is maintained in each fold
- Reports **mean ± std** for all metrics

**Metrics Tracked**:
- Accuracy (mean ± std)
- Precision (mean ± std)
- Recall (mean ± std)
- F1 Score (mean ± std)
- ROC AUC (mean ± std)

#### 4. **Additional Models**

Three models are trained and compared:
1. **LogisticRegression** (Baseline + Tuned)
   - Interpretability: **HIGH**
   - Coefficients directly interpretable
   - Fast training and inference

2. **RandomForest** (Baseline + Tuned)
   - Interpretability: **MEDIUM**
   - Feature importance available
   - Robust to outliers and non-linear relationships

3. **GradientBoosting** (New ensemble model)
   - Interpretability: **LOW**
   - Often highest performance
   - Boosting trees sequentially

#### 5. **Model Comparison**

Models are compared side-by-side using:
- **Test Set Performance Table**: Shows all metrics for each model
- **Cross-Validation Summary**: Shows mean ± std for robustness assessment
- **Interpretability Analysis**: Qualitative assessment of model explainability

#### 6. **Model Selection Framework**

**Selection Criteria (Weighted)**:
- 40% Recall (most important for fraud detection - minimize false negatives)
- 30% F1 Score (balance precision and recall)
- 20% ROC AUC (probability calibration quality)
- 10% Interpretability (regulatory compliance and user trust)

**Justification Documentation**:
The script explicitly documents:
- Why recall is prioritized in fraud detection
- Trade-off between false positives (user frustration) and false negatives (financial loss)
- Interpretability importance for:
  - Regulatory compliance (explain why a transaction was flagged)
  - User trust (customers deserve explanations)
  - Model debugging and improvement

**Final Selection**:
- Composite score calculated for each model
- Best model recommended with clear justification
- Model saved to `models/` directory

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Data is Available
Make sure the following files exist:
- `data/raw/Fraud_Data.csv`
- `data/raw/IpAddress_to_Country.csv`
- `data/raw/creditcard.csv`

### 3. Run the Training Pipeline
```bash
python train_models.py
```

### 4. Monitor Progress
The script will:
- Print detailed progress logs
- Show cross-validation results
- Display model comparison tables
- Provide model selection justification
- Save models and results

## Output Files

After running, you'll find:

### Models Directory (`models/`)
- `ECommerce_Fraud_best_model.pkl`: Best model for e-commerce fraud
- `CreditCard_Fraud_best_model.pkl`: Best model for credit card fraud

### Results Directory (`results/`)
- `ECommerce_Fraud_model_comparison.csv`: Test set performance comparison
- `ECommerce_Fraud_cv_results.csv`: Cross-validation results
- `CreditCard_Fraud_model_comparison.csv`: Test set performance comparison
- `CreditCard_Fraud_cv_results.csv`: Cross-validation results

### MLflow Tracking (if available)
- `mlruns/`: MLflow experiment tracking directory
- View in browser: `mlflow ui` (then navigate to http://localhost:5000)

## Model Performance Interpretation

### Key Metrics for Fraud Detection

1. **Recall (Sensitivity)**: Most critical
   - High recall means we catch most fraudulent transactions
   - Low recall means fraudsters get through (false negatives)
   - Target: >90% for fraud detection

2. **Precision**: Important but secondary
   - High precision means fewer false alarms
   - Low precision means legitimate transactions are blocked
   - Target: Balance with recall (aim for >70%)

3. **F1 Score**: Overall balance
   - Harmonic mean of precision and recall
   - Good for comparing models
   - Target: Maximize while prioritizing recall

4. **ROC AUC**: Probability calibration
   - How well the model separates classes at different thresholds
   - Useful for threshold tuning
   - Target: >0.95 for good fraud detection

### Understanding the Trade-offs

**High Recall, Lower Precision**:
- Pro: Catches most fraud (security priority)
- Con: Some legitimate transactions flagged (user friction)

**High Precision, Lower Recall**:
- Pro: Very few false alarms (good user experience)
- Con: Some fraud slips through (financial loss)

**Balanced (High F1)**:
- Pro: Good overall performance
- Con: May not optimize for specific business needs

## Interpretability Considerations

### Why Interpretability Matters

1. **Regulatory Compliance**:
   - Fair Lending laws require explainable decisions
   - GDPR "right to explanation"
   - Financial regulations demand transparency

2. **User Trust**:
   - Customers deserve to know why transactions were flagged
   - Clear explanations reduce complaints
   - Builds confidence in the system

3. **Model Improvement**:
   - Interpretable models easier to debug
   - Can identify data quality issues
   - Facilitates feature engineering

### Model Interpretability Ranking

1. **LogisticRegression** (HIGH):
   - Coefficients show feature importance and direction
   - Easy to explain: "Each unit increase in X increases fraud probability by Y%"
   - SHAP values simple to compute

2. **RandomForest** (MEDIUM):
   - Feature importance available
   - Can't easily explain individual predictions
   - SHAP values more complex but available

3. **GradientBoosting** (LOW):
   - Complex sequential boosting process
   - Feature importance available but less intuitive
   - SHAP values computationally expensive

## Best Practices Implemented

1. ✅ **Stratified Splitting**: Maintains class distribution
2. ✅ **Class Imbalance Handling**: SMOTE + class weights
3. ✅ **Cross-Validation**: 5-fold stratified CV with mean ± std
4. ✅ **Hyperparameter Tuning**: Grid/Random search with CV
5. ✅ **Multiple Models**: Baseline + ensemble models
6. ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC AUC
7. ✅ **Model Selection Framework**: Performance + interpretability
8. ✅ **Experiment Tracking**: MLflow integration
9. ✅ **Reproducibility**: Random seeds fixed
10. ✅ **Documentation**: Clear justification for decisions

## Next Steps

### Model Deployment
- Use saved models in `models/` directory
- Load with: `joblib.load('models/ECommerce_Fraud_best_model.pkl')`
- Apply same preprocessing pipeline for inference

### Model Monitoring
- Track prediction distributions over time
- Monitor false positive/negative rates
- Retrain when performance degrades

### Threshold Tuning
- Adjust decision threshold based on business needs
- Use ROC curve to find optimal operating point
- Consider cost of false positives vs false negatives

### Feature Importance Analysis
- Use SHAP values for detailed explanations
- Identify most important fraud indicators
- Guide future feature engineering

## Troubleshooting

### Common Issues

1. **Memory Error**:
   - Reduce hyperparameter grid size
   - Use `RandomizedSearchCV` with fewer iterations
   - Process datasets separately

2. **Slow Training**:
   - Reduce `n_jobs` if system is overloaded
   - Use smaller CV folds (3 instead of 5)
   - Sample data for faster iteration

3. **MLflow Not Available**:
   - Script will run without MLflow
   - Results still saved to files
   - Install with: `pip install mlflow`

4. **Imbalanced-learn Import Error**:
   - Install with: `pip install imbalanced-learn`
   - Check version compatibility with scikit-learn

## References

- **SMOTE**: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- **Class Weighting**: sklearn documentation on `class_weight` parameter
- **Cross-Validation**: sklearn documentation on `StratifiedKFold`
- **Hyperparameter Tuning**: sklearn documentation on `GridSearchCV` and `RandomizedSearchCV`

---

**Author**: Fraud Detection Team  
**Date**: 2025-12-24  
**Version**: 1.0

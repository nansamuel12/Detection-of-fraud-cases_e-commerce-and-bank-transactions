import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import logging
from typing import List, Union, Tuple, Optional

class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass

def ip_to_country(ip: int, ip_country_df: pd.DataFrame) -> str:
    """
    Maps an integer IP address to a country using the range dataset.
    This function is optimized for pandas apply but can be slow for very large datasets without vectorization.
    
    Args:
        ip (int): IP address in integer format.
        ip_country_df (pd.DataFrame): DataFrame with 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.
        
    Returns:
        str: Country name or 'Unknown'.
    """
    try:
        match = ip_country_df[(ip >= ip_country_df['lower_bound_ip_address']) & 
                              (ip <= ip_country_df['upper_bound_ip_address'])]
        
        if not match.empty:
            return match['country'].values[0]
        return 'Unknown'
    except Exception as e:
        logging.warning(f"Error mapping IP {ip}: {e}")
        return 'Unknown'

def assign_countries(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently maps IPs to Countries using a merge_asof or comparable logic.
    Since pandas merge_asof requires sorting, we will use that.
    """
    logging.info("Mapping IP addresses to countries...")
    
    try:
        # Generic copy to avoid SettingWithCopy warnings on input df if it's a slice
        fraud_df = fraud_df.copy()
        ip_country_df = ip_country_df.copy()

        # Ensure types
        fraud_df['ip_address'] = fraud_df['ip_address'].astype(float).astype(int) # Ensure int
        ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(float).astype(int)
        
        # Sort for merge_asof
        fraud_df = fraud_df.sort_values('ip_address')
        ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
        
        # Use merge_asof to find the nearest lower bound
        merged = pd.merge_asof(fraud_df, ip_country_df, 
                               left_on='ip_address', 
                               right_on='lower_bound_ip_address', 
                               direction='backward')
        
        # Filter where IP is actually within the upper bound
        # If ip > upper_bound, it matched the lower bound of a range but exceeded the range.
        mask = merged['ip_address'] <= merged['upper_bound_ip_address']
        merged.loc[~mask, 'country'] = 'Unknown'
        merged['country'] = merged['country'].fillna('Unknown')
        
        return merged.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    except Exception as e:
        raise PreprocessingError(f"Error in assigning countries: {e}")

def feature_engineering_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """
    Extracts time-based features from date columns.
    """
    logging.info("Engineering date features...")
    try:
        for col in date_cols:
            if col in df.columns:
                prefix = col.replace('_time', '')
                df[f'{prefix}_hour'] = df[col].dt.hour
                df[f'{prefix}_day'] = df[col].dt.dayofweek
                df[f'{prefix}_month'] = df[col].dt.month
                
        # Specific logic for fraud data: diff between signup and purchase
        if 'signup_time' in df.columns and 'purchase_time' in df.columns:
            df['time_diff_minutes'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 60
            
        return df
    except Exception as e:
        raise PreprocessingError(f"Error in date feature engineering: {e}")

def feature_engineering_velocity(df: pd.DataFrame, user_col: str = 'user_id', time_col: str = 'purchase_time') -> pd.DataFrame:
    """
    Simple velocity check: Number of transactions per user in the dataset.
    (In a real stream, this would be a rolling window).
    """
    logging.info("Engineering velocity features...")
    try:
        if user_col in df.columns:
            df['user_tx_count'] = df.groupby(user_col)[user_col].transform('count')
        return df
    except Exception as e:
        raise PreprocessingError(f"Error in velocity feature engineering: {e}")

def get_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Returns a ColumnTransformer for preprocessing.
    """
    try:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        return preprocessor
    except Exception as e:
        raise PreprocessingError(f"Error creating preprocessor: {e}")

def handle_imbalance_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to the training data.
    """
    logging.info("Applying SMOTE to handle class imbalance...")
    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logging.info(f"Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        raise PreprocessingError(f"Error applying SMOTE: {e}")


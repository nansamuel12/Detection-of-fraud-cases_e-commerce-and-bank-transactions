import pandas as pd
import logging
import os
from typing import Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Args:
        file_path (str): The absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
        
    Raises:
        DataLoaderError: If file doesn't exist or loading fails (logged, returns None safe).
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading data: {e}")
        return None

def basic_cleaning(df: pd.DataFrame, date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Performs basic data cleaning: removing duplicates and converting date columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        date_columns (list, optional): List of column names to convert to datetime objects.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        logging.info("Starting basic cleaning...")
        
        if df is None:
            raise ValueError("Input DataFrame is None")

        df = df.copy()

        # Remove duplicates
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            logging.info(f"Removed {dropped_rows} duplicate rows.")
        else:
            logging.info("No duplicates found.")
            
        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logging.info(f"Converted column '{col}' to datetime.")
                else:
                    logging.warning(f"Date column '{col}' not found in DataFrame.")
        
        return df
    except Exception as e:
        logging.error(f"Error during basic cleaning: {e}")
        # Return original df in case of failure to avoid breaking pipeline completely if possible, 
        # or re-raise if critical. Here we log and return current state.
        return df


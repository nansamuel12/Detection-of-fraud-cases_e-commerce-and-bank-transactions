import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Args:
        file_path (str): The absolute or relative path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def basic_cleaning(df, date_columns=None):
    """
    Performs basic data cleaning: removing duplicates and converting date columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        date_columns (list): List of column names to convert to datetime objects.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        logging.info("Starting basic cleaning...")
        
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
        return df

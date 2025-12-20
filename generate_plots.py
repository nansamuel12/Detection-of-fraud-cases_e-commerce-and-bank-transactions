import sys
import os
import pandas as pd
import logging

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_data, basic_cleaning
from src.preprocessing import assign_countries, feature_engineering_dates, feature_engineering_velocity
from src.visualization import set_style, plot_class_distribution, plot_numerical_distributions, plot_categorical_breakdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    save_dir = os.path.join(os.getcwd(), 'photo')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    set_style()
    
    # --- 1. E-Commerce Data ---
    logging.info("Processing E-Commerce Data for Visualizations...")
    
    fraud_df = load_data('data/raw/Fraud_Data.csv')
    ip_country_df = load_data('data/raw/IpAddress_to_Country.csv')
    
    if fraud_df is not None:
        fraud_df = basic_cleaning(fraud_df, date_columns=['signup_time', 'purchase_time'])
        fraud_df = assign_countries(fraud_df, ip_country_df)
        fraud_df = feature_engineering_dates(fraud_df, ['signup_time', 'purchase_time'])
        
        # 1. Class Distribution
        plot_class_distribution(fraud_df['class'], 
                                title="Fraud Class Distribution (E-commerce)", 
                                save_path=os.path.join(save_dir, 'ecommerce_class_distribution.png'))
        
        # 2. Numerical Distributions
        # Plotting a subset to avoid too many files, focusing on key features
        plot_numerical_distributions(fraud_df, ['purchase_value', 'age', 'time_diff_minutes'], hue='class', save_dir=save_dir)
        
        # 3. Categorical Breakdown (Top Countries)
        top_countries = fraud_df['country'].value_counts().head(10).index
        subset_df = fraud_df[fraud_df['country'].isin(top_countries)]
        
        plot_categorical_breakdown(subset_df, ['country'], target='class', save_dir=save_dir)


    # --- 2. Credit Card Data ---
    logging.info("Processing Credit Card Data for Visualizations...")
    
    cc_df = load_data('data/raw/creditcard.csv')
    
    if cc_df is not None:
        cc_df = basic_cleaning(cc_df)
        
        # 1. Class Distribution
        plot_class_distribution(cc_df['Class'], 
                                title="Fraud Class Distribution (Credit Card)", 
                                save_path=os.path.join(save_dir, 'creditcard_class_distribution.png'))
                                
        # 2. Key Distributions (Amount, Time)
        plot_numerical_distributions(cc_df, ['Amount', 'Time'], hue='Class', save_dir=save_dir)

    logging.info("All visualizations saved to 'photo/' directory.")

if __name__ == "__main__":
    main()

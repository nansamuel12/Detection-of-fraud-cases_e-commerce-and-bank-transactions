import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    ip_to_country, 
    assign_countries, 
    feature_engineering_dates, 
    feature_engineering_velocity, 
    get_preprocessor,
    PreprocessingError
)

@pytest.fixture
def ip_country_df():
    return pd.DataFrame({
        'lower_bound_ip_address': [100, 200, 300],
        'upper_bound_ip_address': [199, 299, 399],
        'country': ['A', 'B', 'C']
    })

def test_ip_to_country(ip_country_df):
    assert ip_to_country(150, ip_country_df) == 'A'
    assert ip_to_country(250, ip_country_df) == 'B'
    assert ip_to_country(50, ip_country_df) == 'Unknown'
    assert ip_to_country(400, ip_country_df) == 'Unknown'

def test_assign_countries(ip_country_df):
    fraud_df = pd.DataFrame({
        'ip_address': [150, 50, 350, 250, 450]
    })
    
    result = assign_countries(fraud_df, ip_country_df)
    
    assert 'country' in result.columns
    assert result.iloc[0]['country'] == 'Unknown' # 50 (sorted first)
    # Wait, generic sort might change order.
    # Sorted IPs: 50, 150, 250, 350, 450
    # 50 -> Unknown
    # 150 -> A
    # 250 -> B
    # 350 -> C
    # 450 -> Unknown
    
    countries = result.set_index('ip_address')['country']
    assert countries[150] == 'A'
    assert countries[50] == 'Unknown'
    assert countries[350] == 'C'

def test_feature_engineering_dates():
    df = pd.DataFrame({
        'purchase_time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00']),
        'signup_time': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-02 10:30:00'])
    })
    
    result = feature_engineering_dates(df, ['purchase_time'])
    
    assert 'purchase_hour' in result.columns
    assert 'time_diff_minutes' in result.columns
    assert result.iloc[0]['time_diff_minutes'] == 60.0
    assert result.iloc[1]['time_diff_minutes'] == 30.0

def test_feature_engineering_velocity():
    df = pd.DataFrame({
        'user_id': [1, 1, 2, 1, 2],
        'purchase_time': pd.date_range('2023-01-01', periods=5)
    })
    
    result = feature_engineering_velocity(df)
    
    assert 'user_tx_count' in result.columns
    assert result.iloc[0]['user_tx_count'] == 3 # ID 1 appears 3 times
    assert result.iloc[2]['user_tx_count'] == 2 # ID 2 appears 2 times

def test_get_preprocessor():
    prep = get_preprocessor(['num1'], ['cat1'])
    assert prep is not None

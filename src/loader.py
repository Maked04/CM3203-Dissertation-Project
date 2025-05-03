# Utilities for loading the data
import pandas as pd
import numpy as np
import os
import json
from token_lifecycle_utils import remove_price_anomalies
import sqlite3

def get_data_dir():
    """Get the absolute path to the data directory."""
    # Get the directory where this script is located (src/data)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up to project root (parent of src)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Data directory is at the same level as src
    data_dir = os.path.join(project_root, "data")

    
    return data_dir

def load_data_file(file_name):
    file_path = os.path.join(get_data_dir(), file_name)
    return open(file_path)


def get_metric_by_tx_sig(tx_sig):
    db_path = os.path.join(get_data_dir(), "wallet_metrics.db")
    if not os.path.exists(db_path):
        db_path =  "D:/wallet_metrics.db"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Makes rows accessible like dicts
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT * FROM metrics WHERE tx_sig = ?", (tx_sig,)
        ).fetchone()
    
    return dict(row) if row else None


def load_feature_vector_data(file_name: str = "Jan_25_Token_Features.csv") -> pd.DataFrame:
    """Load feature vector data from the data directory."""
    file_path = os.path.join(get_data_dir(), file_name)
    df = pd.read_csv(file_path)
    return df

def load_token_data(token_address: str, folder_name: str = "Jan_25_Tokens") -> pd.DataFrame:
    """
    Load all transaction data for a specific token from its CSV file.
    
    Args:
        token_address (str): The token address to load data for
        folder_name (str): Name of the folder containing token data
    
    Returns:
        pd.DataFrame: DataFrame with all transaction data, indexed by block_time
    """
    try:
        # Construct file path using absolute path
        base_path = os.path.join(get_data_dir(), folder_name)
        file_path = os.path.join(base_path, f"{token_address}.csv")
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Convert numeric columns to appropriate types
        numeric_columns = [
            'slot', 
            'block_time',
            'fee', 
            'token_price',
            'bc_spl_before',
            'bc_spl_after',
            'bc_sol_before',
            'bc_sol_after',
            'signer_spl_before',
            'signer_spl_after',
            'signer_sol_before',
            'signer_sol_after'
        ]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert boolean columns
        df['is_creator'] = df['is_creator'].astype(bool)
        
        # Set block_time as index
        df = df.set_index('block_time')
        
        # Add token address as attribute
        df.attrs['token_address'] = token_address
        
        return df
        
    except FileNotFoundError:
        print(f"Error: No data file found for token {token_address}")
        return None
    except Exception as e:
        print(f"Error loading data for token {token_address}: {str(e)}")
        print(f"File path attempted: {file_path}")
        return None

def load_token_price_data(token_address: str, folder_name: str = "Jan_25_Tokens", remove_anomalies=True, use_datetime=True) -> pd.DataFrame:
    """
    Load and clean price data for a specific token from its CSV file.
    
    Args:
        token_address (str): The token address to load data for
        folder_name (str): Name of the folder containing token data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime index and price data
    """
    try:
        # Construct file path using absolute path
        base_path = os.path.join(get_data_dir(), folder_name)
        file_path = os.path.join(base_path, f"{token_address}.csv")
        
        # Read CSV
        df = pd.read_csv(file_path)

        if remove_anomalies:
            df = remove_price_anomalies(df)
        
        # Convert block_time to datetime
        if use_datetime:
            df['block_time'] = pd.to_datetime(df['block_time'])
        
        # Create processed dataframe with just time and price
        price_data = pd.DataFrame({
            'timestamp': df['block_time'],
            'price': df['token_price']
        }).set_index('timestamp')
        
        # Store cleaning stats as attributes
        price_data.attrs['token_address'] = token_address
        
        return price_data
        
    except FileNotFoundError:
        print(f"Error: No data file found for token {token_address}")
        return None
    except Exception as e:
        print(f"Error loading data for token {token_address}: {str(e)}")
        return None

def list_available_tokens(folder_name: str = "Jan_25_Tokens") -> list:
    """
    Get a list of all available token addresses from the data folder.
    
    Args:
        folder_name (str): Name of the folder containing token data
    
    Returns:
        list: List of token addresses (without .csv extension)
    """
    try:
        # Get the absolute path to the tokens directory
        base_path = os.path.join(get_data_dir(), folder_name)
        # Get all CSV files in the directory
        files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        # Remove .csv extension
        tokens = [f[:-4] for f in files]
        return tokens
    except Exception as e:
        print(f"Error listing tokens: {str(e)}")
        return []

def get_token_stats(price_data: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for a token's price data.
    
    Args:
        price_data (pd.DataFrame): DataFrame with price data
        
    Returns:
        dict: Dictionary containing price statistics
    """
    return {
        'initial_price': price_data['price'].iloc[0],
        'final_price': price_data['price'].iloc[-1],
        'max_price': price_data['price'].max(),
        'min_price': price_data['price'].min(),
        'price_change_pct': ((price_data['price'].iloc[-1] - price_data['price'].iloc[0]) 
                            / price_data['price'].iloc[0] * 100),
        'trading_days': (price_data.index[-1] - price_data.index[0]).days,
        'transaction_count': len(price_data)
    }

def load_time_bucket_data(folder_name="time_bucket_1"):
    folder_path = os.path.join(get_data_dir(), "time_bucket_data", folder_name)
    
    with open(os.path.join(folder_path, "config.json")) as f:
        config = json.load(f)

    token_time_buckets = {}
    for token in os.listdir(folder_path):
        token_path = os.path.join(folder_path, token)
        if token == "config.json" or not os.path.isdir(token_path):
            continue

        X = np.load(os.path.join(token_path, "X.npy"))
        y = np.load(os.path.join(token_path, "y.npy"))
        bucket_times = np.load(os.path.join(token_path, "bucket_times.npy"))

        token_time_buckets[token] = {"X": X, "y": y, "bucket_times": bucket_times}

    return token_time_buckets, config
 
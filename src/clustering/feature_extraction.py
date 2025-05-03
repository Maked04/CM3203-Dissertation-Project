from loader import load_token_data, get_data_dir
from token_lifecycle_utils import remove_price_anomalies, segment_token_lifecycle
import pandas as pd
import math
import os


def get_inital_liquidity(df):
    """Get initial liquidity from first trade if it's a creator trade"""
    creator_txs = df[df['is_creator'] == True]
    if len(creator_txs) == 1:
        return creator_txs.iloc[0]['bc_sol_after']
    
    return None

def get_active_lifetime(df):
    """Get time difference between first and last trade in seconds"""
    return df.index[-1] - df.index[0]

def get_unique_trader_count(df):
    """Count unique traders"""
    return df['signer'].nunique()

def get_average_trade_size(df):
    """Calculate average trade size in SOL"""
    return (df['signer_sol_after'] - df['signer_sol_before']).abs().mean()

def get_max_market_cap(df):
    """Get maximum pool SOL value"""
    return df['bc_sol_after'].max()

def get_trade_frequency(df):
    """Calculate average time between trades in seconds"""
    return get_active_lifetime(df) / len(df)

def get_price_volatility(df):
    prices = df["token_price"].dropna()
    if len(prices) == 0:
        return None
    
    avg_price = prices.mean()
    square_diff = [math.pow((price - avg_price), 2) for price in prices]
    variance = sum(square_diff) / len(prices)
    std = math.sqrt(variance)

    return std


def get_token_feature_vector(token_address):
    """
    Create feature vector for a token.
    
    Args:
        token_address (str): Token address to analyze
        
    Returns:
        pd.DataFrame: Single row DataFrame with features
    """
    # Load data
    df = load_token_data(token_address)

    launch_df, peak_df, decline_df, df = segment_token_lifecycle(df)
        
    # Clean data for price features
    cleaned_df = remove_price_anomalies(df)

    if cleaned_df.empty:
        return None
    
    features = pd.DataFrame({
        "token_address": token_address,
        "trade_frequency": [get_trade_frequency(df)],
        "average_trade_size": [get_average_trade_size(df)],
        "unique_traders": [get_unique_trader_count(df)],
        "max_market_cap": [get_max_market_cap(df)],
        "initial_liquidity": [get_inital_liquidity(df)],
        "price_volatility": [get_price_volatility(cleaned_df)],
        "total_trades": [len(df)],
        "lifetime_seconds": [get_active_lifetime(df)]
    })
    
    return features

def save_feature_vectors(token_addresses, output_file="Jan_25_Token_Features.csv"):
    all_features = []
    skipped_tokens = []
    
    for token in token_addresses:
        feature_vector = get_token_feature_vector(token)
        
        # Check if feature vector exists and has no NA values
        if feature_vector is not None and not feature_vector.isna().any().any():
            all_features.append(feature_vector)
        else:
            skipped_tokens.append(token)
    
    if not all_features:
        print("No valid feature vectors found")
        return
    
    combined_features = pd.concat(all_features)
    
    # Save to CSV
    combined_features.to_csv(os.path.join(get_data_dir(), output_file))
    
    # Print summary
    print(f"Successfully processed {len(all_features)} tokens")
    print(f"Skipped {len(skipped_tokens)} tokens due to undefined values")
    if skipped_tokens:
        print("First few skipped tokens:")
        for token in skipped_tokens[:5]:
            print(f"- {token}")
import pandas as pd

def remove_price_anomalies(df, threshold=3.0):
    """
    Remove price anomalies from a DataFrame using Median Absolute Deviation (MAD).
    
    Args:
        df (pd.DataFrame): DataFrame containing token price data
        threshold (float): MAD threshold for anomaly detection (default: 3.0)
    
    Returns:
        pd.DataFrame: DataFrame with anomalies removed
    """
    if df is None or df.empty:
        return df
    
    # Calculate median and MAD for token prices
    median_price = df['token_price'].median()
    mad = (df['token_price'] - median_price).abs().median()
    
    if mad == 0:
        return df
    
    # Calculate modified z-scores
    modified_z_scores = (df['token_price'] - median_price).abs() / mad
    
    # Create mask for valid prices
    valid_prices = modified_z_scores < threshold
    
    # Get cleaned DataFrame
    cleaned_df = df[valid_prices].copy()
    
    return cleaned_df


def trim_main_trading_period_old(df, min_trades_per_hour=50, window_size_hours=1):
    """
    Identify and extract the main trading period based on trade frequency.
    
    Args:
        df: DataFrame with trading data (timestamp indexed)
        min_trades_per_hour: Minimum trade frequency to consider active
        window_size_hours: Size of the window to calculate trade frequency
    """
    # Convert window_size to seconds
    window_size = window_size_hours * 3600
    
    # Create a Series of 1s with the same index as df
    trades = pd.Series(1, index=df.index)
    
    # Calculate trade frequency in rolling windows
    def count_trades_in_window(timestamp):
        window_start = timestamp - window_size
        return trades[(trades.index >= window_start) & 
                      (trades.index <= timestamp)].count()
    
    trade_frequency = pd.Series(index=df.index)
    for ts in df.index:
        trade_frequency[ts] = count_trades_in_window(ts) / window_size_hours
    
    # Find active trading periods (above threshold)
    active_periods = trade_frequency >= min_trades_per_hour
    
    # Find the first and last timestamps of the longest continuous active period
    active_blocks = []
    current_block = []
    
    for ts, is_active in zip(df.index, active_periods):
        if is_active:
            current_block.append(ts)
        elif current_block:
            active_blocks.append(current_block)
            current_block = []
    
    if current_block:  # Don't forget the last block
        active_blocks.append(current_block)
    
    if not active_blocks:
        return df  # No active blocks found
        
    # Find the longest continuous active block
    longest_block = max(active_blocks, key=len)
    
    # Return only the data within the longest active block
    return df.loc[min(longest_block):max(longest_block)]


def trim_main_trading_period(df, min_trades_per_hour=50, window_size_hours=1):
    """
    Identify and extract the main trading period based on trade frequency.
    
    Args:
        df (pd.DataFrame): DataFrame with trading data (timestamp indexed)
        min_trades_per_hour (int): Minimum trade frequency to consider active
        window_size_hours (int): Size of the window to calculate trade frequency
    
    Returns:
        pd.DataFrame: Trimmed DataFrame containing only the main trading period
    """
    if df is None or df.empty:
        print("Warning: Received empty or None DataFrame.")
        return df
    
    # Ensure the index is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Convert window_size to timedelta
    window_size = pd.Timedelta(hours=window_size_hours)
    
    # Create a Series of 1s with the same index as df
    trades = pd.Series(1, index=df.index)

    # Calculate trade frequency in rolling windows
    def count_trades_in_window(timestamp):
        window_start = timestamp - window_size  # Now using Timedelta
        return trades[(trades.index >= window_start) & (trades.index <= timestamp)].count()
    
    trade_frequency = pd.Series(index=df.index)
    for ts in df.index:
        trade_frequency[ts] = count_trades_in_window(ts) / window_size_hours

    # Find active trading periods (above threshold)
    active_periods = trade_frequency >= min_trades_per_hour

    # Find the first and last timestamps of the longest continuous active period
    active_blocks = []
    current_block = []
    
    for ts, is_active in zip(df.index, active_periods):
        if is_active:
            current_block.append(ts)
        elif current_block:
            active_blocks.append(current_block)
            current_block = []
    
    if current_block:  # Don't forget the last block
        active_blocks.append(current_block)
    
    if not active_blocks:
        return df  # No active blocks found
        
    # Find the longest continuous active block
    longest_block = max(active_blocks, key=len)

    # Return only the data within the longest active block
    return df.loc[min(longest_block):max(longest_block)]


def identify_launch_phase(df, method="trades", percentage=0.1, hours=24):
    """
    Extract the launch phase data from token trading history.
    
    Args:
        df: DataFrame with trading data indexed by timestamp
        method: Identification method - "trades", "time", or "hybrid"
        percentage: Percentage of total trades to consider as launch phase (for "trades" method)
        hours: Number of hours from first trade to consider as launch phase (for "time" method)
    
    Returns:
        DataFrame containing only the launch phase data
    """    
    if method == "trades":
        # Take the first X% of trades
        n_trades = int(len(df) * percentage)
        return df.iloc[:n_trades]
    
    elif method == "time":
        # Take all trades within X hours of the first trade
        first_timestamp = df.index.min()
        cutoff_timestamp = first_timestamp + (hours * 3600)  # Convert hours to seconds
        return df[df.index <= cutoff_timestamp]
    
    elif method == "hybrid":
        # Take the earlier cutoff between X% of trades and Y hours
        n_trades = int(len(df) * percentage)
        first_timestamp = df.index.min()
        cutoff_timestamp = first_timestamp + (hours * 3600)
        
        trades_cutoff = df.iloc[n_trades-1].name if n_trades < len(df) else df.index.max()
        time_cutoff = cutoff_timestamp
        
        return df[df.index <= min(trades_cutoff, time_cutoff)]
    
    return df


def identify_peak_phase(df, method="market_cap", window_size=0.2):
    """
    Extract the peak phase data from token trading history.
    
    Args:
        df: DataFrame with trading data
        method: Metric to identify peak - "market_cap", "volume", "price", "trade_frequency"
        window_size: Size of window around peak (as fraction of total trades)
    
    Returns:
        DataFrame containing only the peak phase data
    """

    if method == "market_cap":
        # Find the timestamp of maximum market cap
        peak_idx = df["bc_sol_after"].idxmax()
    
    elif method == "price":
        # Find the timestamp of maximum price
        peak_idx = df["token_price"].idxmax()
    
    elif method == "volume" or method == "trade_frequency":
        # Use a rolling window to find period of highest trading activity
        trade_counts = pd.Series(1, index=df.index)
        window_length = int(len(df) * 0.1)  # 10% of total trades
        rolling_count = trade_counts.rolling(window_length).sum()
        peak_idx = rolling_count.idxmax()
    
    # Calculate window boundaries
    window_half_size = int(len(df) * window_size / 2)
    
    # Find position of peak in the dataframe
    peak_position = df.index.get_loc(peak_idx)
    if isinstance(peak_position, slice):
        peak_position = peak_position.start  # Pick the first position
    
    # Calculate window boundaries
    start_pos = max(0, peak_position - window_half_size)
    end_pos = min(len(df), peak_position + window_half_size)
    
    # Extract the window around the peak
    return df.iloc[start_pos:end_pos]


def segment_token_lifecycle(df):
    """
    Segment a token's trading history into launch, peak, and decline phases.
    
    Returns:
        tuple: (launch_df, peak_df, decline_df, full_df)
    """
    if len(df) < 10:  # Not enough data to segment meaningfully
        return df, df, df, df
    
    # Identify launch phase (first 10% of trades or 24 hours, whichever comes first)
    launch_df = identify_launch_phase(df, method="hybrid", percentage=0.1, hours=24)
    launch_df = trim_main_trading_period(launch_df, window_size_hours=0.5)
    
    # Identify peak phase (window around maximum market cap)
    peak_df = identify_peak_phase(df, method="market_cap", window_size=0.1)
    peak_df = trim_main_trading_period(peak_df, window_size_hours=0.5)
    
    # Everything after peak phase is considered decline phase
    last_peak_timestamp = peak_df.index.max()
    decline_df = df[df.index > last_peak_timestamp]
    decline_df = trim_main_trading_period(decline_df, window_size_hours=6)
    
    return launch_df, peak_df, decline_df, df
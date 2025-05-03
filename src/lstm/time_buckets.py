import numpy as np
from feature_extraction import FeaturesConfig

class TimeBucketConfig:
    def __init__(
        self,
        bucket_size=30,
        prediction_horizon=1,
        min_txs_per_second=1,
        use_max_multiple=True,
        use_cumulative_price_change=False,
        step_size=1,  # How many seconds to advance when creating the next bucket
        max_seq_length=300
    ):
        # Bucket configuration
        self.bucket_size = bucket_size
        self.prediction_horizon = prediction_horizon
        self.min_txs_per_second = min_txs_per_second
        
        # Target variable configuration
        self.use_max_multiple = use_max_multiple
        self.use_cumulative_price_change = use_cumulative_price_change
        
        # Step size for sliding window (defaults to 1 second)
        self.step_size = step_size

        # Max sequence length per time buclet
        self.max_seq_length = max_seq_length
        
        # Validate configuration
        if self.use_max_multiple and self.use_cumulative_price_change:
            raise ValueError("Cannot use both max_multiple and cumulative_price_change for target calculation")

def get_time_buckets(feature_matrix, timestamps, prices, config: TimeBucketConfig):
    """
    Creates sliding windows of time buckets from feature matrix for time series prediction.
   
    Args:
        feature_matrix: Matrix containing only the features
        timestamps: Array of timestamps for each feature row
        prices: Array of token prices for each feature row
        config: TimeBucketConfig object with settings for bucket creation and target calculation
       
    Returns:
        X: Input sequences as standard list as they may have different lengths so cant be numpy array
        y: Target values calculated according to the configuration
        bucket_times: List of tuples containing (start_time, end_time) for each bucket
    """
    X, y = [], []
    bucket_times = []  # List to store start and end times of each bucket
    
    min_time = np.min(timestamps)
    max_time = np.max(timestamps)
    
    for start_time in np.arange(min_time, max_time, config.step_size):
        end_time = start_time + config.bucket_size
        
        # Get indices of times falling in range
        bucket_indexes = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
        bucket_features = feature_matrix[bucket_indexes]
        
        if len(bucket_features) / config.bucket_size < config.min_txs_per_second:
            continue
            
        # Get target variable
        target_variable = None
        horizon_end_time = end_time + (config.bucket_size * config.prediction_horizon)
        horizon_indexes = np.where((timestamps >= end_time) & (timestamps < horizon_end_time))[0]
        
        if config.use_max_multiple:
            if len(bucket_indexes) == 0:
                continue
            price_at_end = prices[bucket_indexes[-1]]  # Price at end of bucket
            horizon_prices = prices[horizon_indexes]
            if horizon_prices.size > 0:
                max_upside = max(horizon_prices) / price_at_end - 1
                max_downside = min(horizon_prices) / price_at_end - 1
                target_variable = max_upside if abs(max_upside) > abs(max_downside) else max_downside  # Max move
        elif config.use_cumulative_price_change:
            price_changes = np.diff(prices[horizon_indexes]) / prices[horizon_indexes[:-1]] if len(horizon_indexes) > 1 else np.array([])
            target_variable = np.prod(1 + price_changes) - 1 if len(price_changes) > 0 else None  # Cumulative change
       
        if target_variable is not None and not np.isnan(target_variable) and not np.isinf(target_variable):
            X.append(bucket_features)
            y.append(target_variable)
            bucket_times.append((start_time, end_time))  # Store the start and end time of this bucket
    
    X = pad_sequences_with_price_importance(X, config.max_seq_length)
    y = np.array(y)
    return X, y, bucket_times


def pad_sequences_with_price_importance(X_list, max_seq_length=300):
    """
    Pads sequences to a fixed length or truncates them based on price impact importance.
   
    Args:
        X_list: List of feature arrays with variable lengths
        max_seq_length: Maximum sequence length to pad/truncate to
   
    Returns:
        X_padded: Numpy array with shape (n_samples, max_seq_length, n_features)
    """
    if not X_list:
        return np.array([])
   
    n_samples = len(X_list)
    n_features = X_list[0].shape[1]
   
    # Initialize padded array with zeros
    X_padded = np.zeros((n_samples, max_seq_length, n_features))
   
    for i, sequence in enumerate(X_list):
        if len(sequence) <= max_seq_length:
            # If sequence is shorter than max_length, use left padding
            # Place sequence at the end of the padded array
            start_idx = max_seq_length - len(sequence)
            X_padded[i, start_idx:, :] = sequence
        else:
            # If sequence is longer, keep transactions with largest price changes
            # Assuming the last column is price_change
            price_change_col = sequence.shape[1] - 1
           
            # Calculate absolute price changes for importance
            abs_price_changes = np.abs(sequence[:, price_change_col])
           
            # Get indices of transactions sorted by importance (largest price change first)
            important_indices = np.argsort(abs_price_changes)[::-1][:max_seq_length]
           
            # Sort these indices to maintain temporal order
            important_indices = np.sort(important_indices)
           
            # Select the most important transactions while preserving order
            important_transactions = sequence[important_indices]
           
            # Fill the padded array (with left padding if needed)
            start_idx = max_seq_length - len(important_transactions)
            X_padded[i, start_idx:, :] = important_transactions
   
    return X_padded

def reduce_time_bucket_features(X: np.ndarray, config: FeaturesConfig) -> np.ndarray:
    # Assuming config.__dict__ contains a dictionary with the feature mask as boolean values
    features_mask = list(config.__dict__.values())
    
    # Filter the features based on the mask
    filtered = X[:, :, features_mask]
    
    return filtered
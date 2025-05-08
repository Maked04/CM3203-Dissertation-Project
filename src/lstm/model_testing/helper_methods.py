import numpy as np
import os
import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # type: ignore
from src.lstm.time_buckets import reduce_time_bucket_features
from src.lstm.feature_extraction import FeaturesConfig
from src.loader import load_time_bucket_data


def generate_data(features_config, time_bucket_folder, test_size):
    # Get the train test data set used to train the model were testing

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    token_time_buckets, time_bucket_config = load_time_bucket_data(time_bucket_folder)

    token_datasets = []
    for token_address, data in token_time_buckets.items():
        X = data["X"]
        y = data["y"]
        bucket_times = data["bucket_times"]

        # Only get the features listed in features_config
        X = reduce_time_bucket_features(X, features_config)

        token_datasets.append((X, y, token_address, bucket_times))

    # Combine all token data
    all_X = np.vstack([data[0] for data in token_datasets])
    all_y = np.vstack([data[1].reshape(-1, 1) for data in token_datasets])

    # Scale features
    num_samples, time_steps, features = all_X.shape
    X_reshaped = all_X.reshape(num_samples * time_steps, features)
    X_scaled = X_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, time_steps, features)

    # Scale target variable also using StandardScaler to preserve direction
    y_scaled = y_scaler.fit_transform(all_y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

def order_features_config(features_config_dict):
    """
    Orders a features_config dict according to the expected FeaturesConfig fields.
    If a field is missing, it defaults to False.
    """
    feature_keys = [
        "trade_size_ratio",
        "liquidity_ratio",
        "relative_time",
        "absolute_time",
        "price_change",
        "wallet_trade_size_deviation",
        "volume_prior",
        "trade_count_prior",
        "rough_pnl",
        "average_roi",
        "win_rate",
        "average_hold_duration"
    ]
    return [features_config_dict.get(key, False) for key in feature_keys]


def get_active_features(features_config: FeaturesConfig):
    # Get the attribute names of the instance
    active_features = [
        attr for attr, value in vars(features_config).items() if value
    ]
    return active_features


def get_test_tokens_with_large_pred(token_datasets, test_size, y_pred_actual, min_pred_size=1.5):
    total_buckets = sum(len(data[1]) for data in token_datasets)
    test_start_idx = int((1 - test_size) * total_buckets)

    fully_test_tokens = {}
    current_idx = 0
    y_pred_test_index = 0  # Index within y_pred_actual

    for X, y, token_address, bucket_times in token_datasets:
        token_len = len(y)
        token_start_idx = current_idx
        current_idx += token_len

        # Only process tokens fully in the test set
        if token_start_idx >= test_start_idx:
            bucket_pred_map = []

            for i in range(token_len):
                if y_pred_test_index >= len(y_pred_actual):
                    break  # Prevent overflow if mismatch in lengths
                pred = y_pred_actual[y_pred_test_index].item()
                if pred >= min_pred_size:
                    bucket_pred_map.append({
                        'bucket_time': tuple(bucket_times[i]),
                        'prediction': pred
                    })
                y_pred_test_index += 1

            if bucket_pred_map:
                fully_test_tokens[token_address] = bucket_pred_map

    return fully_test_tokens


def get_token_datasets(time_bucket_folder):
    token_time_buckets, time_bucket_config = load_time_bucket_data(time_bucket_folder)

    token_datasets = []
    for token_address, data in token_time_buckets.items():
        X = data["X"]
        y = data["y"]
        bucket_times = data["bucket_times"]

        token_datasets.append((X, y, token_address, bucket_times))

    return token_datasets

def remove_zero_rows(arr):
    # Check where all elements in each row are zero along the last dimension (num_features)
    non_zero_rows = np.all(arr == 0, axis=(1, 2))
    
    # Use boolean indexing to remove the rows where all features are zero
    return arr[~non_zero_rows]
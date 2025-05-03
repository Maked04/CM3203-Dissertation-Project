import numpy as np

class SlidingWindowConfig:
    def __init__(
        self,
        sequence_length=10,
        prediction_horizon=1,
        use_time_horizon=False
    ):
        """
        Configuration for creating sliding windows for time series prediction.

        Args:
            sequence_length (int): Number of past transactions to use as input.
            prediction_horizon (int): Number of future transactions or seconds (based on time mode).
            use_time_horizon (bool): If True, uses seconds for prediction horizon; else, uses number of future transactions.
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.use_time_horizon = use_time_horizon



def get_sliding_windows(feature_matrix, timestamps, config: SlidingWindowConfig):
    """
    Creates sliding windows from the feature matrix for time series prediction.

    Args:
        feature_matrix (np.ndarray): Matrix of features (each row = transaction features).
        timestamps (np.ndarray): Timestamps corresponding to each row (used if time horizon is enabled).
        config (SlidingWindowConfig): Configuration object with window parameters.

    Returns:
        X (np.ndarray): Input sequences (num_samples, sequence_length, num_features)
        y (np.ndarray): Targets (cumulative price change)
    """
    X, y = [], []

    feature_times = np.array(timestamps)
    features = feature_matrix[:, :-1]  # All but last column are features
    price_changes = feature_matrix[:, -1]  # Last column is price_change

    for i in range(len(features) - config.sequence_length - config.prediction_horizon + 1):
        X_seq = features[i : i + config.sequence_length]
        X.append(X_seq)

        if config.use_time_horizon:
            pred_start_time = feature_times[i + config.sequence_length]
            pred_end_time = pred_start_time + config.prediction_horizon
            horizon_idxs = np.where(
                (feature_times > pred_start_time) & (feature_times <= pred_end_time)
            )[0]
        else:
            horizon_idxs = np.arange(i + config.sequence_length, i + config.sequence_length + config.prediction_horizon)

        # Filter only valid indices
        horizon_idxs = horizon_idxs[horizon_idxs < len(price_changes)]

        future_changes = price_changes[horizon_idxs]

        if len(future_changes) == 0:
            X.pop()
            continue

        cumulative_change = np.prod(1 + future_changes) - 1

        if not np.isnan(cumulative_change) and not np.isinf(cumulative_change):
            y.append(cumulative_change)
        else:
            X.pop()

    return np.array(X), np.array(y)
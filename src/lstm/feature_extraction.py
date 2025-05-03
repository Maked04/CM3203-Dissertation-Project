from src.loader import load_token_data, get_metric_by_tx_sig
from src.token_lifecycle_utils import remove_price_anomalies
import numpy as np


class FeaturesConfig:
    def __init__(
        self,
        trade_size_ratio=False,
        liquidity_ratio=False,
        relative_time=False,
        absolute_time=False,
        price_change=False,
        wallet_trade_size_deviation=False,
        volume_prior=False,
        trade_count_prior=False,
        rough_pnl=False,
        average_roi=False,
        win_rate=False,
        average_hold_duration=False
    ):
        # Standard features
        self.trade_size_ratio = trade_size_ratio
        self.liquidity_ratio = liquidity_ratio
        self.relative_time = relative_time
        self.absolute_time = absolute_time
        self.price_change = price_change
        # Wallet specific features
        self.wallet_trade_size_deviation = wallet_trade_size_deviation
        self.volume_prior = volume_prior
        self.trade_count_prior = trade_count_prior
        self.rough_pnl = rough_pnl
        self.average_roi = average_roi
        self.win_rate = win_rate
        self.average_hold_duration = average_hold_duration


def get_trade_size_ratio(row):
    """Calculate the ratio of trade size to balance after/before trade."""
    if row["bc_spl_after"] > row["bc_spl_before"]:  # Sell
        if row["bc_spl_after"] == 0 or not np.isfinite(row["bc_spl_after"]):
            return None  # Cannot calculate ratio with zero denominator or infinite value
        ratio = abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_after"]
        return None if not np.isfinite(ratio) else ratio
    elif row["bc_spl_after"] < row["bc_spl_before"]:  # Buy
        if row["bc_spl_before"] == 0 or not np.isfinite(row["bc_spl_before"]):
            return None  # Cannot calculate ratio with zero denominator or infinite value
        ratio = abs(row["bc_spl_after"] - row["bc_spl_before"]) / row["bc_spl_before"]
        return None if not np.isfinite(ratio) else ratio
    return 0  # No change

def get_trade_liquidity_ratio(row):
    """Calculate the ratio of SOL balance to SPL balance before trade."""
    if row["bc_spl_before"] == 0 or not np.isfinite(row["bc_spl_before"]) or not np.isfinite(row["bc_sol_before"]):
        return None  # Cannot calculate ratio with zero denominator or infinite values
    ratio = row["bc_sol_before"] / row["bc_spl_before"]
    return None if not np.isfinite(ratio) else ratio

def validate_features(features):
    for feature in features:
        if feature is None:
            return False
        if not isinstance(feature, (int, float, np.number)):
            return False
        if not np.isfinite(feature):  # This includes both inf and nan
            return False
    return True


def get_token_features_and_metadata(token_address, min_sol_size=0.1):
    """
    Extract features from token data along with metadata needed for target creation.
    
    Returns:
        feature_matrix: Matrix containing only the features
        timestamps: Array of timestamps for each row
        prices: Array of token prices for each row
    """
    df = load_token_data(token_address)
    cleaned_df = remove_price_anomalies(df)

    # Compute price changes
    price_changes = cleaned_df["token_price"].pct_change().iloc[1:].values  # Convert to array
    
    start_time = cleaned_df.iloc[0]["slot"]  # Initialize start time with the first transaction's time
    previous_time = cleaned_df.iloc[0]["slot"]  # Initialize previous time with the first transaction's time

    # Feature matrix
    all_features = []
    timestamps = []
    prices = []
    for i, (_, row) in enumerate(cleaned_df.iloc[1:].iterrows()): 
        if abs(row["bc_sol_before"] - row["bc_sol_after"]) < min_sol_size:
            continue

        wallet_metrics = get_metric_by_tx_sig(row["tx_sig"])
        if wallet_metrics is None:
            #continue
            wallet_metrics = {"trade_size_deviation": 1,
                              "volume_prior": 1,
                              "trade_count_prior": 1,
                              "rough_pnl": 1,
                              "average_roi": 1,
                              "win_rate": 1,
                              "average_hold_duration": 1}

        relative_time = row["slot"] - previous_time
        absolute_time = row["slot"] - start_time

        previous_time = row["slot"]  # Update previous time for relative time calculation

        # IMPORTANT, order of features must match the order of variables in features config
        features = [get_trade_size_ratio(row),
                    get_trade_liquidity_ratio(row),
                    relative_time,
                    absolute_time,
                    price_changes[i],
                    wallet_metrics["trade_size_deviation"],
                    wallet_metrics["volume_prior"],
                    wallet_metrics["trade_count_prior"],
                    wallet_metrics["rough_pnl"],
                    wallet_metrics["average_roi"],
                    wallet_metrics["win_rate"],
                    wallet_metrics["average_hold_duration"],
                    ]
        
        if validate_features(features):
            all_features.append(features)

            # Store metadata for target calculation separately
            timestamps.append(row.name)
            prices.append(row["token_price"])

    # Return separate arrays for features and metadata
    return np.array(all_features), np.array(timestamps), np.array(prices)
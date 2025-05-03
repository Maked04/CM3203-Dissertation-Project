import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def evaluate_large_move_model(y_true, y_pred, threshold=0.02,
                               prediction_weight=0.4, freq_weight=0.3, error_weight=0.3, f1_weight=0.5):
    """
    Evaluates how well predictions capture large market moves.

    Parameters:
    - y_true: array-like of true returns
    - y_pred: array-like of predicted returns
    - threshold: minimum % move considered 'large' (default 2%)
    - *_weight: weights for combining submetrics into a final score

    Returns:
    - metrics: dict of all submetrics and the combined score
    """

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Identify large moves
    true_large_idx = np.where(np.abs(y_true) >= threshold)[0]
    pred_large_idx = np.where(np.abs(y_pred) >= threshold)[0]

    if len(true_large_idx) == 0:
        print("⚠️ Warning: No large moves in ground truth.")
        return None

    # Subset for large moves
    y_true_large = y_true[true_large_idx]
    y_pred_large = y_pred[true_large_idx]

    # Directional accuracy
    directional_acc = np.mean(np.sign(y_true_large) == np.sign(y_pred_large))

    # Relative MAE (normalized error on large true values)
    abs_true = np.abs(y_true_large)
    relative_mae = np.mean(np.abs(y_true_large - y_pred_large) / abs_true)

    # Turn relative MAE into a "score" where higher = better
    error_component = 1 / (1 + relative_mae)

    # Frequency alignment
    true_freq = len(true_large_idx) / len(y_true)
    pred_freq = len(pred_large_idx) / len(y_pred)
    freq_ratio = pred_freq / true_freq if true_freq > 0 else 0
    freq_component = 1 - abs(1 - freq_ratio) if freq_ratio <= 2 else 1 / freq_ratio

    # F1 score (did model predict large moves well as a classification task?)
    y_true_bin = np.zeros_like(y_true)
    y_true_bin[true_large_idx] = 1
    y_pred_bin = np.zeros_like(y_pred)
    y_pred_bin[pred_large_idx] = 1
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Final combined score
    combined_score = (
        prediction_weight * directional_acc +
        freq_weight * freq_component +
        error_weight * error_component
    ) * (f1_weight * f1)

    return {
        'directional_accuracy': directional_acc,
        'relative_mae': relative_mae,
        'error_score': error_component,
        'frequency_ratio': freq_ratio,
        'frequency_score': freq_component,
        'f1_score': f1,
        'combined_score': combined_score,
        'true_large_count': len(true_large_idx),
        'pred_large_count': len(pred_large_idx)
    }


def test_trailing_strategy(token_predictions, tokens_tx_data, hold_time=60, buy_size=0.1, min_time_between_buys=60, verbose=True):
    """
    Simulates a trading strategy based on predicted future returns.
    Assumes a prediction of 0.0 = no change, 1.0 = 100% gain, -0.5 = 50% loss, etc.
    """
    dynamic_sell_thresholds = [0.75, 0.875]  # e.g., take partial profits as % to full target
    trailing_stop_targets = [0.5, 0.75]      # trailing stop trigger drawdowns from peak
    buy_fee_pct = 0.01
    sell_fee_pct = 0.01
    slippage_pct = 0.005

    total_profit = 0
    num_trades = 0
    num_wins = 0
    num_losses = 0
    profits = []

    for token_address, bucket_pred_map in token_predictions.items():
        price_df = tokens_tx_data[token_address]
        price_df = price_df.groupby(price_df.index).mean().sort_index()

        if verbose:
            print(f"\nSimulating for token: {token_address}")
        
        last_buy_time = None

        for pred_data in bucket_pred_map:
            bucket_time = pred_data['bucket_time']
            pred = pred_data['prediction']

            if pred <= 0.0:
                continue  # Only trade if prediction is for gain

            start_time, end_time = bucket_time

            if last_buy_time and (end_time - last_buy_time) < min_time_between_buys:
                continue

            if end_time not in price_df.index:
                closest_idx = price_df.index.get_indexer([end_time], method='nearest')[0]
                closest_time = price_df.index[closest_idx]
            else:
                closest_time = end_time

            raw_buy_price = price_df.loc[closest_time, 'price']
            if pd.isna(raw_buy_price):
                continue

            buy_price = raw_buy_price * (1 + slippage_pct)
            entry_value = buy_size * (1 - buy_fee_pct)
            tokens_bought = entry_value / buy_price
            remaining_tokens = tokens_bought
            profit = -buy_size
            peak_price = buy_price
            last_buy_time = end_time

            if verbose:
                print(f"  Prediction: {pred:+.2%} | Buy Price: {buy_price:.6e} (with slippage/fees)")

            future_prices = price_df.loc[closest_time:]
            if len(future_prices) < 2:
                continue

            predicted_target_price = buy_price * (1 + pred)
            sell_targets_prices = [
                buy_price + (predicted_target_price - buy_price) * t 
                for t in dynamic_sell_thresholds
            ]

            target_idx = 0
            stop_idx = 0

            for time, row in future_prices.iterrows():
                raw_price = row['price']
                peak_price = max(peak_price, raw_price)
                sell_price = raw_price * (1 - slippage_pct) * (1 - sell_fee_pct)

                if time - end_time > hold_time:
                    profit += remaining_tokens * sell_price
                    if verbose:
                        print(f"    Hold time expired, sold all | Price: {raw_price:.6e}")
                    remaining_tokens = 0
                    break

                if raw_price >= predicted_target_price:
                    profit += remaining_tokens * sell_price
                    if verbose:
                        print(f"    Target hit, sold all | Price: {raw_price:.6e}")
                    remaining_tokens = 0
                    break

                if target_idx < len(sell_targets_prices) and raw_price >= sell_targets_prices[target_idx]:
                    sell_amount = remaining_tokens * 0.5
                    profit += sell_amount * sell_price
                    remaining_tokens -= sell_amount
                    if verbose:
                        print(f"    Partial sell at {dynamic_sell_thresholds[target_idx]*100:.1f}% | Price: {raw_price:.6e}")
                    target_idx += 1

                drawdown = 1 - (raw_price / peak_price)
                if stop_idx < len(trailing_stop_targets) and drawdown >= trailing_stop_targets[stop_idx]:
                    sell_amount = remaining_tokens * 0.5
                    profit += sell_amount * sell_price
                    remaining_tokens -= sell_amount
                    if verbose:
                        print(f"    Trailing stop hit ({trailing_stop_targets[stop_idx]*100:.1f}%) | Price: {raw_price:.6e}")
                    stop_idx += 1

                if remaining_tokens <= 0:
                    if verbose:
                        print(f"    All tokens sold.")
                    break

            if verbose:
                print(f"Profit: {profit:.4f} SOL")
            num_trades += 1
            total_profit += profit
            profits.append(profit)
            if profit > 0:
                num_wins += 1
            else:
                num_losses += 1

    return {
        "num_trades": num_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "total_profit": total_profit
    }


def test_base_strategy(token_predictions, tokens_tx_data, buy_size=0.1, min_time_between_buys=30, verbose=True):
    """
    Simulates a trading strategy based on predicted future returns.
    Assumes a prediction of 0.0 = no change, 1.0 = 100% gain, -0.5 = 50% loss, etc.
    """
    buy_fee_pct = 0.01
    sell_fee_pct = 0.01
    slippage_pct = 0.005

    total_profit = 0
    num_trades = 0
    num_wins = 0
    num_losses = 0
    profits = []
    for token_address, bucket_pred_map in token_predictions.items():
        price_df = tokens_tx_data[token_address]
        price_df = price_df.groupby(price_df.index).mean().sort_index()

        if verbose:
            print(f"\nSimulating for token: {token_address}")
        
        last_buy_time = None

        for pred_data in bucket_pred_map:
            bucket_time = pred_data['bucket_time']
            pred = pred_data['prediction']

            if pred <= 0.0:
                continue  # Only trade if prediction is for gain

            start_time, end_time = bucket_time

            bucket_length_secs = end_time - start_time

            if last_buy_time and (end_time - last_buy_time) < min_time_between_buys:
                continue

            if end_time not in price_df.index:
                closest_idx = price_df.index.get_indexer([end_time], method='nearest')[0]
                closest_time = price_df.index[closest_idx]
            else:
                closest_time = end_time

            raw_buy_price = price_df.loc[closest_time, 'price']
            if pd.isna(raw_buy_price):
                continue

            buy_price = raw_buy_price * (1 + slippage_pct)
            entry_value = buy_size * (1 - buy_fee_pct)
            tokens_bought = entry_value / buy_price
            remaining_tokens = tokens_bought
            profit = -buy_size
            peak_price = buy_price
            last_buy_time = end_time

            if verbose:
                print(f"  Prediction: {pred:+.2%} | Buy Price: {buy_price:.6e} (with slippage/fees)")

            future_prices = price_df.loc[closest_time:]
            if len(future_prices) < 2:
                continue

            predicted_target_price = buy_price * (1 + pred)

            for time, row in future_prices.iterrows():
                raw_price = row['price']
                peak_price = max(peak_price, raw_price)
                sell_price = raw_price * (1 - slippage_pct) * (1 - sell_fee_pct)

                if time - end_time > bucket_length_secs:
                    profit += remaining_tokens * sell_price
                    if verbose:
                        print(f"    Hold time expired, sold all | Price: {raw_price:.6e}")
                    remaining_tokens = 0
                    break

                elif raw_price >= predicted_target_price:
                    profit += remaining_tokens * sell_price
                    if verbose:
                        print(f"    Target hit, sold all | Price: {raw_price:.6e}")
                    remaining_tokens = 0
                    break

            if verbose:
                print(f"Profit: {profit:.4f} SOL")
            num_trades += 1
            total_profit += profit
            profits.append(profit)
            if profit > 0:
                num_wins += 1
            else:
                num_losses += 1

    return {
        "num_trades": num_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "total_profit": total_profit
    }


def print_strategy_results(results):
    # Summary
    num_trades = results['num_trades']
    num_wins = results['num_wins']
    num_losses = results['num_losses']
    total_profit = results['total_profit']

    print("\n--- Strategy Summary ---")
    print(f"Total Trades: {num_trades}")
    print(f"Wins: {num_wins} | Losses: {num_losses}")
    print(f"Win Rate: {(num_wins / num_trades * 100):.2f}%" if num_trades > 0 else "No trades made")
    print(f"Total Profit: {total_profit:.4f} SOL")
    if num_trades > 0:
        print(f"Average Profit per Trade: {(total_profit / num_trades):.4f} SOL")


def get_model_comparison(model_results, metric_func=evaluate_large_move_model, threshold=1):
    model_comparison = []
    
    for model_name, results in model_results.items():
        y_pred = results["pred"]
        y_real = results["real"]
       
        # Compute the large move metrics
        metrics = metric_func(y_real, y_pred, threshold=threshold)
       
        if metrics is not None:
            metrics["model_name"] = model_name
            model_comparison.append(metrics)
    
    return model_comparison

def pair_predictions_with_input_sequence(X_test, y_pred, y_test, min_pred_size=1):
    pred_large_idx = np.where(np.abs(y_pred) >= min_pred_size)[0]

    y_pred_large = y_pred[pred_large_idx]
    y_real_equivalent = y_test[pred_large_idx]
    X_test_equivalent = X_test[pred_large_idx]


    return X_test_equivalent, y_pred_large, y_real_equivalent
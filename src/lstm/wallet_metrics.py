import pandas as pd
import numpy as np
from collections import defaultdict


def generate_cumulative_wallet_metrics(wallet_csv):
    # Load your CSV
    df = pd.read_csv(wallet_csv)

    # Metrics DataFrame will be joined with original tx_sig for lookup
    metric_df = pd.DataFrame()
    metric_df["tx_sig"] = df["tx_sig"]  # <- Include this for lookups
    metric_df["signer"] = df["signer"]
    df["volume_this_tx"] = np.abs(df["bc_sol_after"] - df["bc_sol_before"])
    df["sol_change"] = df["bc_sol_after"] - df["bc_sol_before"]

    metric_df["block_time"] = df["block_time"]
    metric_df["volume_prior"] = df["volume_this_tx"].cumsum().shift(1).fillna(0)
    metric_df["trade_count_prior"] = pd.Series(range(len(df)))

    avg_trade_prior = df["volume_this_tx"].expanding().mean().shift(1)
    metric_df["trade_size_deviation"] = ((df["volume_this_tx"] - avg_trade_prior) / avg_trade_prior).fillna(0)

    pnl_stats = get_cumulative_realised_pnl(df)

    metric_df["rough_pnl"] = pd.Series(pnl_stats["cumulative_pnl"]).shift(1).fillna(0)
    metric_df["average_roi"] = pd.Series(pnl_stats["average_rois"]).shift(1).fillna(0)
    metric_df["win_rate"] = pd.Series(pnl_stats["win_rates"]).shift(1).fillna(0)
    metric_df["average_hold_duration"] = pd.Series(pnl_stats["average_hold_durations"]).shift(1).fillna(0)

    return metric_df

def get_cumulative_realised_pnl(transactions_df):
    pnls = []
    total_pnl = 0
    average_rois = []
    wins = 0
    win_rates = []
    running_hold_durations = []
    
    token_positions = defaultdict(lambda: {
        'tokens_held': 0,
        'token_amortized_buy_price': None,
        'last_buy_time': None
    })

    hold_time_sum = 0
    hold_time_count = 0

    for trade_num, tx in transactions_df.iterrows():
        token = tx["token_address"]
        is_buy = tx["bc_sol_after"] > tx["bc_sol_before"]
        token_price = tx["token_price"]
        block_time = tx["block_time"]

        # Use last average ROI if it exists
        average_roi = average_rois[-1] if average_rois else 0.0

        if is_buy:
            # User gained tokens
            new_tokens = tx["bc_spl_before"] - tx["bc_spl_after"]
            if new_tokens == 0:
                pnls.append(total_pnl)
                win_rates.append(wins / (trade_num + 1))
                average_rois.append(average_roi)
                running_hold_durations.append(
                    hold_time_sum / hold_time_count if hold_time_count > 0 else 0
                )
                continue

            held = token_positions[token]["tokens_held"]
            current_price = token_positions[token]["token_amortized_buy_price"]

            if current_price is None:
                amortized_price = token_price
            else:
                amortized_price = (
                    (held * current_price + new_tokens * token_price) /
                    (held + new_tokens)
                )

            token_positions[token]["token_amortized_buy_price"] = amortized_price
            token_positions[token]["tokens_held"] += new_tokens
            token_positions[token]["last_buy_time"] = block_time

        elif not is_buy and token_positions[token]["tokens_held"] > 0:
            # User sold tokens
            tokens_sold = tx["bc_spl_after"] - tx["bc_spl_before"]
            held = token_positions[token]["tokens_held"]
            simulated_tokens_sold = min(held, tokens_sold)

            cost_basis = simulated_tokens_sold * token_positions[token]["token_amortized_buy_price"]
            sale_value = simulated_tokens_sold * token_price
            realised_profit = sale_value - cost_basis

            token_positions[token]["tokens_held"] -= simulated_tokens_sold
            total_pnl += realised_profit

            if realised_profit > 0:
                wins += 1

            # ROI calc
            roi = (sale_value - cost_basis) / cost_basis if cost_basis > 0 else 0
            average_roi = (average_roi * trade_num + roi) / (trade_num + 1)

            # Hold duration
            buy_time = token_positions[token]["last_buy_time"]
            if buy_time:
                hold_duration = block_time - buy_time
                hold_time_sum += hold_duration
                hold_time_count += 1

        win_rate = wins / (trade_num + 1)
        win_rates.append(win_rate)
        average_rois.append(average_roi)

        avg_hold_duration = hold_time_sum / hold_time_count if hold_time_count > 0 else 0
        running_hold_durations.append(avg_hold_duration)

        pnls.append(total_pnl)
    
    return {
        "cumulative_pnl": pnls,
        "average_rois": average_rois,
        "win_rates": win_rates,
        "average_hold_durations": running_hold_durations
    }
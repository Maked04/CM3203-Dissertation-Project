import os
import json
import numpy as np
import sqlite3
import pandas as pd

from src.loader import get_data_dir, load_data_file
from feature_extraction import get_token_features_and_metadata
from time_buckets import get_time_buckets
from sliding_tx_windows import get_sliding_windows
from wallet_metrics import generate_cumulative_wallet_metrics


def pre_process_time_bucket_data(time_bucket_config, tokens_file="cluster_2_tokens.txt"):
    # Create train test data folder if it doesnt exist
    base_dir = os.path.join(get_data_dir(), "time_bucket_data")
    os.makedirs(base_dir, exist_ok=True)

    # Name of new directory is time_bucket_num where num increases per folder
    folder_name = f"time_bucket_{len(os.listdir(base_dir)) + 1}"
    time_bucket_dir = os.path.join(base_dir, folder_name)
    os.makedirs(time_bucket_dir, exist_ok=True)

    config_json = vars(time_bucket_config)

    # Save time bucket config
    with open(os.path.join(time_bucket_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=4)

    # Read cluster 2 tokens
    with load_data_file(os.path.join(get_data_dir(), tokens_file)) as f:
        token_addresses = f.read().splitlines()[:-1]

    # Get features
    for token_address in token_addresses:
        features, timestamps, prices = get_token_features_and_metadata(token_address)

        # Get time buckets
        X, y, bucket_times = get_time_buckets(features, timestamps, prices, time_bucket_config) 
        if len(X) == 0 or len(y) == 0 or len(X) != len(y):  # Make sure we have data
            continue
        
        token_dir = os.path.join(time_bucket_dir, token_address)
        os.makedirs(token_dir, exist_ok=True)
        # Save X, y, bucket_times per token
        np.save(os.path.join(token_dir, "X"), X)
        np.save(os.path.join(token_dir, "y"), y)
        np.save(os.path.join(token_dir, "bucket_times"), bucket_times)


def pre_process_sliding_tx_data(sliding_window_config, tokens_file="cluster_2_tokens.txt"):
    # Create train test data folder if it doesnt exist
    base_dir = os.path.join(get_data_dir(), "sliding_window_data")
    os.makedirs(base_dir, exist_ok=True)

    # Name of new directory is sliding_window_num where num increases per folder
    folder_name = f"sliding_window_{len(os.listdir(base_dir)) + 1}"
    sliding_window_dir = os.path.join(base_dir, folder_name)
    os.makedirs(sliding_window_dir, exist_ok=True)

    config_json = vars(sliding_window_config)

    # Save sliding window config
    with open(os.path.join(sliding_window_dir, "config.json"), "w") as f:
        json.dump(config_json, f, indent=4)

    # Read cluster 2 tokens
    with load_data_file(os.path.join(get_data_dir(), tokens_file)) as f:
        token_addresses = f.read().splitlines()[:-1]

    # Get features
    for token_address in token_addresses:
        features, timestamps, prices = get_token_features_and_metadata(token_address)

        # Get sliding windows
        X, y = get_sliding_windows(features, timestamps, prices, sliding_window_config) 
        if len(X) == 0 or len(y) == 0 or len(X) != len(y):  # Make sure we have data
            continue
        
        token_dir = os.path.join(sliding_window_dir, token_address)
        os.makedirs(token_dir, exist_ok=True)
        # Save X, y per token
        np.save(os.path.join(token_dir, "X"), X)
        np.save(os.path.join(token_dir, "y"), y)


def insert_wallet_metrics_to_sqlite(metric_df, sqlite_db_path, table_name="metrics"):
    """
    Inserts the wallet metric DataFrame into a SQLite table using tx_sig as the primary key.

    Args:
        metric_df (pd.DataFrame): The metrics DataFrame returned from generate_cumulative_wallet_metrics.
        sqlite_db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to insert into (default: "metrics").
    """
    # Define schema if table doesn't exist
    schema = """
        CREATE TABLE IF NOT EXISTS {table_name} (
            tx_sig TEXT PRIMARY KEY,
            signer TEXT,
            block_time INTEGER,
            volume_prior REAL,
            trade_count_prior INTEGER,
            trade_size_deviation REAL,
            rough_pnl REAL,
            average_roi REAL,
            win_rate REAL,
            average_hold_duration REAL
        )
    """.format(table_name=table_name)

    # Connect and create table if not exists
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute(schema)

    # Insert rows with upsert (replace on conflict)
    insert_sql = f"""
        INSERT OR REPLACE INTO {table_name} (
            tx_sig, signer, block_time, volume_prior, trade_count_prior, trade_size_deviation,
            rough_pnl, average_roi, win_rate, average_hold_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows = metric_df[[
        "tx_sig", "signer", "block_time", "volume_prior", "trade_count_prior",
        "trade_size_deviation", "rough_pnl", "average_roi", "win_rate", "average_hold_duration"
    ]].values.tolist()

    cursor.executemany(insert_sql, rows)
    conn.commit()
    conn.close()

def pre_process_wallet_metrics(input_directory, sqlite_db_path):
    files = os.listdir(input_directory)

    all_metrics = []  # We'll concat these later

    total_files = len(files)

    for idx, file in enumerate(files):
        print(f"Processing {idx+1}/{total_files} - {file}")
        metrics = generate_cumulative_wallet_metrics(wallet_csv=os.path.join(input_directory, file))
        all_metrics.append(metrics)


        progress = (idx + 1) / total_files * 100
        print(f"Progress: {progress:.2f}%")

    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    insert_wallet_metrics_to_sqlite(combined_df, sqlite_db_path)
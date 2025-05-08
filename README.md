# CM3203-Dissertation-Project

This repo contains python files used to pre-process Solana Pumpfun data and perform multiple forms of analysis on it including:
- Feature extraction
- Clustering
- LSTM model generation

and code to visualise results produced


To use this code directly, you must create a "data" directory at the same level as the "src" folder and add to it:

- Token data folder containing csv's where each each csv is a token's history
- Wallet data folder containing csv's where each csv is a wallet's history

csv schema must be:

tx_sig,
block_time,
slot,
fee,
token_price,
token_address,
is_creator,
bc_spl_before,
bc_spl_after,
bc_sol_before,
bc_sol_after,
signer_spl_before,
signer_spl_after,
signer_spl_before,
signer_spl_after,
signer_sol_before,
signer_sol_after,


Methods you can then use:

- Convert token data in token feature vectors --> Perform clustering on these tokens to identify legit vs scammy tokens
clusterting/feature_extraction.py
clusterting/db_scan.ipynb
clusterting/k_means.ipynb

- Convert wallet data into cumulative wallet metrics over time --> Stored in sqlite database so a wallet's metric at a given time can be queried
lstm/wallet_metrics.py
lstm/preprocessing.py

- Convert token data in transaction specific feature vectors (Using precomputed wallet metrics stored in sqlite database)
lstm/feature_extraction.py
lstm/preprocessing.py

- Transform transaction specific feature vectors into multiple different (sequence -> price prediction) formats
lstm/time_buckets.py
lstm/sliding_tx_windows.py

- Train an lstm model to learn sequences of transaction specific feature vectors to predict price changes (saves models to "trained_models" folder)
lstm/model_generation/train_model.ipynb
lstm/model_generation/optimise_model.ipynb

- Visualise the performance of the models stored in "trained_models" folder
lstm/model_testing/individual_model_testing.ipynb
lstm/model_testing/compare_models.ipynb
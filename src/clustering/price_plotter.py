from src.loader import load_token_price_data, get_token_stats
from src.token_lifecycle_utils import trim_main_trading_period
import matplotlib.pyplot as plt
import math


def plot_token_price(token_addresses: str | list, title=None, trim_dead_period=False):
    """
    Create price plots for one or more tokens.
   
    Args:
        token_addresses: Single token address (str) or list of token addresses
        title: Optional title for the overall figure
    """
    if isinstance(token_addresses, str):
        token_addresses = [token_addresses]
   
    num_tokens = len(token_addresses)
    if num_tokens == 0:
        print("No tokens provided")
        return
   
    cols = min(2, num_tokens)  # Max 2 columns
    rows = math.ceil(num_tokens / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 6 * rows))

    # Ensure axes is always iterable
    if num_tokens == 1:
        axes = [axes]  # Convert single AxesSubplot to a list
    else:
        axes = axes.flatten()  # Flatten for consistent iteration
   
    for idx, (ax, token_address) in enumerate(zip(axes, token_addresses)):
        price_data = load_token_price_data(token_address)

        if trim_dead_period:
            price_data = trim_main_trading_period(price_data, min_trades_per_hour=100, window_size_hours=0.5)
       
        if price_data is None:
            ax.text(0.5, 0.5, f"No data available for\n{token_address}",
                    ha='center', va='center')
            continue
       
        stats = get_token_stats(price_data)
        ax.plot(price_data.index, price_data['price'], linewidth=2)
        ax.set_title(f'Token Price Over Time\n{token_address}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (SOL)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

        stats_text = (
            f"Initial Price: {stats['initial_price']:.8f}\n"
            f"Final Price: {stats['final_price']:.8f}\n"
            f"Change: {stats['price_change_pct']:.2f}%"
        )
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
   
    for idx in range(len(token_addresses), len(axes)):
        axes[idx].axis('off')
   
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()
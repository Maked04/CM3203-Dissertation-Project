import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.loader import load_token_price_data
import plotly.graph_objects as go


def plot_error_distribution(pred, real, bins=30, title='Error Distribution'):
    pred = np.array(pred)
    real = np.array(real)
    
    errors = real - pred
    mean_error = np.mean(errors)

    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Error')
    plt.axvline(x=mean_error, color='green', linestyle='-', linewidth=1.5, label=f'Mean Error: {mean_error:.4f}')

    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add error stats box
    error_stats = f'Mean: {mean_error:.4f}\n' \
                  f'Std Dev: {np.std(errors):.4f}\n' \
                  f'Median: {np.median(errors):.4f}\n' \
                  f'Min: {np.min(errors):.4f}\n' \
                  f'Max: {np.max(errors):.4f}'
    plt.text(0.95, 0.95, error_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_distribution(values, bins=30, name="", color='lightgreen'):
    values = np.array(values)

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, alpha=0.7, color=color, edgecolor='black')
    mean_pred = np.mean(values)
    plt.axvline(x=mean_pred, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean_pred:.4f}')
    plt.xlabel(f'{name}Values')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {name} Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for predictions
    pred_stats = f'Mean: {mean_pred:.4f}\n' \
                 f'Std Dev: {np.std(values):.4f}\n' \
                 f'Median: {np.median(values):.4f}\n' \
                 f'Min: {np.min(values):.4f}\n' \
                 f'Max: {np.max(values):.4f}'
    plt.text(0.95, 0.95, pred_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def plot_value_distributions(pred, real, bins=30):
    pred = np.array(pred)
    real = np.array(real)

    # -------- Plot 1: Distribution of Predicted Values --------
    plt.figure(figsize=(10, 5))
    plt.hist(pred, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    mean_pred = np.mean(pred)
    plt.axvline(x=mean_pred, color='blue', linestyle='-', linewidth=1.5, label=f'Mean: {mean_pred:.4f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for predictions
    pred_stats = f'Mean: {mean_pred:.4f}\n' \
                 f'Std Dev: {np.std(pred):.4f}\n' \
                 f'Median: {np.median(pred):.4f}\n' \
                 f'Min: {np.min(pred):.4f}\n' \
                 f'Max: {np.max(pred):.4f}'
    plt.text(0.95, 0.95, pred_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # -------- Plot 2: Distribution of Real Values --------
    plt.figure(figsize=(10, 5))
    plt.hist(real, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    mean_real = np.mean(real)
    plt.axvline(x=mean_real, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_real:.4f}')
    plt.xlabel('Real Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Real Values')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add stats for actuals
    real_stats = f'Mean: {mean_real:.4f}\n' \
                 f'Std Dev: {np.std(real):.4f}\n' \
                 f'Median: {np.median(real):.4f}\n' \
                 f'Min: {np.min(real):.4f}\n' \
                 f'Max: {np.max(real):.4f}'
    plt.text(0.95, 0.95, real_stats, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_directional_accuracy(pred, real, bins=None):
    if bins is None:
        bins = [-np.inf, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, np.inf]  # Default bins including sign
    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
   
    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    # Assign Predicted Values to Bins
    binned_data = pd.cut(pred, bins, labels=bin_labels)
    
    # Compute Directional Accuracy per Bin
    bin_accuracy = {}
    bin_counts = {}
    for bin_label in bin_labels:
        indices = (binned_data == bin_label)
        bin_counts[bin_label] = indices.sum()
        
        if indices.sum() > 0:  # Avoid division by zero
            correct_signs = np.sum(np.sign(pred[indices]) == np.sign(real[indices]))
            accuracy = correct_signs / indices.sum()
            bin_accuracy[bin_label] = accuracy
        else:
            bin_accuracy[bin_label] = None  # No data points in this bin
    
    # Remove empty bins
    bin_accuracy = {k: v for k, v in bin_accuracy.items() if v is not None}
    bin_counts = {k: v for k, v in bin_counts.items() if v > 0}
    
    # Prepare data for plotting
    labels = list(bin_accuracy.keys())
    accuracies = list(bin_accuracy.values())
    counts = [bin_counts[label] for label in labels]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel("Predicted Value Bins")
    plt.ylabel("Directional Accuracy")
    plt.title("Directional Accuracy per Bin (with Prediction Counts)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'n={count}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return bin_accuracy, bin_counts


def plot_magnitude_accuracy(pred, real, bins=None, use_percentage_error=False):
    print("here")
    if bins is None:
        bins = [-np.inf, -1, -0.5, -0.1, 0.1, 0.5, 1, np.inf]  # Exclude zero to avoid division issues
    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
   
    # Filter out values where real is exactly 0 if using percentage error
    if use_percentage_error:
        nonzero_mask = real != 0  # Change to filter based on real values
        pred = pred[nonzero_mask]
        real = real[nonzero_mask]
    
    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    binned_data = pd.cut(pred, bins, labels=bin_labels)
    bin_errors = {}
    bin_counts = {}
    for bin_label in bin_labels:
        indices = (binned_data == bin_label)
        bin_counts[bin_label] = indices.sum()
        if indices.sum() > 0:
            if use_percentage_error:
                # Correct formula: |real - pred| / |real| * 100
                errors = np.abs((real[indices] - pred[indices]) / real[indices]) * 100  # Multiply by 100 for percentage
            else:
                errors = np.abs(real[indices] - pred[indices])
            avg_error = np.mean(errors)
            bin_errors[bin_label] = avg_error
        else:
            bin_errors[bin_label] = None
    
    # Remove empty bins
    bin_errors = {k: v for k, v in bin_errors.items() if v is not None}
    bin_counts = {k: v for k, v in bin_counts.items() if v > 0}
    labels = list(bin_errors.keys())
    avg_errors = list(bin_errors.values())
    counts = [bin_counts[label] for label in labels]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_errors, color='salmon')
    plt.xlabel("Predicted Value Bins")
    ylabel = "Average Percentage Error (%)" if use_percentage_error else "Average Absolute Error"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Bin (with Prediction Counts)")
    plt.xticks(rotation=45, ha='right')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'n={count}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    return bin_errors, bin_counts

def plot_magnitude_accuracy_old(pred, real, bins=None, use_percentage_error=False):
    if bins is None:
        bins = [-np.inf, -1, -0.5, -0.1, 0.1, 0.5, 1, np.inf]  # Exclude zero to avoid division issues

    pred = np.array(pred).flatten()
    real = np.array(real).flatten()
    
    # Filter out values where prediction is exactly 0 if using percentage error
    if use_percentage_error:
        nonzero_mask = pred != 0
        pred = pred[nonzero_mask]
        real = real[nonzero_mask]

    bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    binned_data = pd.cut(pred, bins, labels=bin_labels)

    bin_errors = {}
    bin_counts = {}

    for bin_label in bin_labels:
        indices = (binned_data == bin_label)
        bin_counts[bin_label] = indices.sum()

        if indices.sum() > 0:
            if use_percentage_error:
                errors = np.abs((real[indices] - pred[indices]) / pred[indices])
            else:
                errors = np.abs(real[indices] - pred[indices])
            avg_error = np.mean(errors)
            bin_errors[bin_label] = avg_error
        else:
            bin_errors[bin_label] = None

    # Remove empty bins
    bin_errors = {k: v for k, v in bin_errors.items() if v is not None}
    bin_counts = {k: v for k, v in bin_counts.items() if v > 0}

    labels = list(bin_errors.keys())
    avg_errors = list(bin_errors.values())
    counts = [bin_counts[label] for label in labels]

    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_errors, color='salmon')
    plt.xlabel("Predicted Value Bins")
    ylabel = "Average Percentage Error" if use_percentage_error else "Average Absolute Error"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Bin (with Prediction Counts)")
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return bin_errors, bin_counts


def plot_prediction_vs_actual(pred, real):
    plt.figure(figsize=(10, 6))
    plt.scatter(real, pred, alpha=0.5)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.show()


def plot_predictions_on_price_graph(token_address, bucket_pred_map, min_abs_pred_size=0.5):

    # Load and process price data
    price_df = load_token_price_data(token_address, use_datetime=False)
    if price_df is None or price_df.empty:
        print("No price data available to plot.")
        return

    # Aggregate duplicate timestamps by averaging
    price_df = price_df.groupby(price_df.index).mean()

    # Base price line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df['price'],
        mode='lines',
        name='Token Price',
        line=dict(color='black')
    ))

    # Add predictions as arrows/markers
    for pred_data in bucket_pred_map:
        # Extract bucket time and prediction from the list of dicts
        bucket_time = pred_data['bucket_time']
        pred = pred_data['prediction']

        if abs(pred) < min_abs_pred_size:
            continue

        start_time, end_time = bucket_time

        # Handle missing times
        if end_time not in price_df.index:
            closest_idx = price_df.index.get_indexer([end_time], method='nearest')[0]
            closest_time = price_df.index[closest_idx]
        else:
            closest_time = end_time

        price_at_end = price_df.loc[closest_time, 'price']
        color = 'green' if pred > 0 else 'red'

        fig.add_trace(go.Scatter(
            x=[closest_time],
            y=[price_at_end],
            mode='markers+text',
            marker=dict(color=color, size=10, symbol='arrow-up' if pred > 0 else 'arrow-down'),
            text=[f'{pred:.2f}x'],
            textposition='top center',
            name='Prediction'
        ))

    # Layout tweaks
    fig.update_layout(
        title=f"Token Price with Predictions: {token_address}",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.show()


def plot_model_comparison(results):
    """
    Plots a comparison of model evaluation metrics.

    Parameters:
    - results: list of dicts containing evaluation metrics for each model.
      Each dict should include 'model_name', 'combined_score', 'directional_accuracy',
      'error_score', 'frequency_score', 'f1_score'.
    """

    model_names = [metrics["model_name"] for metrics in results]
    combined_scores = [metrics['combined_score'] for metrics in results]
    directional_scores = [metrics['directional_accuracy'] for metrics in results]
    error_scores = [metrics['error_score'] for metrics in results]
    freq_scores = [metrics['frequency_score'] for metrics in results]
    f1_scores = [metrics['f1_score'] for metrics in results]

    x = np.arange(len(model_names))
    width = 0.15  # Adjusted width to fit all bars

    # Plot Combined Score
    plt.figure(figsize=(12, 6))
    plt.bar(x, combined_scores, color='skyblue')
    plt.xticks(x, model_names, rotation=45)
    plt.ylabel("Combined Score")
    plt.title("Model Ranking by Combined Score")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    # Plot Sub-metrics including F1 Score
    plt.figure(figsize=(12, 6))
    plt.bar(x - width*1.5, directional_scores, width=width, label='Direction', color='lightgreen')
    plt.bar(x - width*0.5, error_scores, width=width, label='Error', color='lightcoral')
    plt.bar(x + width*0.5, freq_scores, width=width, label='Frequency', color='lightblue')
    plt.bar(x + width*1.5, f1_scores, width=width, label='F1 Score', color='orange')
    plt.xticks(x, model_names, rotation=45)
    plt.ylabel("Sub-metric Scores (0-1)")
    plt.title("Model Sub-metric Breakdown")
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import math
import ipywidgets as widgets
from IPython.display import display, clear_output

class DistributionNavigator:
    def __init__(self, X_test_equivalents, y_predictions, y_real_equivalents, feature_names, bins=30, figsize_per_plot=(5, 3), log_scale=False):
        self.X_test_equivalents = X_test_equivalents
        self.y_predictions = y_predictions
        self.y_real_equivalents = y_real_equivalents
        self.feature_names = feature_names
        self.bins = bins
        self.figsize_per_plot = figsize_per_plot
        self.index = 0
        self.log_scale = log_scale

        # Widgets
        self.index_display = widgets.IntText(value=self.index, description='Index:', disabled=True)
        self.next_button = widgets.Button(description='Next')
        self.back_button = widgets.Button(description='Back')
        self.output = widgets.Output()

        # Bind events
        self.next_button.on_click(self._on_next)
        self.back_button.on_click(self._on_back)

    def _plot(self):
        with self.output:
            clear_output(wait=True)
            data_index = self.index % self.X_test_equivalents.shape[0]
            X_test_equivalents = self.X_test_equivalents[data_index]
            y_pred = self.y_predictions[data_index]
            y_real_equivalent = self.y_real_equivalents[data_index]
            # Use shape to get how many features in each step in x test
            num_features = X_test_equivalents.shape[1]

            cols = math.ceil(math.sqrt(num_features))
            rows = math.ceil(num_features / cols)
            
            figsize = (cols * self.figsize_per_plot[0], rows * self.figsize_per_plot[1])
            fig, axs = plt.subplots(rows, cols, figsize=figsize)
            plt.suptitle(f"Feature distributions for pred: {y_pred}, real: {y_real_equivalent}", fontsize=16)
            axs = np.array(axs).reshape(-1)  # Flatten grid of axes to 1D

            for i in range(num_features):
                feature_values = X_test_equivalents[:, i]
                # Plot the histogram
                axs[i].hist(feature_values, bins=self.bins, color='skyblue', edgecolor='black')
                feature_name = self.feature_names[i]
                axs[i].set_title(f'Distribution of {feature_name}')
                
                # Apply log scale if specified
                if self.log_scale:
                    axs[i].set_yscale('log')  # Apply log scale on y-axis
                
            plt.show()

    def _on_next(self, _):
        self.index += 1
        self.index_display.value = self.index % self.X_test_equivalents.shape[0]
        self._plot()

    def _on_back(self, _):
        self.index -= 1
        self.index_display.value = self.index % self.X_test_equivalents.shape[0]
        self._plot()

    def show(self):
        """
        Displays the interactive plot navigator.
        """
        self._plot()
        controls = widgets.HBox([self.back_button, self.next_button, self.index_display])
        display(controls, self.output)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_large_move_evaluation(y_true, y_pred, threshold=0.5):
    """
    Evaluates how well predictions capture large market moves and visualizes the results.
    
    Parameters:
    - y_true: array-like of true returns
    - y_pred: array-like of predicted returns
    - threshold: minimum % move considered 'large' (default 2%)
    
    Returns:
    - None (displays plots)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Identify large moves
    true_large_idx = np.where(np.abs(y_true) >= threshold)[0]
    pred_large_idx = np.where(np.abs(y_pred) >= threshold)[0]
    
    if len(true_large_idx) == 0:
        print("⚠️ Warning: No large moves in ground truth.")
        return None
    
    # Binary classification for large moves
    y_true_bin = np.zeros_like(y_true)
    y_true_bin[true_large_idx] = 1
    y_pred_bin = np.zeros_like(y_pred)
    y_pred_bin[pred_large_idx] = 1
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Large Move'],
                yticklabels=['Normal', 'Large Move'],
                ax=ax1)
    ax1.set_title('Confusion Matrix for Large Moves')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot 2: Precision, Recall, F1 Score
    metrics = [precision, recall, f1]
    ax2.bar(['Precision', 'Recall', 'F1 Score'], metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylim([0, 1])
    ax2.set_title('Precision, Recall, and F1 Score')
    
    # Add values on top of bars
    for i, v in enumerate(metrics):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Add a text box with summary statistics
    textbox = (
        f"True Large Moves: {len(true_large_idx)}\n"
        f"Predicted Large Moves: {len(pred_large_idx)}\n"
        f"Precision: {precision:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"F1 Score: {f1:.3f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(1.05, 0.5, textbox, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    return fig
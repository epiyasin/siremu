import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import attach_identifier, distinct_rates
from dataset import MintDataset
import matplotlib.lines as mlines
from matplotlib.widgets import CheckButtons

def plot_comparison(predictions, actual, ABM_data, settings):
    predictions_np = attach_identifier(predictions)
    actual_np = attach_identifier(actual)

    num_plots = settings["plotting"]["num_plots"]

    # Get distinct indices based on rates
    indices = distinct_rates(ABM_data, num_plots)

    selected_infection_rates = [ABM_data[i]['infection_rate'] for i in indices]
    selected_recovery_rates = [ABM_data[i]['recovery_rate'] for i in indices]

    fig, axes = plt.subplots(int(num_plots**0.5), int(num_plots**0.5), figsize=settings["plotting"]["figure_size_comparison"])

    handles = [
        plt.Line2D([0], [0], color='grey', label='Actual'),
        plt.Line2D([0], [0], linestyle='--', color='black', label='Predicted'),
        plt.Line2D([0], [0], linestyle='--', color='red', label='Average Actual')
    ]

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        infection_rate = selected_infection_rates[i]
        recovery_rate = selected_recovery_rates[i]

        matched_indices = [j for j, d in enumerate(ABM_data) if d['infection_rate'] == infection_rate and d['recovery_rate'] == recovery_rate]

        actual_means = np.mean([actual_np[j, :-1] for j in matched_indices], axis=0)

        for m_idx in matched_indices:
            ax.plot(actual_np[m_idx, :-1], color='grey')

        ax.plot(predictions_np[idx, :-1], label='Predicted', linestyle='--', color='black')
        ax.plot(actual_means, linestyle='--', color='red')

        max_val = np.max([actual_means.max(), predictions_np[idx, :-1].max()])
        rounded_max = int(np.ceil(max_val / 10.0)) * 10
        ax.set_ylim(0, rounded_max)

        ax.set_title(f'Epidemics for Inf. Rate {infection_rate:.3f} & Rec. Rate {recovery_rate:.3f}', fontsize = 9)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Incidence (factor: 1000)')
        ax.legend(handles=handles)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()

def plot_emulation(predicted_output, settings):
    plt.figure(figsize=settings["plotting"]["figure_size_emulation"])
    plt.plot(predicted_output[0].numpy(), label='Predicted Incidence')
    plt.title('Emulated Epidemic')
    plt.xlabel('Time Step')
    plt.ylabel('New Infections')
    plt.legend()
    plt.show()

def plot_mint_compare(predictions, actual, settings, num_samples=9):
    file_path = settings["MINT"]["data"].get('data_path')
    emulator_data = MintDataset(input_file=file_path)

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)

    # Create a 3x3 grid of scatter plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Scatterplots of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.scatter(selected_actual, selected_predictions, c='black', alpha=0.5, marker='o', label='Predicted')
        ax.scatter(selected_actual, selected_actual, c='red', alpha=0.5, marker='o', label='Actual')
        ax.plot([np.min(selected_actual), np.max(selected_actual)], [np.min(selected_actual), np.max(selected_actual)], c='gray', linestyle='--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Scatterplot #{row_index}')
        ax.legend()

    plt.show()


def plot_mint_time_series(predictions, actual, settings, num_samples=9):
    file_path = settings["MINT"]["data"].get('data_path')
    emulator_data = MintDataset(input_file=file_path)

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Select random row indices
    row_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)

    # Create a 3x3 grid of time series plots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Time series of actual vs predicted values for selected time series')

    for i, row_index in enumerate(row_indices):
        # Get the selected row from the predictions and actual arrays
        selected_predictions = flattened_predictions[row_index]
        selected_actual = flattened_actual[row_index]

        ax = axs[i//3, i%3]
        ax.plot(selected_predictions, c='black', label='Predictions', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.plot(selected_actual, c='red', label='Actual', alpha=0.5, linestyle='-', marker='o', markersize=4)
        ax.set_xlabel('Year')
        ax.set_ylabel('Values')
        ax.set_title(f'Time series #{row_index}')

        # Adjust the x-axis tick labels
        tick_positions = np.arange(0, len(selected_predictions), 12)
        tick_labels = [2017 + i for i in tick_positions // 12]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.legend()

    plt.show()
    
def plot_mint_avg_compare(predictions, actual, settings):
    file_path = settings["MINT"]["data"].get('data_path')
    emulator_data = MintDataset(input_file=file_path)

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Compute the average predicted and actual values
    avg_predicted = np.mean(flattened_predictions, axis=0)
    avg_actual = np.mean(flattened_actual, axis=0)

    # Create a scatter plot
    plt.figure(figsize=(7, 7))
    plt.scatter(avg_actual, avg_predicted, c='black', alpha=0.5, marker='o', label='Predicted')
    plt.scatter(avg_actual, avg_actual, c='red', alpha=0.5, marker='o', label='Actual')
    plt.plot([np.min(avg_actual), np.max(avg_actual)], [np.min(avg_actual), np.max(avg_actual)], c='gray', linestyle='--')
    plt.xlabel('Average Actual')
    plt.ylabel('Average Predicted')
    plt.title('Scatterplot of Average Actual vs Average Predicted')
    plt.legend()

    plt.show()


def plot_mint_avg_time_series(predictions, actual, settings):
    file_path = settings["MINT"]["data"].get('data_path')
    emulator_data = MintDataset(input_file=file_path)

    # Flatten the predictions and actual values into one-dimensional arrays
    flattened_predictions = np.concatenate(predictions, axis=0)
    flattened_actual = np.concatenate(actual, axis=0)

    # Compute the average predicted and actual values
    avg_predicted = np.mean(flattened_predictions, axis=0)
    avg_actual = np.mean(flattened_actual, axis=0)

    # Create a time series plot
    fig, ax = plt.subplots(figsize=(10, 7))

    lines = []

    # Plot the random trajectories and store them in a list
    random_lines = []

    num_samples = 100
    random_indices = np.random.choice(flattened_predictions.shape[0], size=num_samples, replace=False)
    for i in random_indices:
        random_lines.append(ax.plot(flattened_predictions[i], c='lightgray', alpha=0.5, linestyle='-', marker='^', markersize=2, visible=False)[0])
        random_lines.append(ax.plot(flattened_actual[i], c='darkgray', alpha=0.5, linestyle='--', marker='o', markersize=2, visible=False)[0])

    # Plot the average predicted and actual values
    lines.append(ax.plot(avg_predicted, c='black', label='Average Predictions', alpha=0.7, linestyle='-', marker='^', markersize=4)[0])
    lines.append(ax.plot(avg_actual, c='red', label='Average Actual', alpha=0.7, linestyle='-', marker='o', markersize=4)[0])

    # Adjust the x-axis tick labels and limits
    tick_positions = np.arange(0, len(avg_predicted), 12)
    tick_labels = [2017 + i for i in tick_positions // 12]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Values')
    ax.set_title('Time Series of Average Actual vs Average Predicted')

    # Create custom legend handles
    red_line = mlines.Line2D([], [], color='red', marker='o', linestyle='-', markersize=4, label='Average Actual')
    black_line = mlines.Line2D([], [], color='black', marker='^', linestyle='-', markersize=4, label='Average Predictions')
    ax.legend(handles=[red_line, black_line])

    # Create a CheckButtons widget to toggle the random trajectories
    rax = plt.axes([0.05, 0.9, 0.1, 0.08], frameon=True) # Changed second and fourth parameters
    check = CheckButtons(rax, ['Extend'], [False]) 

    def func(label):
        for line in random_lines:
            line.set_visible(not line.get_visible())
        
        # Check if the random lines are visible or not
        random_lines_visible = random_lines[0].get_visible()

        # Update the legend accordingly
        if random_lines_visible:
            # Add new legend handles for the extended trajectories
            light_gray_line = mlines.Line2D([], [], color='lightgray', marker='^', linestyle='-', markersize=4, label='Extended Predicted')
            dark_gray_line = mlines.Line2D([], [], color='darkgray', marker='o', linestyle='--', markersize=4, label='Extended Actual')
            ax.legend(handles=[red_line, black_line, light_gray_line, dark_gray_line])
        else:
            # Remove the additional legend handles
            ax.legend(handles=[red_line, black_line])
        
        plt.draw()

    check.on_clicked(func)

    plt.show()
    
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
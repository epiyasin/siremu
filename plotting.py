import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import attach_identifier, distinct_rates

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

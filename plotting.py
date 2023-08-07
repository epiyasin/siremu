import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_comparison(predictions, actual, ABM_data, settings):
    # Convert predictions and actual to numpy for easier handling
    predictions_np = np.concatenate(predictions, axis=0)
    actual_np = np.concatenate(actual, axis=0)

    # Select 9 random epidemics for plotting
    num_plots = settings["plotting"]["num_plots"]
    indices = np.random.choice(range(predictions_np.shape[0]), size=num_plots, replace=False)

    # Extract infection and recovery rates for the selected epidemics
    selected_infection_rates = [ABM_data[i]['infection_rate'] for i in indices]
    selected_recovery_rates = [ABM_data[i]['recovery_rate'] for i in indices]

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(int(settings["plotting"]["num_plots"]**0.5),
                             int(settings["plotting"]["num_plots"]**0.5),
                             figsize=settings["plotting"]["figure_size_comparison"])

    handles = [
        plt.Line2D([0], [0], color='grey', label='Actual'),
        plt.Line2D([0], [0], linestyle='--', color='black', label='Predicted'),
        plt.Line2D([0], [0], linestyle='--', color='red', label='Average Actual')
    ]

    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        infection_rate = selected_infection_rates[i]
        recovery_rate = selected_recovery_rates[i]

        # Find all actual epidemics with the same infection and recovery rate
        matched_indices = [j for j, d in enumerate(ABM_data) if d['infection_rate'] == infection_rate and d['recovery_rate'] == recovery_rate]
        
        # Compute the average of the actual epidemics
        actual_means = np.mean([actual_np[j] for j in matched_indices], axis=0)

        # Plot all matched actual epidemics
        for m_idx in matched_indices:
            ax.plot(actual_np[m_idx], color='grey')
            
        # Plot predicted epidemic
        ax.plot(predictions_np[idx], label='Predicted', linestyle='--', color='black')

        # Plot the average of actual epidemics
        ax.plot(actual_means, linestyle='--', color='red')  # This line plots the average

        ax.set_title(f'Epidemics for Inf. Rate {infection_rate:.3f} & Rec. Rate {recovery_rate:.3f}', fontsize = 9)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Incidence (factor: 1000)')
        ax.legend(handles=handles)
        
    # Adjust the layout so the plots do not overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()

def plot_emulation(scenario, model, settings):
    # Use the model to emulate the scenario
    with torch.no_grad():
        predicted_incidence = model(scenario)

    # Plot the predicted incidence
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_incidence[0].numpy(), label='Predicted Incidence')
    plt.title('Emulated Epidemic')
    plt.xlabel('Time Step')
    plt.ylabel('New Infections')
    plt.legend()
    plt.show()

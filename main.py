import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# User settings
settings = {
    "num_workers": 0,
    "random_seed": 42,
    "infection_rate_range": (0.10, 0.20),
    "recovery_rate_range": (0.05, 0.15),
    "population_size": 5000,
    "num_time_steps": 256,
    "num_realisations": 1000,
    "nn_epochs": 256,
    "nn_batch_size": 32,
    "input_size": 3,
    "hidden_size": 64,
    "output_size": 256,
    "learning_rate": 0.001,
    "model_type": "GRU",  # FFNN or "GRU"
    "test_pct": 0.1,
    "val_pct": 0.1,
    "mode": "comparison",  # emulation or "comparison"
    "scenario": [0.15, 0.10, 1000]  # a specific scenario with the infection rate, recovery rate, and population size
}

# ABM Functions
def update_agents(agents, infection_rate, recovery_rate, N):
    new_infections = 0
    new_recoveries = 0
    for i in np.where(agents == 'I')[0]:
        if np.random.uniform() < infection_rate: 
            contact = np.random.choice(N)
            if agents[contact] == 'S': 
                agents[contact] = 'I' 
                new_infections += 1
        if np.random.uniform() < recovery_rate: 
            agents[i] = 'R'
            new_recoveries += 1
    return agents, new_infections, new_recoveries

def generate_ABM_data(settings):
    all_realisations = []
    for _ in tqdm(range(settings["num_realisations"]), desc='Generating ABM data'):
        infection_rate = np.random.uniform(*settings["infection_rate_range"])
        recovery_rate = np.random.uniform(*settings["recovery_rate_range"])
        agents = np.repeat('S', settings["population_size"])
        agents[np.random.choice(settings["population_size"])] = 'I'
        SIR_data = {'S': [], 'I': [], 'R': []}
        incidences = []
        for t in range(1, settings["num_time_steps"] + 1):
            agents, new_infections, new_recoveries = update_agents(agents, infection_rate, recovery_rate, settings["population_size"])
            SIR_data['S'].append(np.sum(agents == 'S'))
            SIR_data['I'].append(np.sum(agents == 'I'))
            SIR_data['R'].append(np.sum(agents == 'R'))
            incidences.append(new_infections)
        all_realisations.append({'SIR_data': pd.DataFrame(SIR_data), 'incidences': incidences})
    return all_realisations

# For a FFNN model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # Define ReLU operation
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # Apply ReLU to output to contrain function to non-negative values
        x = self.relu(x)
        return x

# For a GRU model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x.view(len(x), 1, -1))
        x = self.fc(x.view(len(x), -1))
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    progress_bar = tqdm(range(epochs), desc='Training model', dynamic_ncols=True)

    for epoch in progress_bar:
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        if val_loader:
            with torch.no_grad():
                val_loss = 0
                val_batches = 0
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    val_batches += 1
                avg_val_loss = val_loss / val_batches
                progress_bar.set_description(f'Training model (Loss: {loss.item():.4f}, Avg Val Loss: {avg_val_loss:.4f}), completed')


# Run the emulator
def run_emulator(model, test_loader):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Running emulator'):
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actual.append(targets.numpy())
            
    return predictions, actual


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Data generation
    np.random.seed(settings["random_seed"])
    ABM_data = generate_ABM_data(settings)

    X = []
    Y = []

    for realisation in ABM_data:
        X.append([np.random.uniform(*settings["infection_rate_range"]), np.random.uniform(*settings["recovery_rate_range"]), settings["population_size"]])
        Y.append(realisation['incidences'])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

    # Split data into train, validation and test
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=settings["test_pct"], random_state=settings["random_seed"])
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=settings["val_pct"]/(1-settings["test_pct"]), random_state=settings["random_seed"])

    # Convert to tensor datasets
    train_data = TensorDataset(X_train, Y_train)
    val_data = TensorDataset(X_val, Y_val)
    test_data = TensorDataset(X_test, Y_test)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])
    val_loader = DataLoader(val_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])
    test_loader = DataLoader(test_data, batch_size=settings["nn_batch_size"], shuffle=True, num_workers=settings["num_workers"])

    # Model selection
    if settings["model_type"] == 'FFNN':
        model = FFNN(settings["input_size"], settings["hidden_size"], settings["output_size"])
    elif settings["model_type"] == 'GRU':
        model = GRU(settings["input_size"], settings["hidden_size"], settings["output_size"])

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"])

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader, settings["nn_epochs"])

    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')

    # Load the model when you want to run the emulator
    if settings["model_type"] == 'FFNN':
        model = FFNN(settings["input_size"], settings["hidden_size"], settings["output_size"])
    elif settings["model_type"] == 'GRU':
        model = GRU(settings["input_size"], settings["hidden_size"], settings["output_size"])

    model.load_state_dict(torch.load('model.pth'))

    if settings["mode"] == "comparison":
        # Run the emulator
        predictions, actual = run_emulator(model, test_loader)

        # Convert predictions and actual to numpy for easier handling
        predictions_np = np.concatenate(predictions, axis=0)
        actual_np = np.concatenate(actual, axis=0)

        # Select 25 random epidemics for plotting
        num_plots = 25
        indices = np.random.choice(range(predictions_np.shape[0]), size=num_plots, replace=False)

        # Create a 5x5 grid of subplots
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))

        for i, ax in enumerate(axes.flat):
            idx = indices[i]
            ax.plot(predictions_np[idx], label='Predicted')
            ax.plot(actual_np[idx], label='Actual')
            ax.set_title(f'Epidemic {idx+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('New Infections')
            ax.legend()

        # Adjust the layout so the plots do not overlap
        plt.tight_layout()
        plt.show()

    elif settings["mode"] == "emulation":
        # Define the specific scenario to emulate
        scenario = torch.tensor([settings["scenario"]], dtype=torch.float32)

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

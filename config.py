import os

# User settings
settings = {
    "data": {
        "generate_ABM": False, # If True, generates Agent-Based Model (ABM) data; if False, uses data from a saved file (WARNING: If set to "True" and no data is detected it will run simulation program)
        "data_dir": os.path.join(os.getcwd(), "data"), # Directory containing the preprocessed ABM dataset (ideally siremu)
        "num_workers": 0, # Number of workers to use for loading data in DataLoader
        "shuffle": True, # If True, shuffles the data in DataLoader
        "test_pct": 0.2, # Fraction of data used for testing
        "val_pct": 0.2  # Fraction of data used for validation
    },
    "execution": {
        "max_workers": 16, # Maximum number of workers for ProcessPoolExecutor (optimal for current system configuration)
        "random_seed": 42, # Seed for random number generator to ensure reproducibility
        "mode": "comparison",  # Mode of operation: 'emulation' to emulate the ABM or 'comparison' to compare with other methods
        "cached_model": True # Used saved trained model (WARNING: If set to "True" and no cached model is detected it will run training program)
    },
    "ABM": {
        "infection_rate_range": (0.1, 0.5), # Range of daily infection rates to sample from
        "recovery_rate_range": (0.1, 0.5), # Range of daily recovery rates to sample from
        "population_size": 10000, # Total population size
        "num_time_steps": 256, # Number of time-series steps in ABM
        "num_realisations": 128, # Number of different realisations (i.e., simulations) for a given set of rates
        "num_iterations": 16, # Number of iterations to re-run the ABM with a fixed set of rates
        "scenario": [0.2, 0.3, 10000]  # A specific scenario detailing daily infection rate, daily recovery rate, and population size
    },
    "neural_net": {
        "nn_epochs": 256, # Number of training epochs
        "nn_batch_size": 64, # Number of samples per batch to load
        "input_size": 3, # Number of input neurons
        "hidden_size": 64, # Number of hidden neurons in the layer
        "output_size": 256, # Number of output neurons
        "model_type": "BiRNN", # Type of neural network model: FFNN, GRU, LSTM or BiRNN
        "lr_scheduler": { 
            "learning_rate": 0.0001, # Initial learning rate for the optimizer
            "step_size": 64, # Number of epochs before changing the learning rate
            "gamma": 0.8 # Factor to reduce the learning rate by
        },
        "dropout_prob": 0.5
    },
    "plotting": {
        "num_plots": 9,  # Number of random epidemics for plotting in comparison mode
        "figure_size_comparison": (20, 20),  # Size of figure in comparison mode
        "figure_size_emulation": (10, 5),  # Size of figure in emulation mode
    }
}
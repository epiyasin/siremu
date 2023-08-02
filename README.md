# Siremu: A Python Package for Epidemic Simulation and Emulation

Siremu is a Python package designed to simulate and emulate epidemic dynamics using agent-based modeling (ABM) and neural networks (Feed-Forward Neural Network and Gated Recurrent Unit). The package generates the ABM data for given infection rates, recovery rates and population size. It can also train neural network models on this data and use the models to predict the dynamics of the epidemic.

## Installation

## Install siremu

```
git clone https://github.com/your_username/siremu.git
cd siremu
pip install -r requirements.txt
```

### Install Dependencies

The package uses numpy, pandas, tqdm, torch, matplotlib, and scikit-learn. You can install these dependencies with pip:

```
pip install numpy pandas torch tqdm matplotlib sklearn
```

After the dependencies are installed, you can download this package and use it in your Python code.

## Usage

You can run the main file with the following line:

```
python main.py
```

## Configuration

The user settings can be found at the bottom of the main file. These are the available settings:

- **num_workers**: number of workers for data loading (default is 0).
- **random_seed**: seed for random number generation (default is 42).
- **infection_rate_range**: tuple representing the range of infection rates (default is (0.05, 0.25)).
- **recovery_rate_range**: tuple representing the range of recovery rates (default is (0.05, 0.25)).
- **population_size**: size of the population for the ABM (default is 50000).
- **num_time_steps**: number of time steps in the ABM simulation (default is 256).
- **num_realisations**: number of realisations of the ABM (default is 100).
- **num_iterations**: number of iterations for ABM data generation (default is 10).
- **nn_epochs**: number of epochs for neural network training (default is 256).
- **nn_batch_size**: batch size for neural network training (default is 32).
- **input_size**: size of the input layer for the neural network models (default is 3).
- **hidden_size**: size of the hidden layer for the neural network models (default is 64).
- **output_size**: size of the output layer for the neural network models (default is 256).
- **learning_rate**: learning rate for the neural network models (default is 0.001).
- **model_type**: the type of neural network model to use ('FFNN' or 'GRU', default is 'FFNN').
- **test_pct**: the percentage of data to use for testing (default is 0.1).
- **val_pct**: the percentage of data to use for validation (default is 0.1).
- **mode**: the mode to run the main script in ('emulation' or 'comparison', default is 'comparison').
- **scenario**: a specific scenario with the infection rate, recovery rate, and population size for emulation mode (default is [0.15, 0.10, 1000]).

## Functions

The package includes several functions. Here are the key ones:

- **generate_ABM_data(settings)**: generates the ABM data.
- **update_agents(agents, infection_rate, recovery_rate, N)**: updates the agents in the ABM.
- **train_model(model, criterion, optimizer, train_loader, val_loader, epochs)**: trains the specified model.
- **run_emulator(model, test_loader)**: runs the emulator using the specified model.
- **FFNN(input_size, hidden_size, output_size)**: defines the architecture of the feed-forward neural network.
- **GRU(input_size, hidden_size, output_size)**: defines the architecture of the gated recurrent unit.

## License

This project is licensed under the terms of the MIT license.
from dataclasses import dataclass, field, InitVar
import os

@dataclass
class MINTData:
    preprocess: bool = False  # If True, preprocesses the data; if False, reads the data from the folder
    data_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "mint_data", "mint_data_scaled.csv"))  # Path to the MINT data file

@dataclass
class Execution:
    source: str = "MINT"  # Source of the data: "MINT" or "ABM"
    max_workers: int = 16  # Maximum number of workers for ProcessPoolExecutor
    random_seed: int = 42  # Seed for random number generator to ensure reproducibility
    cached_model: bool = False  # Use a saved trained model (WARNING: If set to "True" and no cached model is detected, the training program will run)

@dataclass
class LrScheduler:
    learning_rate: float = 0.0001  # Initial learning rate for the optimizer
    step_size: int = 64  # Number of epochs before changing the learning rate
    gamma: float = 0.8  # Factor to reduce the learning rate by

@dataclass
class NeuralNet:
    nn_epochs: int = 32  # Number of training epochs
    nn_batch_size: int = 64  # Number of samples per batch to load
    hidden_size: int = 64  # Number of hidden neurons in the layer
    model_type: str = "FFNN"  # Type of neural network model: FFNN, GRU, LSTM, or BiRNN
    lr_scheduler: LrScheduler = LrScheduler()  # Learning rate scheduler settings
    dropout_prob: float = 0.5  # Dropout probability
    shuffle: bool = True  # If True, shuffles the data in DataLoader
    num_workers: int = 0  # Number of workers to use for loading data in DataLoader
    test_pct: float = 0.2  # Fraction of data used for testing
    val_pct: float = 0.2  # Fraction of data used for validation
    input_size: int = 20  # Number of input neurons
    output_size: int = 61  # Number of output neurons

@dataclass
class Plotting:
    num_plots: int = 9  # Number of random epidemics for plotting in comparison mode
    figure_size_comparison: tuple = (20, 20)  # Size of the figure in comparison mode
    figure_size_emulation: tuple = (10, 5)  # Size of the figure in emulation mode

@dataclass
class Settings:
    MINT: MINTData = MINTData()  # MINT data settings
    execution: Execution = Execution()  # Execution settings
    neural_net: NeuralNet = NeuralNet()  # Neural network settings
    plotting: Plotting = Plotting()  # Plotting settings

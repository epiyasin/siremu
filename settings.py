from dataclasses import dataclass, field, InitVar
import os

@dataclass
class MINTData:
    preprocess: bool = False
    data_path: str = field(default_factory=lambda: os.path.join(os.getcwd(), "mint_data", "mint_data_scaled.csv"))

@dataclass
class Execution:
    source: str = "MINT"
    max_workers: int = 16
    random_seed: int = 42
    cached_model: bool = False

@dataclass
class LrScheduler:
    learning_rate: float = 0.0001
    step_size: int = 64
    gamma: float = 0.8

@dataclass
class NeuralNet:
    nn_epochs: int = 32
    nn_batch_size: int = 64
    hidden_size: int = 64
    model_type: str = "FFNN"
    lr_scheduler: LrScheduler = LrScheduler()
    dropout_prob: float = 0.5
    shuffle: bool = True
    num_workers: int = 0
    test_pct: float = 0.2
    val_pct: float = 0.2
    input_size: int = 20
    output_size: int = 61

@dataclass
class Plotting:
    num_plots: int = 9
    figure_size_comparison: tuple = (20, 20)
    figure_size_emulation: tuple = (10, 5)

@dataclass
class Settings:
    MINT: MINTData = MINTData()
    execution: Execution = Execution()
    neural_net: NeuralNet = NeuralNet()
    plotting: Plotting = Plotting()

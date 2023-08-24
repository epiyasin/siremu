import torch
from nn_mint_data_handler import prepare_nn_mint_data
from testing import run_emulator
from plotting import plot_mint_compare, plot_mint_time_series, plot_mint_avg_compare, plot_mint_avg_time_series
from nn_data_handler import prepare_nn_data
from model_handler import handle_model
from settings import Settings

if __name__ == "__main__":
    
    settings = Settings()

    train_loader, val_loader, test_loader, scaler = prepare_nn_mint_data(settings)

    # Handle model (loading, training, etc.)
    model = handle_model(train_loader, val_loader, settings)

    # Run the emulator
    predictions, actual = run_emulator(model, test_loader)
    plot_mint_compare(predictions, actual, settings)
    plot_mint_time_series(predictions, actual, settings)
    plot_mint_avg_compare(predictions, actual, settings)
    plot_mint_avg_time_series(predictions, actual, settings)
import torch
from nn_mint_data_handler import prepare_nn_mint_data
from testing import run_emulator
from plotting import plot_comparison, plot_emulation, plot_mint_compare, plot_mint_time_series, plot_mint_avg_compare, plot_mint_avg_time_series
from ABM_simulation_data_handler import generate_data, load_data
from utils import check_data_folder_exists, check_data_exists, generate_settings
from nn_data_handler import prepare_nn_data
from model_handler import handle_model

source = "ABM"

if __name__ == "__main__":
    
    settings = generate_settings(source=source)
    settings["execution"]["source"] = source
    data_folder_path = settings["ABM"]["data"]["data_dir"]

    # Check if the data folder exists and create it if not
    folder_exists = check_data_folder_exists(data_folder_path)

    # If the folder didn't exist, we know we need to generate data.
    # Otherwise, check if the data file exists within the folder
    if settings["execution"]["source"] == "ABM":
        should_generate_data = not folder_exists or (folder_exists and not check_data_exists(data_folder_path))

        if should_generate_data or settings["ABM"]["data"]["generate_ABM"]:
            X, Y, ABM_data = generate_data(settings)
        else:
            X, Y, ABM_data = load_data(settings)
        train_loader, val_loader, test_loader, scaler = prepare_nn_data(X, Y, settings)
    else:
        train_loader, val_loader, test_loader, scaler = prepare_nn_mint_data(settings)

    # Handle model (loading, training, etc.)
    model = handle_model(train_loader, val_loader, settings)

    if settings["execution"]["source"] == "MINT":
        if settings["execution"]["mode"] == "comparison":
	        # Run the emulator
            predictions, actual = run_emulator(model, test_loader)
            plot_mint_compare(predictions, actual, settings)
            plot_mint_time_series(predictions, actual, settings)
            plot_mint_avg_compare(predictions, actual, settings)
            plot_mint_avg_time_series(predictions, actual, settings)

    else:   
        if settings["execution"]["mode"] == "comparison":
            # Run the emulator
            predictions, actual = run_emulator(model, test_loader)
            
            # Call the plotting function
            plot_comparison(predictions, actual, ABM_data, settings)        
        elif settings["execution"]["mode"] == "emulation":
            # Emulate using the specific scenario
            emulation_input = torch.tensor([settings["ABM"]["scenario"]], dtype=torch.float32)
            
            # Normalize input
            emulation_input_scaled = scaler.transform(emulation_input.numpy())
            emulation_input = torch.tensor(emulation_input_scaled, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                emulation_output = model(emulation_input)

            # Call the plotting function for emulation
            plot_emulation(emulation_output, settings)
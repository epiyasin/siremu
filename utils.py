import os
import numpy as np
from models import FFNN, GRU, LSTM, BiRNN
from settings import Settings

settings = Settings("config.json")

def check_data_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found, creating...")
        os.makedirs(folder_path)
        return False
    return True

def check_data_exists(data_dir):
    return os.path.exists(os.path.join(data_dir, 'ABM_data.pth'))

def check_model_exists(model_type, source):
    sub_folder = f"{source}_models"
    model_path = os.path.join("cached_models", sub_folder, f"{source}_{model_type}model.pth")
    return os.path.exists(model_path)


def select_model(settings):
    if settings.model_type == 'FFNN':
        return FFNN(settings.input_size, settings.hidden_size, settings.output_size,settings.dropout_prob)
    elif settings.model_type == 'GRU':
        return GRU(settings.input_size, settings.hidden_size, settings.output_size,settings.dropout_prob)
    elif settings.model_type == 'LSTM':
        return LSTM(settings.input_size, settings.hidden_size, settings.output_size,settings.dropout_prob)
    elif settings.model_type == 'BiRNN':
        return BiRNN(settings.input_size, settings.hidden_size, settings.output_size,settings.dropout_prob)

def attach_identifier(data):
    # Assumes the data is a list of numpy arrays
    identifiers = [np.full((arr.shape[0], 1), i) for i, arr in enumerate(data)]
    return np.concatenate([np.hstack((arr, id)) for arr, id in zip(data, identifiers)], axis=0)

def distinct_rates(ABM_data, num_plots):
    # Fetch distinct rates until you get the desired amount (or close to it)
    distinct_indices = []
    seen_rates = set()
    
    for i, data in enumerate(ABM_data):
        rate_tuple = (data['infection_rate'], data['recovery_rate'])
        if rate_tuple not in seen_rates:
            seen_rates.add(rate_tuple)
            distinct_indices.append(i)
        
        if len(distinct_indices) == num_plots:
            break

    return distinct_indices
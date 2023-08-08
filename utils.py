import os
from models import FFNN, GRU, LSTM, BiRNN

def check_data_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found, creating...")
        os.makedirs(folder_path)
        return False
    return True

def check_data_exists(data_dir):
    return os.path.exists(os.path.join(data_dir, 'ABM_data.pth'))

def check_model_exists(model_type):
    return os.path.exists(os.path.join("cached_models", model_type + 'model.pth'))

def select_model(settings):
    if settings["neural_net"]["model_type"] == 'FFNN':
        return FFNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'GRU':
        return GRU(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'LSTM':
        return LSTM(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'BiRNN':
        return BiRNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])

def distinct_rates(indices, ABM_data):
    # This function will check if the selected indices have distinct infection and recovery rates
    rates = [(round(ABM_data[i]['infection_rate'], 3), round(ABM_data[i]['recovery_rate'], 3)) for i in indices]
    return len(rates) == len(set(rates))
import os
import numpy as np
import torch
from ABM import generate_ABM_data
from settings import Settings

settings = Settings("config.json")

def generate_data(settings):
    print("Generating data...")

    # Data generation
    np.random.seed(settings.get_random_seed())
    ABM_data = generate_ABM_data(settings)

    # Extract X, Y, and the rates from the ABM data
    X = []
    Y = []

    for realisation in ABM_data:
        X.append([
            np.random.uniform(*settings.get_abm_infection_rate_range()),
            np.random.uniform(*settings.get_abm_recovery_rate_range()),
            settings.get_abm_population_size()
        ])
        Y.append(realisation['incidences'])

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.stack([torch.tensor(i, dtype=torch.float32) for i in Y]) 

    data_to_save = {
        'X': X,
        'Y': Y,
        'ABM_data': ABM_data
    }
    torch.save(data_to_save, os.path.join(settings.get_abm_data_dir(), 'ABM_data.pth'))
    return X, Y, ABM_data

def load_data(settings):
    data_folder_path = settings.get_abm_data_dir()
    data_file_path = os.path.join(data_folder_path, 'ABM_data.pth')
    loaded_data = torch.load(data_file_path)
    X = loaded_data['X'].clone().detach()
    Y = loaded_data['Y'].clone().detach()
    ABM_data = loaded_data['ABM_data']

    return X, Y, ABM_data

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    new_infections = (new_infections / N) * 1000
    return agents, new_infections, new_recoveries

def generate_ABM_data(settings):
    all_realisations = []
    
    realisation_progress = tqdm(range(settings["num_realisations"]), desc='Realisations', dynamic_ncols=True)
    
    np.random.seed(settings["random_seed"])

    for _ in realisation_progress:
        infection_rate = np.random.uniform(*settings["infection_rate_range"])
        recovery_rate = np.random.uniform(*settings["recovery_rate_range"])

        iteration_progress = tqdm(range(settings["num_iterations"]), desc=f'Iterations (Infection Rate: {infection_rate:.2f}, Recovery Rate: {recovery_rate:.2f})', leave=False, dynamic_ncols=True)

        for _ in iteration_progress:
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
            iteration_progress.set_description(f'Iterations (Infection Rate: {infection_rate:.2f}, Recovery Rate: {recovery_rate:.2f}), completed')
        realisation_progress.set_description('Realisations, completed')
    return all_realisations
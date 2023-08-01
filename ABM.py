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
    return agents, new_infections, new_recoveries

def generate_ABM_data(settings):
    all_realisations = []
    for _ in tqdm(range(settings["num_realisations"]), desc='Generating ABM data'):
        infection_rate = np.random.uniform(*settings["infection_rate_range"])
        recovery_rate = np.random.uniform(*settings["recovery_rate_range"])
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
    return all_realisations

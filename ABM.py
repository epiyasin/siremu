import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from settings import Settings

settings = Settings("config.json")

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
    #new_infections = (new_infections / N) * 1000
    return agents, new_infections, new_recoveries

def generate_iteration(infection_rate, recovery_rate, settings):
    agents = np.repeat('S', settings.get_abm_population_size())
    agents[np.random.choice(settings.get_abm_population_size())] = 'I'
    SIR_data = {'S': [], 'I': [], 'R': []}
    incidences = []
    for t in range(1, settings.get_abm_num_time_steps() + 1):
        agents, new_infections, new_recoveries = update_agents(agents, infection_rate, recovery_rate, settings.get_abm_population_size())
        SIR_data['S'].append(np.sum(agents == 'S'))
        SIR_data['I'].append(np.sum(agents == 'I'))
        SIR_data['R'].append(np.sum(agents == 'R'))
        incidences.append(new_infections)
    return {'SIR_data': pd.DataFrame(SIR_data), 'incidences': incidences, 'infection_rate': infection_rate, 'recovery_rate': recovery_rate}

def generate_ABM_data(settings):
    all_realisations = []

    realisation_progress = tqdm(range(settings.get_abm_num_realisations()), desc='Realisations', dynamic_ncols=True)

    np.random.seed(settings.get_random_seed())

    for _ in realisation_progress:
        infection_rate = np.random.uniform(*settings.get_abm_infection_rate_range())
        recovery_rate = np.random.uniform(*settings.get_abm_recovery_rate_range())

        with ProcessPoolExecutor(max_workers=settings.get_max_workers()) as executor:
            future_results = [executor.submit(generate_iteration, infection_rate, recovery_rate, settings) for _ in range(settings.get_abm_num_iterations())]
            all_realisations.extend([future.result() for future in future_results])

    return all_realisations
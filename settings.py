import json

class Settings:

    def __init__(self, config_file="config.json"):
        self._load_from_config_file(config_file)

    def _load_from_config_file(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Execution settings
        self.execution_source = config.get('execution', {}).get('source', 'MINT')
        self.max_workers = config.get('execution', {}).get('max_workers', 16)
        self.random_seed = config.get('execution', {}).get('random_seed', 42)
        self.mode = config.get('execution', {}).get('mode', 'comparison')
        self.cached_model = config.get('execution', {}).get('cached_model', True)

        # Neural net settings
        self.nn_epochs = 2048 if self.execution_source == "ABM" else 32
        self.nn_batch_size = config.get('neural_net', {}).get('nn_batch_size', 64)
        self.input_size = 3 if self.execution_source == "ABM" else 20
        self.hidden_size = config.get('neural_net', {}).get('hidden_size', 64)
        self.output_size = 256 if self.execution_source == "ABM" else 61
        self.model_type = config.get('neural_net', {}).get('model_type', 'GRU')
        self.lr_scheduler = config.get('neural_net', {}).get('lr_scheduler', {
            "learning_rate": 0.0001,
            "step_size": 64,
            "gamma": 0.8
        })
        self.dropout_prob = config.get('neural_net', {}).get('dropout_prob', 0.5)
        self.shuffle = config.get('neural_net', {}).get('shuffle', True)
        self.num_workers = config.get('neural_net', {}).get('num_workers', 0)
        self.test_pct = config.get('neural_net', {}).get('test_pct', 0.2)
        self.val_pct = config.get('neural_net', {}).get('val_pct', 0.2)

        # ABM settings
        self.abm_generate = config.get('ABM', {}).get('data', {}).get('generate_ABM', False)
        self.abm_data_dir = config.get('ABM', {}).get('data', {}).get('data_dir', None)
        self.abm_infection_rate_range = config.get('ABM', {}).get('infection_rate_range', (0.05, 0.8))
        self.abm_recovery_rate_range = config.get('ABM', {}).get('recovery_rate_range', (0.05, 0.8))
        self.abm_population_size = config.get('ABM', {}).get('population_size', 10000)
        self.abm_num_time_steps = config.get('ABM', {}).get('num_time_steps', 256)
        self.abm_num_realisations = config.get('ABM', {}).get('num_realisations', 128)
        self.abm_num_iterations = config.get('ABM', {}).get('num_iterations', 8)
        self.abm_scenario = config.get('ABM', {}).get('scenario', [0.2, 0.3, 10000])

        # MINT settings
        self.mint_preprocess = config.get('MINT', {}).get('data', {}).get('preprocess', False)
        self.mint_data_path = config.get('MINT', {}).get('data', {}).get('data_path', None)
        
        # Plotting settings
        self.num_plots = config.get('plotting', {}).get('num_plots', 9)
        self.figure_size_comparison = config.get('plotting', {}).get('figure_size_comparison', (20, 20))
        self.figure_size_emulation = config.get('plotting', {}).get('figure_size_emulation', (10, 5))
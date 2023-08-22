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

    # Getter methods for execution settings
    def get_execution_source(self):
        return self.execution_source

    def get_max_workers(self):
        return self.max_workers

    def get_random_seed(self):
        return self.random_seed

    def get_mode(self):
        return self.mode

    def get_cached_model(self):
        return self.cached_model

    # Getter methods for neural net settings
    def get_nn_epochs(self):
        return self.nn_epochs

    def get_nn_batch_size(self):
        return self.nn_batch_size

    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_output_size(self):
        return self.output_size

    def get_model_type(self):
        return self.model_type

    def get_lr_scheduler_learning_rate(self):
        return self.lr_scheduler["learning_rate"]

    def get_lr_scheduler_step_size(self):
        return self.lr_scheduler["step_size"]

    def get_lr_scheduler_gamma(self):
        return self.lr_scheduler["gamma"]

    def get_dropout_prob(self):
        return self.dropout_prob

    def get_shuffle(self):
        return self.shuffle

    def get_num_workers(self):
        return self.num_workers

    def get_test_pct(self):
        return self.test_pct

    def get_val_pct(self):
        return self.val_pct

    # Getter methods for ABM settings
    def get_abm_generate(self):
        return self.abm_generate

    def get_abm_data_dir(self):
        return self.abm_data_dir

    def get_abm_infection_rate_range(self):
        return self.abm_infection_rate_range

    def get_abm_recovery_rate_range(self):
        return self.abm_recovery_rate_range

    def get_abm_population_size(self):
        return self.abm_population_size

    def get_abm_num_time_steps(self):
        return self.abm_num_time_steps

    def get_abm_num_realisations(self):
        return self.abm_num_realisations

    def get_abm_num_iterations(self):
        return self.abm_num_iterations

    def get_abm_scenario(self):
        return self.abm_scenario

    # Getter methods for MINT settings
    def get_mint_preprocess(self):
        return self.mint_preprocess

    def get_mint_data_path(self):
        return self.mint_data_path

    # Getter methods for plotting settings
    def get_num_plots(self):
        return self.num_plots

    def get_figure_size_comparison(self):
        return self.figure_size_comparison

    def get_figure_size_emulation(self):
        return self.figure_size_emulation

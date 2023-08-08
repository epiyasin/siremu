from models import FFNN, GRU, LSTM, BiRNN

def select_model(settings):
    if settings["neural_net"]["model_type"] == 'FFNN':
        return FFNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'GRU':
        return GRU(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'LSTM':
        return LSTM(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])
    elif settings["neural_net"]["model_type"] == 'BiRNN':
        return BiRNN(settings["neural_net"]["input_size"], settings["neural_net"]["hidden_size"], settings["neural_net"]["output_size"],settings["neural_net"]["dropout_prob"])

import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.fc3 = nn.Linear(output_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.softplus(x) 
        return x
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x, _ = self.gru(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.fc(x.view(len(x), -1))
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.fc(x.view(len(x), -1))
        return x

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, output_size)
        self.ln = nn.LayerNorm(2 * hidden_size)

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))
        x = self.ln(x)
        x = self.fc(x.view(len(x), -1))
        return x

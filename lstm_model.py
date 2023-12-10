from torch import nn

# Build the neural network model
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, :500, :])  # Truncate the outputs to first 500 steps
        return out

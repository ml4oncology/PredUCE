import torch.nn as nn
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,dropout=0.2):
        # super(GRUModel, self).__init__()
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,dropout= dropout)
        self.outlayer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.outlayer(x)
        return x



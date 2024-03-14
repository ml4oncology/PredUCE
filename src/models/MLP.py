import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, output_size)
        self.relu = nn.ReLU()
        # Add a dropout layer with the specified dropout rate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after the first hidden layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after the second hidden layer
        x = self.layer3(x)
        return x
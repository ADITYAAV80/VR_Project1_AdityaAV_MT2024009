import torch
import torch.nn as nn

class NueralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=8100, out_features=128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.10)
        self.layer_2 = nn.Linear(in_features=128, out_features=64)
        self.layer_3 = nn.Linear(in_features=64, out_features= 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        return x
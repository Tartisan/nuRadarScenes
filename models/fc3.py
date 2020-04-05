import torch
from torch import nn

class get_model(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(get_model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(10, 9),nn.BatchNorm1d(9),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(9, 4),nn.BatchNorm1d(4),nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(4, 2)))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
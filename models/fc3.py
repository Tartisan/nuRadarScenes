import torch
from torch import nn

class FC3(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(FC3, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                 nn.BatchNorm1d(n_hidden_1),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden_1, n_hidden_2),
                                 nn.BatchNorm1d(n_hidden_2),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.net(x)
        return x
"""
@File   :   models.py
@Date   :   2023/7/14
@Description    :   This file is for defining the models
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Class to design a MLP model."""

    def __init__(self):
        """Initialisation of the class (constructor)."""

        super().__init__()

        first_layer = 256
        second_layer = 256

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

        self.bnin = nn.BatchNorm1d(first_layer)
        self.bnbout = nn.BatchNorm1d(second_layer)

        self.linin = nn.Linear(72, first_layer, bias=True)
        self.linbout = nn.Linear(first_layer, second_layer, bias=True)
        self.linout = nn.Linear(second_layer, 18, bias=True)

    def forward(self, input_data):
        """The layers are stacked to transport the data through the neural network for the forward part."""
        # Input:
        # input_data; torch.Tensor
        # Output:
        # x; torch.Tensor

        x = self.linin(torch.flatten(input_data, 1))
        x = self.bnin(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linbout(x)
        x = self.bnbout(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linout(x)
        x = self.softmax(x)

        return x

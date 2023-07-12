import torch
import torch.nn as nn
import torch.optim as optim


# 定义MLP模型
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out

class MLP(nn.Module):
    """Class to design a MLP model."""

    def __init__(self):
        """Initialisation of the class (constructor)."""

        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)

        self.bnin = nn.BatchNorm1d(128)
        self.bnbout = nn.BatchNorm1d(128)

        self.linin = nn.Linear(72, 128, bias = True)
        self.linbout = nn.Linear(128, 128, bias = True)
        self.linout = nn.Linear(128, 18, bias = True)

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
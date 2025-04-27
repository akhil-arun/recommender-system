import torch.nn as nn
import torch

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.stack = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        x = self.flatten(x)
        out = self.stack(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return torch.sigmoid(out).squeeze()

"""
Module implements a convolutional neural network
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from .base import PolicyNet, get_cuda, get_numpy


class ConvNet(PolicyNet):
    def __init__(self, input_shape=(4,11,11), num_channels=64, output_size=6, batch_norm=True, activation=F.relu, learning_rate=1E-3):
        super(ConvNet, self).__init__()

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.activation = activation
        self.flattened_size = num_channels * (input_shape[1] - 2) * (input_shape[2] - 2)

        self.conv1 = nn.Conv2d(input_shape[0], num_channels, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            self.bn3 = nn.BatchNorm2d(num_channels)
        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)


    def forward(self, obs):
        x = self.compact_state_list(obs)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.activation(x)
        out = self.fc2(x)
        return out #torch.exp(F.log_softmax(out, dim=1))


    def compact_state_list(self, obs):
        # pylint: disable=maybe-no-member
        return torch.from_numpy(np.array([self.compact_state(s) for s in obs])).float()


    def compact_state(self, obs):
        """
        Returns a compact state representation of a 4x11x11 ndarray
        describing agents position, enemies, danger zones and the board itself.
        Useful for convolution layers with 4 input channels
        """
        board = self.board_no_agents(obs)
        enemy = self.enemy_map(obs)
        danger = self.danger_map(obs)
        position = self.position_map(obs)

        return np.stack((board, enemy, danger, position))
        
    # Used in Q learning
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

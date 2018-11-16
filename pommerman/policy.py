"""
Module implements police networks for the REINFORCE algorithm
"""
import torch
from torch import nn, optim, F
from util import get_cuda, get_numpy
import numpy as np


class PolicyNet(nn.Module):
    """Policy network"""

    def loss(self, action_probabilities, returns):
        return -torch.mean(torch.mul(torch.log(action_probabilities), returns))


    def enemy_map(self, obs):
        """
        Returns an 11x11 ndarray where 1 indicated enemy
        and 0 indicates no enemy
        """
        board = obs['board'].copy()
        enemies = obs['enemies']
        mask = np.isin(board, enemies)
        board[mask] = 1
        board[~mask] = 0
        return board


    def position_map(self, obs):
        """
        Returns an 11x11 ndarray with a single value set to 1
        indicating the agents position. Other values are 0
        """
        board = np.zeros_like(obs['board'])
        pos = obs['position']
        board[pos[0], pos[1]] = 1
        return board    


    def danger_map(self, obs):
        """
        Returns an 11x11 ndarray where 0 indicates no danger
        and >0 indicates some danger level (typically bombs).
        Greater numbers means more immediate danger
        """
        board = np.zeros_like(obs['board'])
        strength = obs['board_blast_strength']
        life = obs['bomb_life']
        for x,y in np.argwhere(life>0):
            s = strength[x,y]
            l = life[x,y]
            board[y,max(0, x-s):min(10, x+s)] = l
            board[max(0, y-s):min(10, y+s),x] = l
        return board



class ConvNet(PolicyNet):
    def __init__(
            self, input_shape, num_channels=64, output_size=512,
            batch_norm=True, activation=F.relu):
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


    def forward(self, x):
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
        return out


    def compact_state(self, obs):
        """
        Returns a compact state representation of a 4x11x11 ndarray
        describing agents position, enemies, danger zones and the board itself.
        Useful for convolution layers with 4 input channels
        """
        board = obs['board']
        enemy = self.enemy_map(obs)
        danger = self.danger_map(obs)
        position = self.position_map(obs)
        return np.stack((board, enemy, danger, position))


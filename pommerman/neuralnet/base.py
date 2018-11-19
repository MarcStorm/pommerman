"""
Module implements general utilifies and base classes for neural networks
"""
import torch
from torch import nn
import numpy as np

use_cuda = torch.cuda.is_available()
print("Cuda:",use_cuda)


def get_cuda(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().numpy()
    return x.numpy()



class PolicyNet(nn.Module):
    """
    Policy network is a base class for policy networks which
    implements general utilities. Every neural network for the
    pommerman environment must have a `forward` method which takes
    the observation state `obs` as input and returns a tensor with the
    size of the action space.
    """

    def loss(self, action_probabilities, returns):
        # pylint: disable=maybe-no-member
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
        strength = obs['bomb_blast_strength']
        life = obs['bomb_life']
        for x,y in np.argwhere(life>0):
            s = strength[x,y]
            l = life[x,y]
            board[y,max(0, x-s):min(10, x+s)] = l
            board[max(0, y-s):min(10, y+s),x] = l
        return board

"""
Module implements base utilities for use in the train module
"""
import torch
import os
import datetime
import numpy as np


class BaseTraining(object):
    """
    Base training class which sets up the basics for the training environment for a net.
    It sets up the environment

    Note:
        This method should never be instantiated on its own. It is used to inherit from.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

    def saveNetwork(self):
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #t = datetime.date.today().strftime("%Y-%m-%d")
        t = datetime.date.today().strftime("%m_%d_%Y")
        filename = "resources/q_agent_{}.pt".format(t)
        PATH = os.path.join(dirpath, filename)
        print (PATH)
        
        torch.save(self.neuralNet.state_dict(), PATH)
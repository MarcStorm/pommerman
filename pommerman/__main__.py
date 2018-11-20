#!/usr/bin/bash

from .train.policylearn import PolicyTraining
from .neuralnet.convolutional import ConvNet

if __name__ == '__main__':
    # Chose a neural network to train
    net = ConvNet()

    # Initialize a trainer
    trainer = PolicyTraining(net, num_episodes=250, discount_factor=0.99, val_freq=100)

    # Start training the network
    trainer.train()

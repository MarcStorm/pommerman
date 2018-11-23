#!/usr/bin/bash

from train import PolicyTraining
from neuralnet.convolutional import ConvNet
from environment import stopEnv

if __name__ == '__main__':
    # Chose a neural network to train
    net = ConvNet()

    # Set up an environment
    env = stopEnv(None)

    # Initialize a trainer
    trainer = PolicyTraining(env, net, num_episodes=250, discount_factor=0.99, val_freq=100)

    # Start training the network
    trainer.train()

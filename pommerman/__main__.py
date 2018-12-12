#!/usr/bin/bash

from train import PolicyTraining
from train.reward import BlockReward
from neuralnet.convolutional import ConvNet
from environment import stopEnv, randomEnv, simpleEnv

if __name__ == '__main__':
    # Chose a neural network to train
    net = ConvNet()

    print("Trainable params: {}".format(net.num_trainable_params()))

    # Set up an environment
    env = simpleEnv(static=True)

    # Chose a reward function to tain with (or None for default reward function)
    r = None

    # Initialize a trainer
    trainer = PolicyTraining(env, net, num_episodes=10, val_freq=2, discount_factor=0.97, visualize=False, reward=r)

    # Start training the network
    trainer.train()

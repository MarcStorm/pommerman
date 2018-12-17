import matplotlib.pyplot as plt
import numpy as np
import json


plt.style.use('seaborn-darkgrid')

num_episodes = 150000

def calc_epsilon(d):
    data = []
    epsilon = 1.0
    for i in range(num_episodes):
        data.append(epsilon)
        epsilon *= num_episodes / (i / (num_episodes / d) + num_episodes)  # decrease epsilon
    return data


def show_plot(d):
    data = calc_epsilon(d)
    w, h = plt.figaspect(1)
    fig = plt.figure(figsize=(w*0.5,h*0.5))
    plt.plot(range(num_episodes), data, color='blue', label='Epsilon')
    plt.xlabel('Iterations')
    plt.ylabel('Epsilon')
    plt.xticks([])
    plt.show()

    fig.savefig("epsilon_{}.pdf".format(d), bbox_inches='tight', transparent=True)

show_plot(20)
show_plot(8)

import matplotlib.pyplot as plt
import numpy as np
import json


plt.style.use('seaborn-darkgrid')

data = []
num_episodes = 150000
epsilon = 1.0
for i in range(num_episodes):
    data.append(epsilon)
    epsilon *= num_episodes / (i / (num_episodes / 8) + num_episodes)  # decrease epsilon


w, h = plt.figaspect(0.35)
fig = plt.figure(figsize=(w*0.5,h*0.5))
plt.plot(range(num_episodes), data, color='blue', label='Epsilon')

plt.legend(loc='lower left')
plt.xlabel('Iterations')
plt.ylabel('Epsilon')
plt.xticks([])
plt.show()

fig.savefig("epsilon_8.pdf", bbox_inches='tight', transparent=True)
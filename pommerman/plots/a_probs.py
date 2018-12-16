import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch.nn.functional as F
import math
import numpy as np
import json

plt.style.use('seaborn-darkgrid')

f = open('a_probs_list_softmax.json', 'r')
data = json.loads(f.readline())

fig = plt.figure()
colors = ['green', 'blue', 'purple', 'orange', 'red', 'cyan']
labels = ['Stop', 'Up', 'Down', 'Left', 'Right', 'Bomb']
for l, c, i in zip(labels, colors, range(0, 6)):
    x = [math.floor(i*16.6667) for i in range(0, len(data))]
    y = [d[i] for d in data]
    plt.plot(x, y, color=c, label=l)

plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.show()

fig.savefig("a_probs.pdf", bbox_inches='tight', transparent=True)
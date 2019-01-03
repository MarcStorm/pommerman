import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch.nn.functional as F
import math
import numpy as np
import json


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


plt.style.use('seaborn-darkgrid')

f = open('./resources/training_rewards_softmax.json', 'r')
train = json.loads(f.readline())
f.close()

f = open('./resources/validation_rewards_softmax.json', 'r')
val = json.loads(f.readline())
f.close()

print(len(train))
print(len(val))


fig = plt.figure()
y = movingaverage(train, 2500)
plt.plot(range(0, len(y)), y, color='blue', label='Training')
n = 50
y = [sum(val[i:i + n])/50 for i in range(0, len(val), n)]
plt.plot([i*25000+25000 for i in range(0, len(y))], y, color='green', label='Validation')

plt.legend(loc='upper left')
plt.ylim(-1, 1) 
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.show()

fig.savefig("train_val.pdf", bbox_inches='tight', transparent=True)
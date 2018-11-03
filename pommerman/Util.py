import numpy as np
import torch


# Flattens a state s on the form list<dict> where each dict contains information of a state
def flatten_state(s):
    # Usage Example:
	# def forward(self, x):
	#     x = flatten_state(x)
	# where x is np.atleast1d(S[0])
	return torch.from_numpy(np.array([flatten_state_aux(x) for x in s])).float()
	

def flatten_state_aux(s):
    # Lists
    alive = [1 if x in s['alive'] else 0 for x in range(10,14)]
    board = s['board']
    bomb_blast_strength = s['bomb_blast_strength']
    bomb_life = s['bomb_life']
    # Tuples
    position = s['position']
    # Ints
    blast_strength = s['blast_strength']
    can_kick = s['can_kick']
    ammo = s['ammo']
    # Enums
    teammate = s['teammate'] #9 for FFA
    enemies = s['enemies'] #11,12,13 for FFA and training agent id = 0
    
    a = np.append(np.array(alive),np.array(board).flatten())
    a = np.append(a,np.array(bomb_blast_strength).flatten())
    a = np.append(a,np.array(bomb_life).flatten())
    a = np.append(a,position[0])
    a = np.append(a,position[1])
    a = np.append(a,blast_strength)
    a = np.append(a,can_kick)
    a = np.append(a,ammo)
    # Commented out as we get size 376 but expected 372. I assume we calculated wrong.
    # Makes sense to ignore these imo
    #a = np.append(a,teammate.value)
    #a = np.append(a,[e.value for e in enemies])
    return a.astype(float)
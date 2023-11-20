import matplotlib.pyplot as plt
import numpy as np
import random

from ringNetwork import RingNetwork
from tqdm import trange

fig, axs = plt.subplots(3,3, figsize=(32, 16))

ax1 = [axs[0,0], axs[1,0], axs[2,0]]
ax2 = [axs[0,1], axs[1,1], axs[2,1]]
ax3 = [axs[0,2], axs[1,2], axs[2,2]]

fig.set_figheight(9)
fig.set_figwidth(9)

t = 100
n_runs = 100

im_max = 1
im_min = -1

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    res = np.array([rn.simulate(seed=random.randint(0,65536)) for _ in trange(n_runs)])

    Vs = res[:,0,:,:]
    Ns = res[:,1,:,:]

    Va = np.mean(Vs, axis=0)

    Nbs = np.sum(Ns[:,-t:,:], axis=1)
    Nba = np.mean(Nbs, axis=0)
    Z = Nbs - Nba

    Ss = Z[:,None,:]*Z[:,:,None]
    Sa = np.mean(Ss, axis=0)

    ax[0].imshow(Sa)
    ax[0].set_xlabel("Neuron")
    ax[0].set_ylabel("Neuron")
    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")

    ax[1].plot(Nba)
    ax[1].set_xlabel("Neuson")
    ax[1].set_ylabel("$\\langle\\bar n_i\\rangle$")

    ax[2].imshow(
        Va, 
        interpolation='nearest', 
        aspect='auto', 
        origin='lower')

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

plt.tight_layout()
fig.subplots_adjust(top=0.9)

fig.suptitle(f"Firring Rate Correlation", fontsize=16)
plt.savefig("q3.png", dpi=600)

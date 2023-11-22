import matplotlib.pyplot as plt
import numpy as np
import random

from ringNetwork import RingNetwork
from tqdm import trange

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(9, 3))


t = 100
n_runs = 1000

im_max = 1
im_min = -1

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    res = np.array([rn.simulate(seed=random.randint(0,65536)) for _ in trange(n_runs)])

    Ns = res[:,1,:,:]
    Nbs = np.sum(Ns[:,-t:,:], axis=1)
    Na = np.mean(Ns, axis=0)
    Nba = np.mean(Nbs, axis=0)
    Z = Nbs - Nba[None,:]

    Ss = Z[:,None,:]*Z[:,:,None]

    #zero_ignore = np.logical_and((Nbs != 0)[:,None,:], (Nbs != 0)[:,:,None])
    #Sa = np.mean(Ss, axis=0, where=zero_ignore)
    #Sa = np.nan_to_num(Sa, np.mean(Sa))

    Sa = np.mean(Ss, axis=0)

    im = ax.imshow(Sa, cmap='plasma')
    ax.set_ylabel("Neuron Tuning ($s_i$)")
    ax.set_yticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax.set_xlabel("Neuron Tuning ($s_i$)")
    ax.set_xticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax.set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im, shrink=0.7)

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.subplots_adjust(top=0.9)

fig.suptitle(f"Firring Rate Correlation (runs={n_runs})", fontsize=16)
plt.tight_layout()

plt.savefig("q3.png", dpi=600)

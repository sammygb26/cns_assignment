import matplotlib.pyplot as plt
import numpy as np
import random

from ringNetwork import RingNetwork
from tqdm import trange

fig, axs = plt.subplots(3,3, figsize=(9, 9))

ax1 = [axs[0,0], axs[1,0], axs[2,0]]
ax2 = [axs[0,1], axs[1,1], axs[2,1]]
ax3 = [axs[0,2], axs[1,2], axs[2,2]]


t = 100
n_runs = 100

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

    im = ax[0].imshow(Sa, cmap='plasma')
    ax[0].set_ylabel("Neuron Tuning ($s_i$)")
    ax[0].set_yticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[0].set_xlabel("Neuron Tuning ($s_i$)")
    ax[0].set_xticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im, shrink=0.7)

    ax[1].plot(Nba)
    ax[1].set_xlim((0, rn.N))
    ax[1].set_xticks([0, rn.N // 2, rn.N])
    ax[1].set_ylabel("$\\langle\\bar n_i\\rangle$")
    ax[1].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)

    ax[2].imshow(
        Na, 
        interpolation='nearest', 
        aspect='auto', 
        origin='lower',
        cmap='plasma')
    ax[2].set_xlabel("Neuron Tuning ($s_i$)")
    ax[2].set_xticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[2].set_ylabel("Time (ms)")

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.subplots_adjust(top=0.9)

fig.suptitle(f"Firring Rate Correlation (runs={n_runs})", fontsize=16)
plt.tight_layout()

plt.savefig("q3.png", dpi=600)

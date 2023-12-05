import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np
from ringNetwork import RingNetwork

fig, axs = plt.subplots(4,3, figsize=(12, 6))

ax1 = [axs[0,0], axs[1,0], axs[2,0], axs[3,0]]
ax2 = [axs[0,1], axs[1,1], axs[2,1], axs[3,1]]
ax3 = [axs[0,2], axs[1,2], axs[2,2], axs[3,2]]

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    V, N, R, I = rn.simulate(verbose=True)

    ytick = ["$-\\pi$", "$0$", "$\\pi$"]

    im1 = ax[0].imshow(V.T, aspect='auto', interpolation='nearest', cmap='plasma')
    ax[0].set_ylabel("$s_i$")
    ax[0].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")
    ax[0].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im1, ax=ax[0], label="$V_i$")

    im2 = ax[1].imshow(R.T, aspect='auto', interpolation='nearest', cmap='plasma')
    ax[1].set_ylabel("$s_i$")
    ax[1].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[1].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im2, ax=ax[1], label="$r_i$")

    Nc = N.copy()
    Nc[Nc == 0] = np.nan
    cmap = mpl.cm.plasma
    cmap.set_bad('black',1.)
    im3 = ax[2].imshow(Nc.T, vmin=0, aspect='auto', cmap=cmap)
    ax[2].set_ylabel("$s_i$")
    ax[2].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[2].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im3, ax=ax[2], label="$n_i$")

    im3 = ax[3].imshow(I.T, aspect='auto', interpolation='nearest', cmap='plasma')
    ax[3].set_xlabel("Time (ms)")
    ax[3].set_ylabel("$s_i$")
    ax[3].set_yticks([0, rn.N // 2, rn.N], ytick)
    fig.colorbar(im3, ax=ax[3], label="$in$")


simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.suptitle("Neuron Voltage ($V_i$), Firing Rate ($r_i$), Spike Counts ($n_i$) and Forward Input Over Time", fontsize=12)
fig.tight_layout() 
plt.savefig(f"q1.png", dpi=800)
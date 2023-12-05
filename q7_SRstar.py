from ringNetwork import RingNetwork

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2,3, figsize=(9, 4))

def simulate(W0, W1, min_weight, ax, regime):
    rn = RingNetwork(100, W0, W1)
    rn.W = np.maximum(min_weight, rn.W)
    V, N, R, _ = rn.simulate(verbose=True)

    im1 = ax[0].imshow(V.T, aspect='auto', interpolation='nearest', cmap='plasma')
    ax[0].set_ylabel("Neuron Tuning ($s_i$)")
    ax[0].set_yticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[0].set_title(regime)
    ax[0].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im1, label="$V_i$")

    im2 = ax[1].imshow(R.T, aspect='auto', interpolation='nearest', cmap='plasma')
    ax[1].set_ylabel("Neuron Tuning ($s_i$)")
    ax[1].set_title(regime)
    ax[1].set_yticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[1].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im2,label="$r_i$")

    Nc = N.copy()
    Nc[Nc == 0] = np.nan
    cmap = mpl.cm.plasma
    cmap.set_bad('black',1.)
    im3 = ax[2].imshow(Nc.T, aspect='auto', vmin=0, cmap=cmap)
    ax[2].set_ylabel("Neuron Tuning ($s_i$)")
    ax[2].set_yticks([0, rn.N // 2, rn.N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[2].set_title(regime)
    ax[2].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
    fig.colorbar(im3,label="$r_i$")


simulate(-10, 11, -21, ax1, "SR")
simulate(-10, 11, -1, ax2, "SR${}^*$")

fig.suptitle("Comparison of SR and SR${}^*$", fontsize=16)
fig.tight_layout()

plt.savefig("q7.png", dpi=800)

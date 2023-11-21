import matplotlib.pyplot as plt
from ringNetwork import RingNetwork

fig, axs = plt.subplots(4,3, figsize=(24, 9))

ax1 = [axs[0,0], axs[1,0], axs[2,0], axs[3,0]]
ax2 = [axs[0,1], axs[1,1], axs[2,1], axs[3,1]]
ax3 = [axs[0,2], axs[1,2], axs[2,2], axs[3,2]]

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    V, N, R, I = rn.simulate(verbose=True)

    ytick = ["$-\\pi$", "$0$", "$\\pi$"]

    im1 = ax[0].imshow(V.T)
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Neuron Tuning ($s_i$)")
    ax[0].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im1, ax=ax[0], aspect=14, shrink=0.6, label="$V_i$", pad=0.01)

    im2 = ax[1].imshow(R.T)
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Neuron Tuning ($s_i$)")
    ax[1].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[1].set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im2, ax=ax[1], aspect=14, shrink=0.6, label="$r_i$", pad=0.01)

    im3 = ax[2].imshow(N.T)
    ax[2].set_xlabel("Time (ms)")
    ax[2].set_ylabel("Neuron Tuning ($s_i$)")
    ax[2].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[2].set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im3, ax=ax[2], aspect=14, shrink=0.6, label="$n_i$", pad=0.01)

    im3 = ax[3].imshow(I.T)
    ax[3].set_xlabel("Time (ms)")
    ax[3].set_ylabel("Neuron Tuning ($s_i$)")
    ax[3].set_yticks([0, rn.N // 2, rn.N], ytick)
    ax[3].set_title(f"$W_0={W0}$ $W_1={W1}$")
    fig.colorbar(im3, ax=ax[3], aspect=14, shrink=0.6, label="${\\bf u}(t)+\sqrt{\\frac{\\tau}{\\delta t}}\\sigma{\\bf q}(t)$", pad=0.01)


simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.tight_layout() 
fig.suptitle("Neuron Voltage ($V_i$), Firing Rate ($r_i$) and Spike Counts ($n_i$) Over Time", fontsize=16)
plt.savefig(f"q1.png", dpi=600)
import matplotlib.pyplot as plt
from ringNetwork import RingNetwork

fig, (ax1, ax2, ax3) = plt.subplots(3)

fig.set_figheight(9)
fig.set_figwidth(9)

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    V = rn.simulate(verbose=True)

    ax.imshow(V.T)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")
    ax.set_title(f"$W_0={W0}$ $W_1={W1}$")

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)
    
plt.savefig(f"q1.png", dpi=600)
import matplotlib.pyplot as plt
import numpy as np
import colormaps as cmaps

from ringNetwork import RingNetwork
from tqdm import trange

fig, axs = plt.subplots(2,4, figsize=(16, 9))

ax1 = [axs[0,0], axs[1, 0]]
ax2 = [axs[0,1], axs[1, 1]]
ax3 = [axs[0,2], axs[1, 2]]
ax4 = [axs[0,3], axs[1, 3]]

t = 400
Dt = 100
n_runs = 400

def plot_Nt_Na(Nt, Na, ax, N):
    colors = cmaps.matter.discrete(500 // Dt + 4).colors

    for i in range(500 // Dt):
        Nt = np.mean(Na[i*Dt:i*Dt+Dt,:], axis=0)
        ax[0].plot(Nt / 0.001, label=f"${i*Dt}-{i*Dt+Dt}$", c=colors[2+i])
        ax[0].set_ylabel("$\\langle n/\\delta t\\rangle$")
        ax[0].tick_params(axis='x', left = False, right = False , labelleft = False , 
                labelbottom = False)
        ax[0].legend()

    ax[1].imshow(Na / 0.001, interpolation='nearest', aspect='auto', origin='lower', cmap='plasma') # Need to average over window
    ax[1].set_xticks([0, N // 2, N], ["$-\\pi$", "$0$", "$\\pi$"])
    ax[1].set_ylabel("Time (ms)")

def simulate(W0, W1, ax, axx):
    rn = RingNetwork(100, W0, W1)
    Ns = np.array([rn.simulate()[1] for _ in trange(n_runs)])

    Na = np.mean(Ns, axis=0)
    Nt = np.mean(Na[t:t+Dt, :], axis=0)

    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")
    plot_Nt_Na(Nt, Na, ax, rn.N)

    return Nt, Na
 
Nt1, Na1 = simulate(0, 0, ax1, ax4)
Nt2, Na2 = simulate(-4, 0, ax2, ax4)
Nt3, Na3 = simulate(-10, 11, ax3, ax4)

M = max(np.block([Nt1, Nt2, Nt3]))
Nta = np.array([Nt1 / max(Nt1), Nt2 / max(Nt2), Nt3 / max(Nt3)])
Nta = np.moveaxis(Nta, 0, -1)

Naa = np.array([Na1, Na2, Na3])
Naa = np.moveaxis(Naa, 0, -1)

plot_Nt_Na(Nt1, Naa, ax4, 100)

ax4[0].cla()
ax4[0].plot(Nt1 / 0.001, c='r', label='F')
ax4[0].plot(Nt2 / 0.001, c='g', label='UI')
ax4[0].plot(Nt3 / 0.001, c='b', label='SR')
ax4[0].legend()

fig.suptitle(f"From t={t}ms to t={t + Dt}ms, {n_runs} repetitions", fontsize=16)
fig.tight_layout()
plt.savefig("q2.png", dpi=600)

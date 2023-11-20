import matplotlib.pyplot as plt
import numpy as np
import random

from ringNetwork import RingNetwork
from tqdm import trange

fig, ax = plt.subplots()

fig.set_figheight(9)
fig.set_figwidth(9)

t = 300
Dt = 200
n_runs = 40

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    Vs = np.array([rn.simulate(seed=random.randint(0,65536))[0] for _ in trange(n_runs)])

    Va = np.mean(Vs, axis=0)
    V_tune = np.mean(Va[t:t+Dt, :], axis=0)

    ax.plot(V_tune, label=f"$W_0={W0}$ $W_1={W1}$")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("V")
    

simulate(0, 0, ax)
simulate(-4, 0, ax)
simulate(-10, 11, ax)

fig.suptitle(f"From t={t}ms to t={t + Dt}ms, {n_runs} repetitions", fontsize=16)
plt.tight_layout()
plt.savefig("q2.png", dpi=600)

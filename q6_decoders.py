import matplotlib.pyplot as plt

from ringNetwork import RingNetwork
from decoders import *

fig, axs = plt.subplots(2,3, figsize=(16, 8))

ax1 = [axs[0,0], axs[1,0]]
ax2 = [axs[0,1], axs[1,1]]
ax3 = [axs[0,2], axs[1,2]]

window = 50

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)

    _, N, _, _ = rn.simulate(verbose=True)

    Nc = get_cumulative_counts(N, window=25, remove_zeros=True)

    wta = rad2deg(winner_take_all_decode(Nc, rn.s))
    pv = rad2deg(population_vector_decode(Nc, rn.s))

    wta_cmse = get_mse(wta, 0)
    pv_cmse = get_mse(pv, 0)

    ax[0].plot(wta_cmse, label="Winner Take All")
    ax[0].plot(pv_cmse, label="Population Vector")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("MSE (deg${}^2$)")
    ax[0].legend()

    ax[1].plot(wta, label="Winner Take All")
    ax[1].plot(pv, label="Population Vector")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Error")
    ax[1].legend()

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.suptitle(f"Cumulative MSE For Decoded Value", fontsize=16)
plt.savefig("q6.png", dpi=600)    

    
import matplotlib.pyplot as plt

from ringNetwork import RingNetwork
from decoders import *

fig, axs = plt.subplots(2,3, figsize=(16, 8))

ax1 = [axs[0,0], axs[1,0]]
ax2 = [axs[0,1], axs[1,1]]
ax3 = [axs[0,2], axs[1,2]]

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)

    _, N, _, _ = rn.simulate(verbose=True)

    Nc = get_cumulative_counts(N, remove_zeros=True)

    wta = winner_take_all_decode(Nc, rn.s)
    pv = population_vector_decode(Nc, rn.s, angle_multipltier=2)

    wta_mse = get_mse(rad2deg(wta), 0)
    pv_mse = get_mse(rad2deg(pv), 0)

    ax[0].plot(wta_mse, label="Winner Take All")
    ax[0].plot(pv_mse, label="Population Vector")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("MSE (deg${}^2$)")
    ax[0].legend()

    ax[1].plot(wta, label="Winner Take All")
    ax[1].plot(pv, label="Population Vector")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Error (deg)")
    ax[1].legend()

simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.suptitle(f"Cumulative MSE For Decoded Value", fontsize=16)
plt.savefig("q4.png", dpi=600)    

    
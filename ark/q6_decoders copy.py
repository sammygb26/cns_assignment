import matplotlib.pyplot as plt

from tqdm import trange
from ringNetwork import RingNetwork
from decoders import *

fig, axs = plt.subplots(2,3, figsize=(9, 5))

n_runs = 100

ax1 = [axs[0,0], axs[1,0]]
ax2 = [axs[0,1], axs[1,1]]
ax3 = [axs[0,2], axs[1,2]]

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)
    res = np.array([rn.simulate() for _ in trange(n_runs)])


    Ns = res[:,1,:,:]
    _, N, _, _ = rn.simulate(verbose=True)

    Nc = get_cumulative_counts(N, remove_zeros=True, window=25)
    Ncs = get_cumulative_counts(Ns, remove_zeros=True, axis=1, window=25)

    wta = winner_take_all_decode(Nc, rn.s)
    wtas = winner_take_all_decode(Ncs, rn.s)

    pv = population_vector_decode(Nc, rn.s)
    pvs = population_vector_decode(Ncs, rn.s)

    nonzeros = np.sum(Ncs, axis=-1) != 0

    wta_mse = get_mse(rad2deg(wta), 0)
    wtas_mse = np.mean(get_mse(rad2deg(wtas), 0), axis=1, where=nonzeros)

    pv_mse = get_mse(rad2deg(pv), 0)
    pvs_mse = np.mean(get_mse(rad2deg(pvs), 0), axis=1, where=nonzeros)

    ax[0].plot(wta_mse, label="Winner Take All")
    ax[0].plot(pv_mse, label="Population Vector")
    ax[0].set_title(f"$W_0={W0}$ $W_1={W1}$")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("MSE (deg${}^2$)")
    ax[0].legend()

    ax[1].plot(wtas_mse, label="Winner Take All")
    ax[1].plot(pvs_mse, label="Population Vector")
    ax[1].set_xlabel("Time (ms)")
    ax[1].set_ylabel("Average MSE (deg${}^2$)")
    ax[1].legend()




simulate(0, 0, ax1)
simulate(-4, 0, ax2)
simulate(-10, 11, ax3)

fig.suptitle(f"MSE For Decodes (runs={n_runs})", fontsize=16)
fig.tight_layout()
plt.savefig("q6.png", dpi=600)    

    
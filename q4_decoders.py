import matplotlib.pyplot as plt

from tqdm import trange
from ringNetwork import RingNetwork
from decoders import *

fig = plt.figure(figsize=(9,6))

ax1 = fig.add_subplot(234)
ax2 = fig.add_subplot(235)
ax3 = fig.add_subplot(236)

ax4 = fig.add_subplot(231)
ax5 = fig.add_subplot(232)

n_runs = 200

def simulate(W0, W1, ax, regime):
    rn = RingNetwork(100, W0, W1)
    res = np.array([rn.simulate() for _ in trange(n_runs)])

    Ns = res[:,1,:,:]
    Ncs = get_cumulative_counts(Ns, remove_zeros=True, axis=1)[10:]

    wta = winner_take_all_decode(Ncs, rn.s)
    pv = population_vector_decode(Ncs, rn.s)

    nonzeros = np.sum(Ncs, axis=-1) != 0
    wta_mse = get_mse(rad2deg(wta), 0)
    wta_msea = np.mean(wta_mse, axis=1, where=nonzeros)
    wta_lower_error = np.std(wta_mse, axis=1, where=wta_mse < wta_msea[:,None])
    wta_upper_error = np.std(wta_mse, axis=1, where=wta_mse > wta_msea[:,None])

    pv_mse = get_mse(rad2deg(pv), 0)
    pv_msea = np.mean(pv_mse, axis=1, where=nonzeros)
    pv_lower_error = np.std(pv_mse, axis=1, where=pv_mse < pv_msea[:,None])
    pv_upper_error = np.std(pv_mse, axis=1, where=pv_mse > pv_msea[:,None])

    ax.plot(wta_msea, label="Winner Take All")
    ax.fill_between(np.array([i for i in range(len(wta_mse))]), 
                    wta_msea-wta_lower_error, wta_msea+wta_upper_error,
                    alpha=.1)

    ax.plot(pv_msea, label="Population Vector")
    ax.fill_between(np.array([i for i in range(len(pv_mse))]), 
                    pv_msea-pv_lower_error, pv_msea+pv_upper_error,
                    alpha=.1)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Average MSE (deg${}^2$)")
    ax.set_title(regime)
    ax.legend()

    ax4.plot(wta_msea, label=regime)
    ax4.fill_between(np.array([i for i in range(len(wta_mse))]), 
                    wta_msea-wta_lower_error, wta_msea+wta_upper_error,
                    alpha=.1)
    ax5.plot(pv_msea, label=regime)
    ax5.fill_between(np.array([i for i in range(len(pv_mse))]), 
                    pv_msea-pv_lower_error, pv_msea+pv_upper_error,
                    alpha=.1)

simulate(0, 0, ax1, "F")
simulate(-4, 0, ax2, "UI")
simulate(-10, 11, ax3, "SR")

ax4.set_title("Winner Take All")
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Average MSE (deg${}^2$)")
ax4.legend()

ax5.set_title("Population Vector")
ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Average MSE (deg${}^2$)")
ax5.legend()

fig.suptitle(f"MSE For Decodes With Cumulative Counts (runs={n_runs})", fontsize=16)
fig.tight_layout()
plt.savefig("q4.png", dpi=600)    

    
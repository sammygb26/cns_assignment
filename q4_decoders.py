import matplotlib.pyplot as plt

from ringNetwork import RingNetwork
from decoders import *
from tqdm import trange

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

def simulate(W0, W1, ax):
    rn = RingNetwork(100, W0, W1)

    V, N = rn.simulate()

    Nc = get_cumulative_counts(N)

    wta = winner_take_all_decode(Nc, rn.s)
    pv = population_vector_decode(Nc, rn.s, angle_multipltier=2)

    ax.plot(wta, label="Winner Take All")
    ax.plot(pv, label="Population Vector")
    


    
import numpy as np

from ringNetwork import RingNetwork

def get_cumulative_counts(N):
    return np.array([np.sum(N[:i+1,:]) for i in range(N.shape[0])])

def winner_take_all_decode(N, s):
    return np.array([s[np.argmax(N[i,:])] for i in range(N.shape[0])])

def population_vector_decode(N, s):
    def population_deode_single(Ns):
        angles = np.exp(1j * s)
        v = np.sum([angles[i] * Ns[i] for i in len(N)])
        return np.imag(np.log(v))

    return np.array([population_deode_single(N[i,:]) for i in range(N.shape[0])])
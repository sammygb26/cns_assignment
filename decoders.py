import numpy as np

from ringNetwork import RingNetwork

def get_mse(pred, act):
    return np.power(pred - act, 2)

def get_cumulative_mse(pred, act):
    mse = get_mse(pred, act)
    return np.array([np.sum(mse[:i+1]) for i in range(len(pred))])

def get_cumulative_counts(N):
    return np.array([np.sum(N[:i+1,:], axis=0) for i in range(N.shape[0])])

def winner_take_all_decode(N, s):
    return np.array([s[np.argmax(N[i,:])] for i in range(N.shape[0])])

def population_vector_decode(N, s, angle_multipltier = 1):
    def population_deode_single(Ns):
        angles = np.exp(1j * s * angle_multipltier)
        return np.imag(np.log(np.sum(angles * Ns))) / angle_multipltier

    return np.array([population_deode_single(N[i,:]) for i in range(N.shape[0])])
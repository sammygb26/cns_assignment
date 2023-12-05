import numpy as np

def rad2deg(x):
    return (x / np.pi) * 180

def get_rmse(pred, act):
    return np.abs(pred - act)

def get_cumulative_mse(pred, act):
    mse = get_rmse(pred, act)
    return np.array([np.sum(mse[:i+1]) for i in range(len(pred))])

def get_cumulative_counts(N, window=0, remove_zeros=False, axis=0):
    N = np.moveaxis(N, axis, 0)

    def cumulate(N, i):
        start = 0 if window == 0 else max(0, i-window+1)
        return np.sum(N[start:i+1,:], axis=0)

    c = [cumulate(N,i) for i in range(N.shape[0])]
    c = [x for x in c if np.sum(x) != 0 or not remove_zeros]
    c = np.array(c)

    return c


def winner_take_all_decode(N, s, axis=-1):
    N = np.moveaxis(N, axis, 0)
    Np= np.reshape(N, (N.shape[0], len(np.ravel(N[0]))))
    d = np.array([s[np.argmax(Np[:,i])] for i in range(Np.shape[1])])
    d = np.reshape(d, N.shape[1:])
    return d


def population_vector_decode(N, s, angle_multipltier = 1, axis=-1):
    def population_deode_single(Ns):
        angles = np.exp(1j * s * angle_multipltier)
        return np.imag(np.log(np.sum(angles * Ns))) / angle_multipltier

    N = np.moveaxis(N, axis, 0)
    Np= np.reshape(N, (N.shape[0], len(np.ravel(N[0]))))
    d = np.array([population_deode_single(Np[:,i]) for i in range(Np.shape[1])])
    d = np.reshape(d, N.shape[1:])
    return d
import numpy as np

from tqdm import trange

class RingNetwork:
    def __init__(self, N, W0, W1):
        self.N = N
        self.s = np.linspace(-np.pi / 2, np.pi / 2, N+1)[:-1] 

        S = np.repeat(np.reshape(self.s, (N, 1)), N, axis=1)
        self.W = W1 * np.cos(2*(S - S.T)) + W0

    def simulate(self, s=0, dt=1, T=500, V0=0, tau=50, beta=0.1, sigma=0.1, u0=0.5, u1=0.5, seed=1, verbose=False):
        n_timesteps = int(T/dt + 0.5)
        np.random.seed(seed)

        V = np.zeros((n_timesteps + 1, self.N))
        N = np.zeros((n_timesteps + 1, self.N))

        V[0,:] = V0
        u = u0 + u1 * np.cos(2*(s - self.s))

        def act(x):
            return np.max(np.array([np.zeros(x.shape), x * beta]), axis=0)

        for i in trange(1,n_timesteps + 1, desc="Simulating...", disable=not verbose):
            v = V[i-1,:]
            q = np.random.normal(np.zeros(self.N), np.ones(self.N))

            n = np.random.poisson(act(v), v.shape)
            N[i,:] = n

            dv_tdt = -v + self.W @ n + u + np.sqrt(tau / dt) * q * sigma
            V[i,:] = v + dt * dv_tdt / tau

        return V, N


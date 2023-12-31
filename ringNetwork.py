import numpy as np

from tqdm import trange

def exp_model(x):
    x = np.abs(x)
    return 2 * ((1-x/np.pi)*np.exp(-x)) - 1

class RingNetwork:
    def __init__(self, N, W0, W1):
        self.N = N
        self.s = np.linspace(-np.pi, np.pi, N+1)[:-1] 

        self.W = W1 * np.cos(self.s[:,None] - self.s[None,:]) + W0
        #self.W = np.maximum(-0.5, self.W)

                

    def simulate(self, s=0, dt=0.001, T=0.5, V0=0, tau=0.05, beta=0.1, sigma=0.1, u0=0.5, u1=0.5, seed=None, verbose=False):
        n_timesteps = int(T/dt + 0.5)
        if seed is not None:
            np.random.seed(seed)

        V = np.zeros((n_timesteps + 1, self.N))
        N = np.zeros((n_timesteps + 1, self.N))
        R = np.zeros((n_timesteps + 1, self.N))
        I = np.zeros((n_timesteps + 1, self.N))

        V[0,:] = V0
        u = u0 + u1 * np.cos(s - self.s)

        def act(x):
            return beta * np.max(np.array([np.zeros(x.shape), x]), axis=0)

        for i in trange(1,n_timesteps + 1, desc="Simulating...", disable=not verbose):
            v = V[i-1,:]
            q = np.random.normal(np.zeros(self.N), np.ones(self.N))
            r = act(v)
            _in = u + np.sqrt(tau / dt) * q * sigma
            n = np.random.poisson(r * dt * 1e3, v.shape)

            dv_tdt = -v + self.W @ n + _in 

            V[i,:] = v + dt * (dv_tdt / tau)
            N[i,:] = n
            R[i,:] = r
            I[i,:] = _in

        return V, N, R, I


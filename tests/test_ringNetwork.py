from ringNetwork import RingNetwork

import numpy as np

def test_s():
    rn = RingNetwork(100, 0, 0)
    assert max(rn.s) < np.pi
    assert min(rn.s) == -np.pi
    assert len(rn.s) == rn.N

def test_W():
    for _ in range(100):
        W0 = np.random.random(1)
        W1 = np.random.random(1)
        rn = RingNetwork(100, W0, W1)

        assert np.min(rn.W) >= W0 - W1
        assert np.max(rn.W) <= W0 + W1
        assert rn.W[0,0] == W0 + W1
        assert rn.W[5,3] - (W0 + W1 * np.cos(rn.s[5] - rn.s[3])) < 1e-6

def test_W0():
    for _ in range(100):
        W0 = np.random.random(1)
        rn = RingNetwork(100, W0, 0)
        assert np.all(rn.W == W0)

        for _ in range(100):
            rand_n = np.random.randint(0, rn.N)
            assert np.abs(rn.W[rand_n,-1] - rn.W[rand_n,0]) < 0.01
            assert np.abs(rn.W[-1, rand_n] - rn.W[0, rand_n]) < 0.01

def test_sim():
    def t(N, dt, T):
        rn = RingNetwork(N, 0, 0)
        V, _ = rn.simulate(dt=dt, T=T, verbose=True)
        assert V.shape == (int(T/dt + 1.5), N)

    for _ in range(5):
        N = np.random.randint(10, 100)
        dt = np.random.random() * 1e-3
        T = 1 + np.random.random() * 1e-2
        t(N, dt, T)
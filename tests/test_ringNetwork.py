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
        assert rn.W[0,0] == W0 + W1 * np.cos(2 * rn.s[0])
        assert rn.W[5,3] == W0 + W1 * np.cos(rn.s[5] - rn.s[3])

def test_W0():
    for _ in range(100):
        W0 = np.random.random(1)
        rn = RingNetwork(100, W0, 0)
        assert np.all(rn.W == W0)
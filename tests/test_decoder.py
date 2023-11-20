from decoders import *

def test_get_cumulative_counts():
    N = np.ones(10)
    assert np.all(get_cumulative_counts(N[:,None]) == np.arange(1, 11, 1)[:,None])

def test_winner_take_all_decode():
    N = np.eye(10)
    s1 = np.arange(0, 10, 1)

    print(N.shape)
    print(s1.shape)

    assert np.all(winner_take_all_decode(N, s1) == s1)

def test_population_vector_decode():
    N1 = np.eye(10)
    s1 = np.linspace(-np.pi, np.pi, 11)[:-1]

    N2 = np.cos(s1[:,None] - s1[None,:])

    assert np.allclose(population_vector_decode(N1, s1), s1)
    assert np.allclose(population_vector_decode(N2, s1), s1)

    s2 = np.linspace(-np.pi / 2, np.pi / 2, 11)[:-1]
    N3 = np.cos(2 * (s2[:,None] - s2[None,:]))
    assert not np.allclose(population_vector_decode(N3, s2), s2)

    d2 = population_vector_decode(N3, s2, angle_multipltier=2)
    assert np.allclose(d2, s2)

def test_get_mse():
    pred1 = np.zeros(10)
    act2 = 0
    
    assert np.allclose(get_mse(pred1, act2), np.zeros(10))
    assert np.allclose(get_cumulative_mse(pred1, act2), np.zeros(10))

    pred2 = np.ones(10)

    assert np.allclose(get_mse(pred2, act2), np.ones(10))
    assert np.allclose(get_cumulative_mse(pred2, act2), np.arange(1,11,1))

    act3 = 5
    pred3 = np.arange(0,10,1)

    assert np.allclose(get_mse(pred3, act3), np.power(np.arange(-5,5,1),2))
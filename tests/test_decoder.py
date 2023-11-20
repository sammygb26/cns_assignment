from decoders import *

def test_get_cumulative_counts():
    N = np.ones(10)
    assert np.all(get_cumulative_counts(N[:,None]) == np.arange(1, 11, 1))

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

    assert np.all(winner_take_all_decode(N1, s1) == s1)
    assert np.all(winner_take_all_decode(N2, s1) == s1)

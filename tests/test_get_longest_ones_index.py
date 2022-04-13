import numpy as np

from bvas.util import get_longest_ones_index


def test_get_longest_ones_index():

    def test_sequence(x, expected):
        assert np.all(get_longest_ones_index(np.array(x)) == expected)

    test_sequence([1], [0])
    test_sequence([1, 1], [0, 1])
    test_sequence([0, 1], [1])
    test_sequence([1, 0], [0])
    test_sequence([1, 0, 0], [0])
    test_sequence([0, 1, 0], [1])
    test_sequence([0, 0, 1], [2])
    test_sequence([1, 1, 0], [0, 1])
    test_sequence([0, 1, 1], [1, 2])
    test_sequence([1, 0, 1], [2])
    test_sequence([1, 1, 1], [0, 1, 2])
    test_sequence([1, 0, 0, 0], [0])
    test_sequence([0, 0, 0, 1], [3])
    test_sequence([0, 1, 1, 0], [1, 2])
    test_sequence([1, 1, 1, 0], [0, 1, 2])
    test_sequence([0, 1, 1, 1], [1, 2, 3])
    test_sequence([1, 1, 1, 1], [0, 1, 2, 3])

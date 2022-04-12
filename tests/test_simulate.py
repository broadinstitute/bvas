import pytest

from bvas import simulate_data


@pytest.mark.parametrize("A", [10, 20])
@pytest.mark.parametrize("include_phi", [False, True])
@pytest.mark.parametrize("strategy", ['global-mean', 'global-median', 'regional'])
def test_simulate_smoketest(A, include_phi, strategy):
    data = simulate_data(num_alleles=A, include_phi=include_phi, strategy=strategy)
    a = 2 * A if include_phi else A
    assert data['Y'].shape == (a,)
    assert data['Gamma'].shape == (a, a)

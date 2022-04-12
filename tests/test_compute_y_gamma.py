from generate_test_data import get_nb_data

from bvas.simulate import compute_y_gamma


def test_compute_y_gamma(A=10):
    N, genotype = get_nb_data(num_alleles=A, return_N=True)
    compute_y_gamma(N, genotype, 'global-median', 1.0)
    compute_y_gamma(N, genotype, 'global-mean', 1.0)
    compute_y_gamma(N, genotype, 'regional', 1.0)

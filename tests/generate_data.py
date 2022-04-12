import math

import torch
from torch.distributions import Bernoulli, NegativeBinomial


def compute_y_gamma(N, genotype, exact_prefactor):
    X = torch.matmul(N, genotype)  # num_regions duration num_alleles
    num_regions, duration, num_alleles = X.shape
    N_sum = N.sum(-1)  # num_regions duration
    X /= N_sum.unsqueeze(-1)

    XX = torch.stack([torch.einsum("va,vb,tv->tab", genotype, genotype, N_r) for N_r in N])
    XX /= N_sum.unsqueeze(-1).unsqueeze(-1)

    eye = torch.eye(num_alleles)
    Gamma = (1 - eye) * (XX - X[:, :, None, :] * X[:, :, :, None]) + eye * (X * (1 - X)).unsqueeze(-1)
    Gamma = Gamma[:, 0:-1].sum(1)  # R A A
    Y = X[:, -1] - X[:, 0]         # R A

    Y = exact_prefactor * Y.sum(0)
    Gamma = exact_prefactor * Gamma.sum(0)

    assert Y.shape == (num_alleles,)
    assert Gamma.shape == (num_alleles, num_alleles)

    return Y, Gamma


def get_nb_data(num_alleles=20, duration=20, num_variants=10, num_regions=10, N0=500,
                R0=1.0, mutation_density=0.3, beta0=0.02, beta1=0.04, k=0.5, seed=0):

    torch.manual_seed(seed)

    genotype = Bernoulli(mutation_density).sample(sample_shape=(num_variants, num_alleles))
    genotype[0:4, 0] = 1
    genotype[2:6, 1] = 1

    N = N0 * torch.ones(num_regions, duration, num_variants)

    R = R0 + beta0 * genotype[:, 0].float() + beta1 * genotype[:, 1].float()
    logits = R.log() - math.log(k)

    for t in range(1, duration):
        N_prev = N[:, t - 1]
        total_count = k * N_prev
        N[:, t] = NegativeBinomial(total_count=total_count, logits=logits).sample()

    exact_prefactor = (N0 * num_variants) / (1 / k + 1 / R0)

    Y, Gamma = compute_y_gamma(N, genotype, exact_prefactor)

    return Y.double(), Gamma.double()

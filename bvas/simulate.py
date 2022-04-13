import math

import torch
from torch.distributions import Bernoulli, Binomial, Multinomial, NegativeBinomial


def __compute_y_gamma(N, genotype, strategy='global-median', center=False, phi=None):
    """
    Helper function used in _compute_y_gamma below.
    """
    assert strategy in ['global-median', 'global-mean', 'regional']

    X = torch.matmul(N, genotype)
    num_regions, duration, num_alleles = X.shape

    if phi is not None:
        assert phi.shape == (num_regions, duration)

    X = torch.matmul(N, genotype)  # num_regions duration num_alleles
    X /= N.sum(-1).unsqueeze(-1)

    if strategy == 'regional':
        if center:
            denominator = (X[:, :-1] - X[:, 1:]).mean(1, keepdim=True)
            denominator = (X[:, :-1] - X[:, 1:] - denominator).pow(2.0).mean(-1)
        else:
            denominator = (X[:, :-1] - X[:, 1:]).pow(2.0).mean(-1)
        numerator = (X[:, :-1] * (1 - X[:, :-1])).mean(-1)
        nu_eff = (numerator / denominator).mean(-1)

        assert nu_eff.shape == (num_regions,)
        nu_eff = nu_eff.unsqueeze(-1).unsqueeze(-1)

    elif strategy in ['global-mean', 'global-median']:
        if center:
            denominator = (X[:, :-1] - X[:, 1:]).mean(1, keepdim=True)
            denominator = (X[:, :-1] - X[:, 1:] - denominator).pow(2.0).mean(-1)
        else:
            denominator = (X[:, :-1] - X[:, 1:]).pow(2.0).mean(-1)
        numerator = (X[:, :-1] * (1 - X[:, :-1])).mean(-1)

        if strategy == 'global-mean':
            nu_eff = (numerator / denominator).mean()
        elif strategy == 'global-median':
            nu_eff = (numerator / denominator).median()

        assert nu_eff.ndim == 0

    nu_eff_phi = nu_eff if phi is None else nu_eff * phi[:, :-1].unsqueeze(-1)

    XX = (N[:, :-1] * nu_eff_phi / N[:, :-1].sum(-1).unsqueeze(-1)).sum(1)
    XX = torch.einsum("va,vb,rv->ab", genotype, genotype, XX)
    X_nu = X[:, :-1] * nu_eff_phi

    Gamma = (XX - torch.einsum("rta,rtb->ab", X_nu, X[:, :-1]))
    Gamma.diagonal(dim1=-1, dim2=-2).copy_(torch.einsum("rta->a", X_nu * (1 - X[:, :-1])))

    Y = torch.einsum("rta->a", (X[:, 1:] - X[:, :-1]) * nu_eff_phi)

    assert Y.shape == (num_alleles,)
    assert Gamma.shape == (num_alleles, num_alleles)

    return Y, Gamma, nu_eff.flatten().data.cpu().numpy()


def _compute_y_gamma(N, genotype, strategy='global-median', center=False, phi=None):
    """
    Function for computing Y and Gamma from time series of (simulated) variant-level counts.

    :param torch.Tensor N: A `torch.Tensor` of shape (num_regions, duration, num_variants) that specifies
        region-local variant-level time series of non-negative case counts.
    :param torch.Tensor genotype: A binary `torch.Tensor` of shape (num_variants, num_alleles) that specifies
        the genotype of each variant in `N`.
    :param str strategy: Controls the strategy for computing the effective population size from `N` and
        `genotype`. Must be one of: global-mean, global-median, and regional. Defaults to global-median.
    :param bool center: Whether the effective population size estimator should use centering. Defaults
        to False.
    :param torch.Tensor phi: Optional time series of region-specific vaccination frequencies, i.e. expected
        to be between 0 and 1. Has shape (num_regions, duration). Defaults to None.

    :returns tuple: Returns a tuple (Y, Gamma, nu_eff) where Y and Gamma are `torch.Tensor`s, with each scaled
        using the indicated effective population size estimation strategy, and nu_eff is a NumPy array of
        estimated effective population size(s).
    """
    if strategy not in ['global-mean', 'global-median', 'regional']:
        raise ValueError("strategy must be one of: global-mean, global-median, regional.")

    if phi is None:
        return __compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=None)
    else:
        Y, Gamma, nu_eff = __compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=None)
        Y_phi, Gamma_cross, _ = __compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=phi)
        _, Gamma_phi, _ = __compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=phi.pow(2.0))

    A = genotype.size(-1)
    Gamma_full = Gamma.new_zeros(2 * A, 2 * A)

    Gamma_full[:A, :A] = Gamma
    Gamma_full[A:, A:] = Gamma_phi
    Gamma_full[:A, A:] = Gamma_cross
    Gamma_full[A:, :A] = Gamma_cross

    Y_full = torch.cat([Y, Y_phi], dim=-1)

    assert Gamma_full.shape == (2 * A, 2 * A)
    assert Y_full.shape == (2 * A,)

    return Y_full, Gamma_full, nu_eff


def simulate_data(num_alleles=100, duration=26, num_variants=100, num_regions=10,
                  N0=10 ** 4, N0_k=10.0, R0=1.0, mutation_density=0.25,
                  k=0.1, seed=0, include_phi=False, sampling_rate=1, strategy='global-mean'):
    r"""
    Simulate pandemic data using a discrete time Negative Binomial branching process.

    :param int num_alleles: The number of alleles to simulate. Defaults to 100.
    :param int duration: The number of timesteps to simulate. Defaults to 26.
    :param int num_variants: The number of viral variants to simulate. Defaults to 100.
    :param int num_regions: The number of geographic regions to simulate. Defaults to 10.
    :param int N0: The mean number of infected individuals at the first time step in each region.
        Defaults to 10000.
    :param float N0_k: Controls the dispersion of the Negative Binomial distribution that is used
        to sample the number of infected individuals at the first time step in each region.
        Defaults to 10.0.
    :param float R0: The basic reproduction number of the wild-type variant. Defaults to 1.0.
    :param float mutation_density: Controls the average number of non-wild-type mutations that
        appear in each viral variant. Defaults to 0.25.
    :param float k: Controls the dispersion of the Negative Binomial distribution that underlies
        the discrete time branching process. Defaults to 0.1.
    :param int seed: Sets the random number seed. Defaults to 0.
    :param bool include_phi: Whether to include vaccine-dependent effects in the simulation. Defauls to False.
    :param float sampling_rate: Controls the observation sampling rate, i.e. the percentage of
        infected individuals whose genomes are sequenced. Defaults to 1, i.e. 1%.
    :param str strategy: Strategy used for estimating the effective population size. Must be
        one of: global-mean, global-median, regional. Defaults to global-mean.

    :returns dict: returns a dictionary that contains Y and Gamma as well as the estimated
        effective population size. Y and Gamma are each scaled
        using the indicated effective population size estimation strategy.
    """

    torch.manual_seed(seed)

    genotype = Bernoulli(mutation_density).sample(sample_shape=(num_variants, num_alleles))

    # 10 non-neutral alleles
    betas = torch.tensor([0.01, 0.02, 0.04, 0.06, 0.08, -0.01, -0.02, -0.04, -0.06, -0.08]).double()
    # 10 non-neutral vaccine-dependent alleles (if include_phi=True)
    phi_betas = torch.tensor([0.01, 0.02, 0.04, 0.06, 0.08, -0.01, -0.02, -0.04, -0.06, -0.08]).double()

    if include_phi:
        true_betas = torch.cat([betas,
                                torch.zeros(2 * num_alleles - betas.shape[0] - phi_betas.shape[0]).double(),
                                phi_betas])
    else:
        true_betas = torch.cat([betas, torch.zeros(num_alleles - betas.shape[0]).double()])

    # sample initial number of infected individuals in each region
    N0_dist = NegativeBinomial(total_count=N0_k, logits=math.log(N0) - math.log(N0_k))
    N0 = N0_dist.sample(sample_shape=(num_regions,))
    N = torch.zeros(num_regions, duration, num_variants)
    for r in range(num_regions):
        N[r, 0, :] = Multinomial(total_count=int(N0[r].item()), logits=torch.zeros(num_variants)).sample()

    # compute R for each variant
    R = R0 + torch.mv(genotype[:, :betas.shape[0]].double(), betas)
    logits = R.log() - math.log(k)

    phi = None

    # simulate discrete brancing process forward in time
    if not include_phi:
        for t in range(1, duration):
            N_prev = N[:, t - 1]
            total_count = k * N_prev
            N[:, t] = NegativeBinomial(total_count=total_count, logits=logits).sample()
    else:
        # assume vaccination rate starts at zero and increases linearly to a random value between 0.5 and 1.0
        phi = 0.5 + 0.5 * torch.rand(num_regions)
        phi = torch.linspace(0.0, 1.0, duration) * phi[:, None]
        for t in range(1, duration):
            N_prev = N[:, t - 1]
            total_count = k * N_prev
            R_t = R + phi[:, t].unsqueeze(-1) * torch.mv(genotype[:, -phi_betas.shape[0]:].double(), phi_betas)
            logits_t = R_t.log() - math.log(k)
            N[:, t] = NegativeBinomial(total_count=total_count, logits=logits_t).sample()

    # downsample observed counts to account for sampling_rate
    if sampling_rate < 100:
        N = Binomial(total_count=N, probs=sampling_rate / 100.0).sample()

    Y, Gamma, nu_eff = _compute_y_gamma(N, genotype, strategy=strategy, center=0, phi=phi)

    data = {'Y': Y}
    data['Gamma'] = Gamma
    data['estimated_nu_eff'] = nu_eff
    data['true_betas'] = true_betas

    return data

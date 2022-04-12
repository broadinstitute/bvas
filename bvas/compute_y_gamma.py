import torch


def _compute_y_gamma(N, genotype, strategy='global-median', center=False, phi=None):
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


def compute_y_gamma(N, genotype, strategy='global-median', center=False, phi=None):
    """
    Function for computing Y and Gamma from time series of variant-level counts.

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
    """
    if strategy not in ['global-mean', 'global-median', 'regional']:
        raise ValueError("strategy must be one of: global-mean, global-median, regional.")

    if phi is None:
        return _compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=None)
    else:
        Y, Gamma, nu_eff = _compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=None)
        Y_phi, Gamma_cross, _ = _compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=phi)
        _, Gamma_phi, _ = _compute_y_gamma(N, genotype, strategy=strategy, center=center, phi=phi.pow(2.0))

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

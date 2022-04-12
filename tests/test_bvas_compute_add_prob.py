import math
from types import SimpleNamespace

import pytest
import torch
from common import assert_close
from generate_data import get_nb_data
from torch import zeros

from bvas import BVASSampler


def get_sample(gamma, log_h_ratio, num_included_alleles):
    A = len(gamma)
    sample = SimpleNamespace(gamma=torch.tensor(gamma).bool(),
                             add_prob=zeros(A), _i_prob=zeros(A),
                             _idx=0, weight=0.0)
    sample._active = torch.nonzero(sample.gamma).squeeze(-1)
    sample._log_h_ratio = log_h_ratio
    if num_included_alleles > 0:
        sample._activei = torch.cat([sample._active, torch.arange(A, A + num_included_alleles, dtype=torch.int64)])
    return sample


def check_gammas(sampler, A, compute_log_factor_ratio, num_included_alleles):
    # TEST GAMMA = 0 0 0
    sample = get_sample([0] * A, sampler.log_h_ratio, num_included_alleles)
    log_odds = sampler._compute_add_prob(sample)
    for a in range(A):
        assert_close(compute_log_factor_ratio([a], []), log_odds[a], atol=1.0e-7)

    # TEST GAMMA = 1 0 0
    sample = get_sample([1] + [0] * (A - 1), sampler.log_h_ratio, num_included_alleles)
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0], []), log_odds[0], atol=1.0e-7)
    for a in range(1, A):
        assert_close(compute_log_factor_ratio([0, a], [0]), log_odds[a], atol=1.0e-7)

    # TEST GAMMA = 1 1 0
    sample = get_sample([1, 1] + [0] * (A - 2), sampler.log_h_ratio, num_included_alleles)
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1], [1]), log_odds[0], atol=1.0e-7)
    assert_close(compute_log_factor_ratio([0, 1], [0]), log_odds[1], atol=1.0e-7)
    for a in range(2, A):
        assert_close(compute_log_factor_ratio([0, 1, a], [0, 1]), log_odds[a], atol=1.0e-7)

    # TEST GAMMA = 1 1 1
    sample = get_sample([1, 1, 1] + [0] * (A - 3), sampler.log_h_ratio, num_included_alleles)
    log_odds = sampler._compute_add_prob(sample)

    assert_close(compute_log_factor_ratio([0, 1, 2], [1, 2]), log_odds[0], atol=1.0e-7)
    assert_close(compute_log_factor_ratio([0, 1, 2], [0, 2]), log_odds[1], atol=1.0e-7)
    assert_close(compute_log_factor_ratio([0, 1, 2], [0, 1]), log_odds[2], atol=1.0e-7)
    for a in range(3, A):
        assert_close(compute_log_factor_ratio([0, 1, 2, a], [0, 1, 2]), log_odds[a], atol=1.0e-7)


@pytest.mark.parametrize("A_num_included", [(4, 0), (4, 1), (4, 2), (5, 0), (5, 1)])
@pytest.mark.parametrize("nu_eff", [0.71, 1.9])
def test_bvas_compute_add_log_prob(A_num_included, nu_eff, tau=0.47):
    A, num_included_alleles = A_num_included
    Y, Gamma = get_nb_data(num_alleles=A + num_included_alleles, num_regions=20, num_variants=10)

    sampler = BVASSampler(Y, Gamma.clone(), S=1.0, tau=tau, nu_eff=nu_eff, num_included_alleles=num_included_alleles)

    if num_included_alleles > 0:
        included_alleles = list(range(A, A + num_included_alleles))
    else:
        included_alleles = []

    def compute_log_factor(ind):
        ind.extend(included_alleles)
        precision = tau * torch.eye(len(ind))
        F = torch.inverse(nu_eff * Gamma[ind][:, ind] + precision)
        YFY = (torch.mv(F, Y[ind]) * Y[ind]).sum(0)
        Gamma_scaled = Gamma[ind][:, ind] * nu_eff + precision
        return 0.5 * (nu_eff ** 2) * YFY - 0.5 * Gamma_scaled.logdet()

    def compute_log_factor_ratio(ind1, ind0):
        return compute_log_factor(ind1) - compute_log_factor(ind0) + sampler.log_h_ratio + 0.5 * math.log(tau)

    check_gammas(sampler, A, compute_log_factor_ratio, num_included_alleles)

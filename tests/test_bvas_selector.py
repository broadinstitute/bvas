import numpy as np
import pytest
import torch
from common import assert_close
from generate_test_data import get_nb_data
from torch.distributions import Bernoulli

from bvas import BVASSelector


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_bvas_selector(device, A=500, T=2000, T_burnin=200, report_frequency=500,
                       beta0=0.04, beta1=0.08, seed=1):

    Y, Gamma = get_nb_data(num_alleles=A, beta0=beta0, beta1=beta1, seed=seed)

    if device == "cpu":
        Y, Gamma = Y.double(), Gamma.double()
    elif device == "gpu":
        if torch.cuda.is_available():
            Y, Gamma = Y.cuda(), Gamma.cuda()
        else:
            return

    genotype_matrix = Bernoulli(0.2).sample(sample_shape=(5, A)).type_as(Y)
    variant_names = ["VarA", "VarB", "VarC", "VarD", "VarE"]

    mutations = ["mut{}".format(k) for k in range(A)]

    S = torch.ones(A).type_as(Y) / 500.0
    S[2] = 0.9999  # strong a priori preference for allele 2
    selector = BVASSelector(Y, Gamma, mutations=mutations,
                            S=S, tau=10.0,
                            genotype_matrix=genotype_matrix, variant_names=variant_names)

    selector.run(T=T, T_burnin=T_burnin, seed=seed)
    summary = selector.summary

    print(selector.summary.iloc[:7])
    assert selector.growth_rates.values.shape[0] == len(variant_names)

    assert_close(summary['PIP'][:3].values, np.ones(3), atol=0.1)
    assert summary['PIP'].values[3:].max().item() < 0.02

    assert_close(summary['Beta'].values[:2], np.array([beta0, beta1]), atol=0.1)
    assert np.fabs(summary['Beta'].values[3:]).max().item() < 1.0e-4

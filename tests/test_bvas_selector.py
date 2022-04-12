import numpy as np
import torch
from common import assert_close
from generate_data import get_nb_data
from torch.distributions import Bernoulli

from bvas import BVASSelector

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def test_bvas_sampler(A=1000, duration=30, V=50, T=2000, regions=5, beta0=0.04, beta1=0.08,
                      T_burnin=200, report_frequency=500, seed=1):

    Y, Gamma = get_nb_data(num_alleles=A, duration=duration, num_variants=V, num_regions=regions,
                           beta0=beta0, beta1=beta1, seed=seed)

    genotype_matrix = Bernoulli(0.2).sample(sample_shape=(5, A))
    variant_names = ["VarA", "VarB", "VarC", "VarD", "VarE"]

    mutations = ["mut{}".format(k) for k in range(A)]

    selector = BVASSelector(Y, Gamma, mutations=mutations,
                            S=(0.1, 100.0), tau=10.0,
                            genotype_matrix=genotype_matrix, variant_names=variant_names)

    selector.run(T=T, T_burnin=T_burnin, seed=seed)
    summary = selector.summary

    print(selector.summary.iloc[:7])
    assert selector.growth_rates.values.shape[0] == len(variant_names)

    assert_close(summary['PIP'][:2].values, np.ones(2), atol=0.1)
    assert summary['PIP'].values[2:].max().item() < 0.01

    assert_close(summary['Beta'].values[:2], np.array([beta0, beta1]), atol=0.1)
    assert np.fabs(summary['Beta'].values[2:]).max().item() < 1.0e-4
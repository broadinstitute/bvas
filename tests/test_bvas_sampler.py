import numpy as np
from common import assert_close
from generate_test_data import get_nb_data
from torch.distributions import Bernoulli

from bvas import BVASSampler
from bvas.util import namespace_to_numpy, stack_namespaces


def test_bvas_sampler(A=500, T=2000, T_burnin=200, report_frequency=500,
                      beta0=0.04, beta1=0.08, seed=1):

    Y, Gamma = get_nb_data(num_alleles=A, beta0=beta0, beta1=beta1, seed=seed)
    Y, Gamma = Y.double(), Gamma.double()
    genotype_matrix = Bernoulli(0.2).sample(sample_shape=(5, A)).double()

    samples = []
    sampler = BVASSampler(Y, Gamma, S=(0.1, 100.0), tau=10.0, genotype_matrix=genotype_matrix)

    for t, (burned, s) in enumerate(sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed)):
        if burned:
            if t % report_frequency == 0:
                print("[iteration {}]  num_active: {}".format(t, s.gamma.sum().item()))
            samples.append(namespace_to_numpy(s))

    samples = stack_namespaces(samples)
    weights = samples.weight / samples.weight.sum()

    if hasattr(samples, 'h_alpha'):
        ratio = samples.h_alpha / (samples.h_alpha + samples.h_beta)
        ratio = np.dot(ratio.T, weights)
        print("h_ratio", ratio)

    pip = np.dot(samples.add_prob.T, weights)
    print("pip[:4]", pip[:4].tolist())
    print("maxpip[2:]", pip[2:].max())
    assert_close(pip[:2], np.ones(2), atol=0.1)
    assert pip[2:].max().item() < 0.01

    beta = np.dot(np.transpose(samples.beta), weights)
    print("beta[:4]", beta[:4].tolist())
    print("maxbeta[2:]", np.fabs(beta)[2:].max())
    assert_close(beta[:2], np.array([beta0, beta1]), atol=0.1)
    assert np.fabs(beta)[2:].max().item() < 3.0e-4

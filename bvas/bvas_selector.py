import numpy as np
import pandas as pd
from tqdm.contrib import tenumerate

from bvas import BVASSampler
from bvas.containers import StreamingSampleContainer
from bvas.util import namespace_to_numpy


class BVASSelector(object):
    r"""
    """
    def __init__(self, Y, Gamma, mutations, S,
                 tau=100.0, nu_eff=1.0,
                 genotype_matrix=None, variant_names=None):

        if Y.ndim != 1 or Gamma.ndim != 2:
            raise ValueError("Y and Gamma must be 1- and 2-dimensional, respectively.")
        if Y.shape != Gamma.shape[0:1] or Y.shape != Gamma.shape[1:2]:
            raise ValueError("Y and Gamma must have shapes (A,) and (A, A), respectively, " +
                             "where A is the number of alleles.")
        if len(mutations) != Y.size(0):
            raise ValueError("mutations must be a list of strings with length equal to Y.")
        if genotype_matrix is not None and (genotype_matrix.ndim != 2 or genotype_matrix.size(-1) != Y.size(-1)):
            raise ValueError("If genotype_matrix is provided it must have shape (num_variants, A), where A" +
                             " is the number of alleles.")
        if (genotype_matrix is not None and variant_names is None) or \
                (genotype_matrix is None and variant_names is not None):
            raise ValueError("genotype_matrix and variant_names must both be provided or both left unprovided.")
        if variant_names is not None and len(variant_names) != genotype_matrix.size(0):
            raise ValueError("variant names must be a list of strings of length V, " +
                             "where (V, A) is the shape of genotype_matrix")

        self.container = StreamingSampleContainer()
        self.mutations = mutations
        self.genotype_matrix = genotype_matrix
        self.variant_names = variant_names

        self.sampler = BVASSampler(Y, Gamma,
                                   nu_eff=nu_eff, S=S, tau=tau,
                                   explore=10,
                                   genotype_matrix=genotype_matrix)

    def run(self, T, T_burnin, seed=None):
        enumerate_samples = tenumerate(self.sampler.mcmc_chain(T=T, T_burnin=T_burnin, seed=seed),
                                       total=T + T_burnin)

        for t, (burned, sample) in enumerate_samples:
            if burned:
                self.container(namespace_to_numpy(sample))

        pip = pd.Series(self.container.pip, index=self.mutations, name="PIP")
        beta = pd.Series(self.container.beta, index=self.mutations, name="Beta")
        beta_std = pd.Series(self.container.beta_std, index=self.mutations, name="BetaStd")
        conditional_beta = pd.Series(self.container.conditional_beta, index=self.mutations, name="ConditionalBeta")
        conditional_beta_std = pd.Series(self.container.conditional_beta_std,
                                         index=self.mutations, name="ConditionalBetaStd")
        self.summary = pd.concat([pip, beta, beta_std, conditional_beta, conditional_beta_std], axis=1)
        self.summary = self.summary.sort_values(by=['PIP'], ascending=False)
        self.summary['Rank'] = np.arange(1, self.summary.values.shape[0] + 1)

        if hasattr(self.container, 'growth_rate'):
            growth_rate = pd.Series(self.container.growth_rate, name="GrowthRate")
            growth_rate_std = pd.Series(self.container.growth_rate_std, name="GrowthRateStd")
            pango = pd.Series(self.variant_names, name="Variant Name")
            self.growth_rates = pd.concat([growth_rate, growth_rate_std, pango], axis=1)

        if hasattr(self.container, 'h'):
            if self.container.h.size == 1:
                print('[h_ratio] {:.4f}'.format(self.container.h.item()))

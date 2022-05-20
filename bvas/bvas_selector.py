import numpy as np
import pandas as pd
from tqdm.contrib import tenumerate

from bvas import BVASSampler
from bvas.containers import StreamingSampleContainer
from bvas.util import namespace_to_numpy


def populate_alpha_beta_stats(container, stats):
    for s in ['h_alpha', 'h_beta', 'h']:
        if hasattr(container, s):
            stats['Mean ' + s] = getattr(container, s)


def populate_weight_stats(selector, stats, weights, quantiles=[5.0, 10.0, 20.0, 50.0, 90.0, 95.0]):
    q5, q10, q20, q50, q90, q95 = np.percentile(weights, quantiles).tolist()
    s = "5/10/20/50/90/95:  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight quantiles'] = s.format(q5, q10, q20, q50, q90, q95)
    s = "mean/std/min/max:  {:.2e}  {:.2e}  {:.2e}  {:.2e}"
    stats['Weight moments'] = s.format(weights.mean().item(), weights.std().item(),
                                       weights.min().item(), weights.max().item())


class BVASSelector(object):
    r"""
    Main analysis class for Bayesian Viral Allele Selection (BVAS).
    Combines a Gaussian diffusion-based likelihood with Bayesian
    Variable Selection. Uses :class:`BVASSampler` under the hood to do MCMC
    inference.

    Usage::

        selector = BVASSelector(Y, Gamma, mutations, S=10.0, tau=100.0)
        selector.run(T=2000, T_burnin=1000)
        print(selector.summary)
        print(selector.growth_rates)
        print(selector.stats)

    Note that computations will be done using the `device` and `dtype` of the provided
    `torch.Tensor`s, i.e. `Y` and `Gamma.` If you would like computations to be done
    with a GPU make sure that these tensors are on the GPU. We recommend doing all
    computations in 64-bit precision, i.e. `Y.dtype == Gamma.dtype == torch.float64`.

    GPU usage::

        selector = BVASSelector(Y.cuda().double(), Gamma.cuda().double(), mutations, S=10.0, tau=100.0)
        selector.run(T=2000, T_burnin=1000)

    The inputs `Y` and `Gamma` are defined in terms of region-specific allele frequencies
    :math:`{\mathbf x}_r(t)` and region-specific effective population sizes :math:`\nu_r` as follows.

    .. math::

        &{\mathbf y}_r(t) = {\mathbf x}_r(t + 1) - {\mathbf x}_r(t)

        &\bar{\mathbf{Y}}^\nu \equiv \sum_r \nu_r \sum_t {\mathbf y}_r(t)

        &\Lambda_{r,ab}(t) = x_{r,ab}(t) - x_{r,a}(t) x_{r,b}(t)

        &\bar{\mathbf{\Lambda}}^\nu \equiv \sum_r \nu_r \sum_t {\mathbf \Lambda}_r(t)

    where :math:`x_{r,ab}(t)` denote pairwise allele frequencies in region :math:`r`.

    :param torch.Tensor Y: A vector of shape `(A,)` that encodes integrated alelle frequency
        increments for each allele and where `A` is the number of alleles.
    :param torch.Tensor Gamma: A matrix of shape `(A, A)` that encodes information about
        second moments of allele frequencies.
    :param list mutations: A list of strings of length `A` that encodes the names of the `A` alleles in `Y`.
    :param S: Controls the expected number of alleles to include in the model a priori. Defaults to 5.0.
        To specify allele-level prior inclusion probabilities provide a A-dimensional `torch.Tensor` of
        the form `(h_1, ..., h_A)`. If a tuple of positive floats `(alpha, beta)` is provided, the a priori
        inclusion probability is a latent variable governed by the corresponding Beta prior so that the sparsity
        level is inferred from the data. Note that for a given choice of `alpha` and `beta` the expected number
        of alleles to include in the model a priori is given by :math:`\frac{\alpha}{\alpha + \beta} \times A`.
        We caution that this approach may be a poor choice for very noisy genomic surveillance data.
        Also note that the mean number of covariates in the posterior can vary significantly from prior
        expectations, since the posterior is in effect a compromise between the prior and the observed data.
    :param float tau: Controls the precision of the coefficients in the prior. Defaults to 100.0.
    :param float nu_eff_multiplier: Additional factor by which to multiply the effective population size, i.e. on
        top of whatever was done when computing `Y` and `Gamma`. Defaults to 1.0.
    :param torch.Tensor genotype_matrix: A matrix of shape `(num_variants, A)` that encodes the genotype
        of various viral variants. If included the sampler will compute variant-level growth rates
        during inference for the varaints in `genotype_matrix`.
        Defaults to None. If not None, user must also provide `variant_names`.
    :param list variant_names: A list of names of the variants specified by `genotype_matrix`. Must have the
        same length as the leading dimension of `genotype_matrix`. Defaults to None.
    """
    def __init__(self, Y, Gamma, mutations, S=5.0,
                 tau=100.0, nu_eff_multiplier=1.0,
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
                                   nu_eff_multiplier=nu_eff_multiplier, S=S, tau=tau,
                                   explore=10,
                                   genotype_matrix=genotype_matrix)

    def run(self, T, T_burnin, seed=None):
        r"""
        Run MCMC inference for :math:`T + T_{\rm burn-in}` iterations.
        The leading :math:`T_{\rm burn-in}` iterations are discarded. After completion the results
        of the MCMC run can be accessed in the `summary`, `growth_rates`, and `stats` attributes.

        The `summary` :class:`pandas.DataFrame` contains six columns.
        The first column lists the Posterior Inclusion Probability (PIP) for each covariate.
        The second column lists the posterior mean of the coefficient
        that corresponds to each covariate. The third column lists the posterior standard deviation for
        each coefficient. The fourth and fifth columns are analogous to the second and third columns,
        respectively, with the difference that the fourth and fifth columns report conditional posterior
        statistics. For example, the fourth column reports the posterior mean of each coefficient
        conditioned on the corresponding covariate being included in the model. The sixth column
        is the PIP rank.

        The `growth_rates` :class:`pandas.DataFrame` reports estimated relative growth rates for
        each variant in `genotype_matrix` if the latter was provided. Note that growth rates are
        relative to the wild-type variant with all-zeros genotype.

        :param int T: Positive integer that controls the number of MCMC samples that are
            generated (i.e. after burn-in/adaptation).
        :param int T_burnin: Positive integer that controls the number of MCMC samples that are
            generated during burn-in/adaptation.
        :param int seed: Random number seed for reproducibility. Defaults to None.
        """
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

        self.stats = {}
        self.weights = np.array(self.container._weights)
        populate_alpha_beta_stats(self.container, self.stats)
        populate_weight_stats(self, self.stats, self.weights)

import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch.distributions import Beta, Categorical, Uniform
from torch.linalg import norm
from torch.linalg import solve_triangular as trisolve

from .sampler import MCMCSampler
from .util import (
    get_loo_inverses,
    leave_one_out_off_diagonal,
    leave_one_out_vector,
    safe_cholesky,
)


class BVASSampler(MCMCSampler):
    r"""
    MCMC Sampler for Bayesian Viral Allele Selection (BVAS).
    Combines a Gaussian diffusion-based likelihood with Bayesian
    Variable Selection. Most users will not use this class directly
    and will instead use :class:`BVASSelector`.

    Note that computations will be done using the `device` and `dtype` of the provided
    `torch.Tensor`s, i.e. `Y` and `Gamma.` If you would like computations to be done
    with a GPU make sure that these tensors are on the GPU. We recommend doing all
    computations in 64-bit precision, i.e. `Y.dtype == Gamma.dtype == torch.float64`.

    The inputs `Y` and `Gamma` are defined in terms of region-specific allele frequencies
    :math:`{\mathbf x}_r(t)` and region-specific effective population sizes :math:`\nu_r` as follows.

    .. math::

        &{\mathbf y}(t) = {\mathbf x}(t + 1) - {\mathbf x}(t)

        &\widebar{\mathbf{y}}^\nu \equiv \sum_{r=1} \nu_r \sum_{t=1} {\mathbf y}_r(t)

        &{\mathbf \Lambda}_{ab}(t) = {\mathbf x}_{ab}(t) - {\mathbf x}_a(t) {\mathbf x}_b(t)

    where :math:`{\mathbf x}_{ab}(t)` denote pairwise allele frequencies.

    :param torch.Tensor Y: A vector of shape `(A,)` that encodes integrated alelle frequency
        increments for each allele and where `A` is the number of alleles.
    :param torch.Tensor Gamma: A matrix of shape `(A, A)` that encodes information about
        second moments of allele frequencies.
    :param S: Controls the expected number of alleles to include in the model a priori. Defaults to 5.0.
        If a tuple of positive floats `(alpha, beta)` is provided, the a priori inclusion probability is a latent
        variable governed by the corresponding Beta prior so that the sparsity level is inferred from the data.
        Note that for a given choice of `alpha` and `beta` the expected number of alleles to include in the model
        a priori is given by :math:`\frac{\alpha}{\alpha + \beta} \times A`.  Also note that the mean number of
        alleles in the posterior can vary significantly from prior expectations, since the posterior is in
        effect a compromise between the prior and the observed data.
    :param float tau: Controls the precision of the coefficients in the prior. Defaults to 100.0.
    :param float nu_eff_multiplier: Additional factor by which to multiply the effective population size, i.e. on top
        of whatever was done when computing `Y` and `Gamma`. Defaults to 1.0.
    :param float explore: This hyperparameter controls how greedy the MCMC algorithm is. Defaults to 10.0.
        For expert users only.
    :param float xi_target: This hyperparameter controls how often :math:`h` MCMC updates are made if :math:`h`
        is a latent variable. Defaults to 0.2. For expert users only.
    :param torch.Tensor genotype_matrix: A matrix of shape `(num_variants, A)` that encodes the genotype
        of various viral variants. If included the sampler will compute variant-level growth rates
        during inference for the varaints encoded by `genotype_matrix`.
        Defaults to None.
    """
    def __init__(self, Y, Gamma,
                 S=5, tau=100.0, nu_eff_multiplier=1.0,
                 explore=10.0, xi_target=0.2,
                 genotype_matrix=None):

        assert Y.ndim == 1 and Gamma.ndim == 2
        self.A = Y.size(-1)
        assert Gamma.shape == (self.A, self.A)

        self.Y = Y
        self.Gamma = Gamma

        self.device = Y.device
        self.dtype = Y.dtype

        self.tau = tau
        self.nu_eff_multiplier = nu_eff_multiplier
        self.genotype_matrix = genotype_matrix

        self.uniform_dist = Uniform(0.0, Y.new_ones(1)[0])

        if not isinstance(S, tuple):
            if S >= self.A or S <= 0:
                raise ValueError("S must satisfy 0 < S < A or must be a tuple.")
        else:
            if len(S) != 2 or not isinstance(S[0], float) or not isinstance(S[1], float) or S[0] <= 0.0 or S[1] <= 0.0:
                raise ValueError("If S is a tuple it must be a tuple of two positive floats (alpha, beta).")
        if explore <= 0.0:
            raise ValueError("explore must satisfy explore > 0.0")
        if xi_target <= 0.0 or xi_target >= 1.0:
            raise ValueError("xi_target must be in the interval (0, 1).")

        self.Gamma_diag = self.Gamma.diagonal()

        if not isinstance(S, tuple):
            self.h = S / self.A
            self.xi = torch.tensor([0.0], device=Y.device)
        else:
            self.h_alpha, self.h_beta = S
            self.h = self.h_alpha / (self.h_alpha + self.h_beta)
            self.xi = torch.tensor([5.0], device=Y.device)
            self.xi_target = xi_target

        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.explore = explore / self.A
        self.epsilon = 1.0e3 * torch.finfo(Y.dtype).tiny

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.A, device=self.device).bool(),
                                 add_prob=torch.zeros(self.A, device=self.device, dtype=self.dtype),
                                 _i_prob=torch.zeros(self.A, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 _idx=0, _log_h_ratio=self.log_h_ratio,
                                 weight=0,
                                 beta=torch.zeros(self.A, device=self.device, dtype=self.dtype))

        if hasattr(self, "h_alpha"):
            sample.h_alpha = torch.tensor(self.h_alpha, device=self.device)
            sample.h_beta = torch.tensor(self.h_beta, device=self.device)

        sample = self._compute_probs(sample)
        return sample

    # compute p(gamma_a | gamma_{-a})
    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)

        assert num_active < self.A, "The MCMC sampler has been driven into a regime where " +\
            "all alleles have been selected. Are you sure you have chosen a reasonable prior? " +\
            "Are you sure there is signal in your data?"

        nu = self.nu_eff_multiplier
        Y_k = self.Y[inactive] * nu
        Gamma_k = self.Gamma_diag[inactive] * nu + self.tau

        if num_active > 0:
            Y_active = self.Y[active] * nu
            Gamma_active = self.Gamma[active][:, active] * nu
            Gamma_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            L_active = safe_cholesky(Gamma_active)

            Yt_active = trisolve(L_active, Y_active.unsqueeze(-1), upper=False).squeeze(-1)

            L_G_I_k = trisolve(L_active, nu * self.Gamma[active][:, inactive], upper=False)
            G_k_inv = Gamma_k - norm(L_G_I_k, dim=0).pow(2.0)

            W_k_sq = (torch.mv(L_G_I_k.t(), Yt_active) - Y_k).pow(2.0) / (G_k_inv + self.epsilon)
            Yt_active_sq = Yt_active.pow(2.0).sum()
            log_det_inactive = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)
        else:
            W_k_sq = Y_k.pow(2.0) / (Gamma_k + self.epsilon)
            Yt_active_sq = 0.0
            log_det_inactive = -0.5 * torch.log(Gamma_k / self.tau)

        if num_active > 0:
            beta_active = trisolve(L_active.t(), Yt_active.unsqueeze(-1), upper=True).squeeze(-1)
            sample._beta_mean = self.Y.new_zeros(self.A)
            sample._beta_mean[active] = beta_active

            sample.beta = self.Y.new_zeros(self.A)
            sample.beta[active] = beta_active + \
                trisolve(L_active, torch.randn(active.size(-1), 1, device=self.device, dtype=self.dtype),
                         upper=False).squeeze(-1)
        elif num_active == 0:
            sample._beta_mean = self.Y.new_zeros(self.A)
            sample.beta = self.Y.new_zeros(self.A)

        if self.genotype_matrix is not None:
            sample.growth_rate = 1.0 + self.genotype_matrix @ sample.beta

        if num_active > 1:
            Gamma_off_diagonal = leave_one_out_off_diagonal(Gamma_active)  # I I-1
            active_loo = leave_one_out_vector(active)  # I I-1

            Y_active_loo = nu * self.Y[active_loo]  # I I-1

            F = torch.cholesky_inverse(L_active, upper=False)
            F_loo = get_loo_inverses(F)  # I I-1 I-1

            Yt_active_loo = matmul(F_loo, Y_active_loo.unsqueeze(-1)).squeeze(-1)
            Yt_active_loo_sq = einsum("ij,ij->i", Yt_active_loo, Y_active_loo)  # I

            triple_term = matmul(F_loo, Gamma_off_diagonal.unsqueeze(-1)).squeeze(-1)
            triple_term = torch.einsum("ij,ij->i", Gamma_off_diagonal, triple_term)
            G_k_inv = self.Gamma_diag[active] * nu + self.tau - triple_term
            log_det_active = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)

        elif num_active == 1:
            Yt_active_loo_sq = 0.0
            log_det_active = -0.5 * Gamma_active[0].log() + 0.5 * math.log(self.tau)

        elif num_active == 0:
            Yt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        log_odds_inactive = 0.5 * W_k_sq + log_det_inactive + sample._log_h_ratio
        log_odds_active = 0.5 * (Yt_active_sq - Yt_active_loo_sq) + log_det_active + sample._log_h_ratio

        log_odds = self.Y.new_zeros(self.A)
        log_odds[inactive] = log_odds_inactive
        log_odds[active] = log_odds_active

        return log_odds

    def _compute_probs(self, sample):
        sample.add_prob = sigmoid(self._compute_add_prob(sample))

        gamma = sample.gamma.type_as(sample.add_prob)
        prob_gamma_i = gamma * sample.add_prob + (1.0 - gamma) * (1.0 - sample.add_prob)
        i_prob = 0.5 * (sample.add_prob + self.explore) / (prob_gamma_i + self.epsilon)

        if hasattr(self, 'h_alpha') and self.t <= self.T_burnin:  # adapt xi
            self.xi += (self.xi_target - self.xi / (self.xi + i_prob.sum())) / math.sqrt(self.t + 1)

        sample._i_prob = torch.cat([self.xi, i_prob])

        return sample

    def mcmc_move(self, sample):
        self.t += 1

        sample._idx = Categorical(probs=sample._i_prob).sample() - 1

        if sample._idx.item() >= 0:
            sample.gamma[sample._idx] = ~sample.gamma[sample._idx]
            sample._active = torch.nonzero(sample.gamma).squeeze(-1)
        else:
            sample = self.sample_alpha_beta(sample)

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()

        return sample

    def sample_alpha_beta(self, sample):
        num_active = sample._active.size(-1)
        num_inactive = self.A - num_active
        sample.h_alpha = torch.tensor(self.h_alpha + num_active, device=self.device)
        sample.h_beta = torch.tensor(self.h_beta + num_inactive, device=self.device)
        h = Beta(sample.h_alpha, sample.h_beta).sample().item()
        sample._log_h_ratio = math.log(h) - math.log(1.0 - h)
        return sample

import math
from types import SimpleNamespace

import torch
from torch import einsum, matmul, sigmoid
from torch import triangular_solve as trisolve
from torch.distributions import Beta, Categorical, Uniform
from torch.linalg import norm

from .sampler import MCMCSampler
from .util import (
    get_loo_inverses,
    leave_one_out_off_diagonal,
    leave_one_out_vector,
    safe_cholesky,
)


class BVASSampler(MCMCSampler):
    r"""
    """
    def __init__(self, Y, Gamma, S=5, nu_eff=1.0,
                 tau=0.01, explore=5,
                 verbose_constructor=True, xi_target=0.2,
                 gene_map=None,
                 num_included_alleles=0,
                 genotype_matrix=None):

        assert Y.ndim == 1 and Gamma.ndim == 2
        self.A = Y.size(-1)
        assert Gamma.shape == (self.A, self.A)

        self.Y = Y
        self.Gamma = Gamma

        self.device = Y.device
        self.dtype = Y.dtype

        self.tau = tau
        self.nu_eff = nu_eff
        self.num_included_alleles = num_included_alleles
        self.Asel = self.A - self.num_included_alleles
        self.included_alleles = torch.arange(self.Asel, self.Asel + self.num_included_alleles,
                                             device=self.device, dtype=torch.int64)
        self.genotype_matrix = genotype_matrix

        self.gene_map = gene_map
        if gene_map is not None:
            assert self.A == self.Asel
            assert self.gene_map.size(-1) == self.A
            self.gene_map_bins = self.gene_map.unique().size(-1)
            assert self.gene_map.max() + 1 == self.gene_map_bins
            gene_map = gene_map.expand(self.gene_map_bins, -1)
            self.gene_map_lengths = ((gene_map - torch.arange(self.gene_map_bins).unsqueeze(-1)) == 0).sum(-1)
        else:
            self.gene_map = torch.zeros(self.Asel, device=self.device, dtype=torch.int64)
            self.gene_map_bins = 1
            self.gene_map_lengths = torch.tensor(self.Asel, device=self.device, dtype=torch.int64)

        self.uniform_dist = Uniform(0.0, Y.new_ones(1)[0])

        if not isinstance(S, tuple):
            if S >= self.Asel or S <= 0:
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
            self.h = S / self.Asel
            self.xi = torch.tensor([0.0], device=Y.device)
        else:
            self.h_alpha, self.h_beta = S
            self.h = self.h_alpha / (self.h_alpha + self.h_beta)
            self.xi = torch.tensor([5.0], device=Y.device)
            self.xi_target = xi_target

        self.log_h_ratio = math.log(self.h) - math.log(1.0 - self.h)
        self.explore = explore / self.Asel
        self.epsilon = 1.0e-18

        if verbose_constructor:
            s2 = " = ({}, {:.1f}, {:.3f}, {:.3f}, {})" if not isinstance(S, tuple) \
                else " = ({}, ({:.1f}, {:.1f}), {:.3f}, {:.3f}, {})"
            S = S if isinstance(S, tuple) else (S,)
            s1 = "Initialized BVASSampler with (A, S, tau, nu_eff, A_included)"
            print((s1 + s2).format(self.Asel, *S, self.tau, nu_eff, self.num_included_alleles))

    def initialize_sample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        sample = SimpleNamespace(gamma=torch.zeros(self.Asel, device=self.device).bool(),
                                 add_prob=torch.zeros(self.Asel, device=self.device, dtype=self.dtype),
                                 _i_prob=torch.zeros(self.Asel, device=self.device, dtype=self.dtype),
                                 _active=torch.tensor([], device=self.device, dtype=torch.int64),
                                 _idx=0, _log_h_ratio=self.log_h_ratio * torch.ones(self.Asel, device=self.device),
                                 weight=0,
                                 beta=torch.zeros(self.A, device=self.device, dtype=self.dtype))
        if self.num_included_alleles > 0:
            sample._activei = self.included_alleles

        if hasattr(self, "h_alpha"):
            sample.h_alpha = torch.tensor(self.h_alpha, device=self.device)
            sample.h_beta = torch.tensor(self.h_beta, device=self.device)

        sample = self._compute_probs(sample)
        return sample

    def _compute_add_prob(self, sample, return_log_odds=False):
        active = sample._active
        activei = sample._activei if self.num_included_alleles > 0 else sample._active
        inactive = torch.nonzero(~sample.gamma).squeeze(-1)
        num_active = active.size(-1)

        assert num_active < self.Asel, "The MCMC sampler has been driven into a regime where " +\
            "all alleles have been selected. Are you sure you have chosen a reasonable prior? " +\
            "Are you sure there is signal in your data?"

        nu = self.nu_eff
        Y_k = self.Y[inactive] * nu
        Gamma_k = self.Gamma_diag[inactive] * nu + self.tau

        if num_active > 0 or self.num_included_alleles > 0:
            Y_active = self.Y[activei] * nu
            Gamma_active = self.Gamma[activei][:, activei] * nu
            Gamma_active.diagonal(dim1=-2, dim2=-1).add_(self.tau)

            L_active = safe_cholesky(Gamma_active)

            Yt_active = trisolve(Y_active.unsqueeze(-1), L_active, upper=False)[0].squeeze(-1)

            L_G_I_k = trisolve(nu * self.Gamma[activei][:, inactive], L_active, upper=False)[0]
            G_k_inv = Gamma_k - norm(L_G_I_k, dim=0).pow(2.0)

            W_k_sq = (torch.mv(L_G_I_k.t(), Yt_active) - Y_k).pow(2.0) / (G_k_inv + self.epsilon)
            Yt_active_sq = Yt_active.pow(2.0).sum()
            log_det_inactive = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)
        else:
            W_k_sq = Y_k.pow(2.0) / (Gamma_k + self.epsilon)
            Yt_active_sq = 0.0
            log_det_inactive = -0.5 * torch.log(Gamma_k / self.tau)

        if num_active > 0 or self.num_included_alleles > 0:
            beta_active = trisolve(Yt_active.unsqueeze(-1), L_active.t(), upper=True)[0].squeeze(-1)
            sample._beta_mean = self.Y.new_zeros(self.A)
            sample._beta_mean[activei] = beta_active

            sample.beta = self.Y.new_zeros(self.A)
            sample.beta[activei] = beta_active + \
                trisolve(torch.randn(activei.size(-1), 1, device=self.device, dtype=self.dtype),
                         L_active, upper=False)[0].squeeze(-1)
        elif num_active == 0:
            sample._beta_mean = self.Y.new_zeros(self.A)
            sample.beta = self.Y.new_zeros(self.A)

        if self.genotype_matrix is not None:
            sample.growth_rate = self.genotype_matrix @ sample.beta

        if num_active > 1:
            Gamma_off_diagonal = leave_one_out_off_diagonal(Gamma_active)  # I I-1
            active_loo = leave_one_out_vector(active)  # I I-1

            if self.num_included_alleles > 0:
                Gamma_off_diagonal = Gamma_off_diagonal[:-self.num_included_alleles]
                included_alleles = self.included_alleles.expand(num_active, -1)
                active_loo = torch.cat([active_loo, included_alleles], dim=-1)

            Y_active_loo = nu * self.Y[active_loo]  # I I-1

            F = torch.cholesky_inverse(L_active, upper=False)
            F_loo = get_loo_inverses(F)  # I I-1 I-1
            if self.num_included_alleles > 0:
                F_loo = F_loo[:-self.num_included_alleles]

            Yt_active_loo = matmul(F_loo, Y_active_loo.unsqueeze(-1)).squeeze(-1)
            Yt_active_loo_sq = einsum("ij,ij->i", Yt_active_loo, Y_active_loo)  # I

            triple_term = matmul(F_loo, Gamma_off_diagonal.unsqueeze(-1)).squeeze(-1)
            triple_term = torch.einsum("ij,ij->i", Gamma_off_diagonal, triple_term)
            G_k_inv = self.Gamma_diag[active] * nu + self.tau - triple_term
            log_det_active = -0.5 * G_k_inv.log() + 0.5 * math.log(self.tau)

        elif num_active == 1:
            if self.num_included_alleles == 0:
                Yt_active_loo_sq = 0.0
                log_det_active = -0.5 * Gamma_active[0].log() + 0.5 * math.log(self.tau)
            else:
                Gamma_assumed = self.Gamma[self.included_alleles][:, self.included_alleles] * nu
                Gamma_assumed.diagonal(dim1=-2, dim2=-1).add_(self.tau)

                L_assumed = safe_cholesky(Gamma_assumed)
                Yt_active_loo = trisolve(nu * self.Y[self.included_alleles].unsqueeze(-1),
                                         L_assumed, upper=False)[0].squeeze(-1)
                Yt_active_loo_sq = norm(Yt_active_loo, dim=0).pow(2.0)
                log_det_active = L_assumed.diagonal().log().sum() - L_active.diagonal().log().sum()
                log_det_active += 0.5 * math.log(self.tau)

        elif num_active == 0:
            Yt_active_loo_sq = 0.0
            log_det_active = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        log_odds_inactive = 0.5 * W_k_sq + log_det_inactive + sample._log_h_ratio[inactive]
        log_odds_active = 0.5 * (Yt_active_sq - Yt_active_loo_sq) + log_det_active + sample._log_h_ratio[active]

        log_odds = self.Y.new_zeros(self.Asel)
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
            if self.num_included_alleles > 0:
                sample._activei = torch.cat([sample._active, self.included_alleles])
        else:
            sample = self.sample_alpha_beta(sample)

        sample = self._compute_probs(sample)
        sample.weight = sample._i_prob.mean().reciprocal()

        return sample

    def sample_alpha_beta(self, sample):
        num_active = torch.bincount(self.gene_map[sample.gamma], minlength=self.gene_map_bins)
        num_inactive = self.gene_map_lengths - num_active
        sample.h_alpha = self.h_alpha + num_active
        sample.h_beta = self.h_beta + num_inactive
        h = Beta(sample.h_alpha, sample.h_beta).sample()
        sample._log_h_ratio = (torch.log(h) - torch.log1p(-h))[self.gene_map]
        return sample

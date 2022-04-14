import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from torch.linalg import solve_triangular as trisolve

from bvas.util import safe_cholesky


def laplace_inference(Y, Gamma, mutations,
                      coef_scale=1.0e-2, seed=0, num_steps=10 ** 4,
                      log_every=500, init_lr=0.01):
    r"""
    Use Maximum A Posteriori (MAP) inference and a diffusion-based likelihood in conjunction
    with a sparsity-inducing Laplace prior on selection coefficients to infer
    selection effects from genomic surveillance data.

    Unlike most of the code in this repository, `laplace_inference` depends on Pyro.

    :param torch.Tensor Y: A vector of shape `(A,)` that encodes integrated alelle frequency
        increments for each allele and where `A` is the number of alleles.
    :param torch.Tensor Gamma: A matrix of shape `(A, A)` that encodes information about
        second moments of allele frequencies.
    :param list mutations: A list of strings of length `A` that encodes the names of the `A` alleles in `Y`.
    :param float coef_scale: The regularization scale of the Laplace prior. Defaults to 0.01.
    :param int seed: Random number seed for reproducibility.
    :param int num_steps: The number of optimization steps to do. Defaults to 10000.
    :param int log_every: Controls logging frequency. Defaults to 500.
    :param float init_lr: The initial learning rate. Defaults to 0.01.

    :returns pandas.DataFrame: Returns a `pandas.DataFrame` containing results of inference.
    """
    pyro.clear_param_store()

    A = Gamma.size(-1)
    assert len(mutations) == A == Gamma.size(-2) == Y.size(0)

    L = safe_cholesky(Gamma, num_tries=10)
    L_Y = trisolve(L, Y.unsqueeze(-1), upper=False).squeeze(-1)

    def model():
        beta = pyro.sample("beta", dist.Laplace(0.0, coef_scale * torch.ones(A).type_as(L)).to_event(1))
        pyro.factor("obs", -0.5 * (L.t() @ beta - L_Y).pow(2.0).sum())

    def fit_svi():
        pyro.set_rng_seed(seed)

        guide = pyro.infer.autoguide.AutoDelta(model)
        optim = pyro.optim.ClippedAdam({"lr": init_lr, "lrd": 0.01 ** (1 / num_steps),
                                        "betas": (0.5, 0.99)})
        svi = pyro.infer.SVI(model, guide, optim, pyro.infer.Trace_ELBO())

        for step in range(num_steps):
            loss = svi.step()
            if log_every and (step % log_every == 0 or step == num_steps - 1):
                print(f"step {step: >4d} loss = {loss:0.6g}")

        return guide

    beta = fit_svi().median()['beta'].data.cpu().numpy()
    beta = pd.DataFrame(beta, index=mutations, columns=['Beta'])
    beta['BetaAbs'] = np.fabs(beta.Beta.values)
    beta = beta.sort_values(by='BetaAbs', ascending=False)
    beta['Rank'] = 1 + np.arange(beta.shape[0])

    return beta[['Beta', 'Rank']]

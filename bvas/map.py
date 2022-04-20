import numpy as np
import pandas as pd
import torch

from bvas.util import safe_cholesky


def map_inference(Y, Gamma, mutations, tau_reg):
    r"""
    Use Maximum A Posteriori (MAP) inference and a diffusion-based likelihood to infer
    selection effects from genomic surveillance data. See reference [1] for details.

    References:

        [1] "Inferring effects of mutations on SARS-CoV-2 transmission from genomic surveillance data,"
        Brian Lee, Muhammad Saqib Sohail, Elizabeth Finney, Syed Faraz Ahmed, Ahmed Abdul Quadeer,
        Matthew R. McKay, John P. Barton.

    :param torch.Tensor Y: A vector of shape `(A,)` that encodes integrated alelle frequency
        increments for each allele and where `A` is the number of alleles.
    :param torch.Tensor Gamma: A matrix of shape `(A, A)` that encodes information about
        second moments of allele frequencies.
    :param list mutations: A list of strings of length `A` that encodes the names of the `A` alleles in `Y`.
    :param float tau_reg: A positive float `tau_reg` that serves as the regularizer in MAP inference
        along the lines of ridge regression. Note that this quantity is called `gamma` in reference [1].

    :returns pandas.DataFrame: Returns a `pandas.DataFrame` containing results of inference.
    """
    L_tau = safe_cholesky(Gamma + tau_reg * torch.eye(Gamma.size(-1)).type_as(Gamma))
    Gamma_inv = torch.cholesky_inverse(L_tau, upper=False)
    beta = torch.mv(Gamma_inv, Y).data.cpu().numpy()
    beta_std = Gamma_inv.diag().sqrt().data.cpu().numpy()

    beta = pd.DataFrame(beta, index=mutations, columns=['Beta'])
    beta['BetaAbs'] = np.fabs(beta.Beta.values)
    beta['BetaStd'] = beta_std
    beta = beta.sort_values(by='BetaAbs', ascending=False)
    beta['Rank'] = 1 + np.arange(beta.shape[0])

    return beta[['Beta', 'BetaStd', 'Rank']]

import torch
from torch.linalg import solve_triangular as trisolve

from bvas.util import safe_cholesky


def map_inference(Y, Gamma, taus=[2 ** exponent for exponent in range(4, 16)]):
    r"""
    Use Maximum A Posteriori (MAP) inference and a diffusion-based likelihood to infer
    selection effects from genomic surveillance data. See reference [1] for details.

    References:

    [1] "Inferring effects of mutations on SARS-CoV-2 transmission from genomic surveillance data,"
        Brian Lee, Muhammad Saqib Sohail, Elizabeth Finney, Syed Faraz Ahmed, Ahmed Abdul Quadeer,
        Matthew R. McKay, John P. Barton.

    :param torch.Tensor Y: A torch.Tensor of shape (A,) that encodes integrated alelle frequency
        increments for each allele and where A is the number of alleles.
    :param torch.Tensor Gamma: A torch.Tensor of shape (A, A) that encodes information about
        second moments of allele frequencies.
    :param list taus: A list of floats encoding regularizers `tau_reg` to use in MAP inference, i.e. we run
        MAP once for each value of `tau_reg`. Note that this quantity is called `gamma` in reference [1].

    :returns dict: Returns a dictionary of inferred selection coefficients beta, one for each value
        in `taus`.
    """
    results = {}

    for tau_reg in taus:
        L_tau = safe_cholesky(Gamma + tau_reg * torch.eye(Gamma.size(-1)).type_as(Gamma))
        Yt = trisolve(L_tau, Y.unsqueeze(-1), upper=False)
        beta = trisolve(L_tau.t(), Yt, upper=True).squeeze(-1)
        results['map_{}'.format(tau_reg)] = {'beta': beta.data.cpu().numpy(),
                                             'tau_reg': tau_reg}

    return results

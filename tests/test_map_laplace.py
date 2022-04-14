import torch
from generate_test_data import get_nb_data

from bvas import laplace_inference, map_inference


def test_bvas_selector(A=100, beta0=0.04, beta1=0.08, seed=1):
    torch.set_default_tensor_type(torch.DoubleTensor)

    Y, Gamma = get_nb_data(num_alleles=A, beta0=beta0, beta1=beta1, seed=seed)

    mutations = ["mut{}".format(k) for k in range(A)]

    laplace_results = laplace_inference(Y, Gamma, mutations, num_steps=4000)
    assert int(laplace_results.loc['mut0'].Rank.item()) == 2
    assert int(laplace_results.loc['mut1'].Rank.item()) == 1

    map_results = map_inference(Y, Gamma, mutations, 500.0)
    assert int(map_results.loc['mut0'].Rank.item()) == 2
    assert int(map_results.loc['mut1'].Rank.item()) == 1

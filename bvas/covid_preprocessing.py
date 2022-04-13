import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.distributions import Bernoulli
from util import get_longest_ones_index


def _compute_y_gamma(N, genotype, locations, args, phi=None, window=3, verbose=1):
    X = torch.matmul(N, genotype)
    num_regions, duration, num_alleles = X.shape
    X = torch.matmul(N, genotype)  # num_regions duration num_alleles
    N_sum = N.sum(-1)

    Gamma = X.new_zeros(num_alleles, num_alleles)
    Y = X.new_zeros(num_alleles)
    nu_effs = []

    N_kept = 0
    timeseries_lengths = []

    for r in range(num_regions):
        N_sum_r = N_sum[r]
        densely_sampled = N_sum_r >= args.min_biweekly_samples
        dense_consecutive = get_longest_ones_index(densely_sampled.data.cpu().numpy())
        dense_consecutive = torch.from_numpy(dense_consecutive).to(X.device)
        timeseries_lengths.append(dense_consecutive.size(-1))

        N_sum_r = N_sum_r[dense_consecutive]
        N_kept += int(N_sum_r.sum().item())

        X_r = X[r, dense_consecutive] / N_sum_r[:, None]

        denominator = (X_r[1:] - X_r[:-1]).pow(2.0).mean(-1)
        numerator = (X_r[:-1] * (1 - X_r[:-1])).mean(-1)
        nu_eff_r = (numerator / denominator).mean().item()
        nu_effs.append(nu_eff_r)
        loc_r = locations[r].split(' / ')[1]
        if verbose:
            print("[{}] nu_eff_r: {:.1f}".format(loc_r, nu_eff_r), "N_sum_r: ", int(N_sum_r.mean().item()),
                  int(N_sum_r.median().item()), "dense ", int(densely_sampled.sum().item()), len(dense_consecutive))

    if verbose:
        print("[time series lengths] {:.2f} +- {:.2f}".format(np.mean(timeseries_lengths), np.std(timeseries_lengths)))

    nu_eff_global = np.median(nu_effs)

    for r in range(num_regions):
        N_sum_r = N_sum[r]
        densely_sampled = N_sum_r >= args.min_biweekly_samples
        dense_consecutive = get_longest_ones_index(densely_sampled.data.cpu().numpy())
        dense_consecutive = torch.from_numpy(dense_consecutive).to(X.device)
        N_sum_r = N_sum_r[dense_consecutive]

        phi_r = torch.ones_like(dense_consecutive) if phi is None else phi[r, dense_consecutive]
        phi_r = phi_r.unsqueeze(-1)

        X_r = X[r, dense_consecutive] / N_sum_r[:, None]
        X_phi_r = phi_r * X_r

        XX_r = (N[r, dense_consecutive][:-1] * phi_r[:-1] / N_sum_r[:-1, None]).sum(0)  # C
        XX_r_geno = XX_r[:, None] * genotype  # C A
        Gamma_r = genotype.t() @ XX_r_geno
        Gamma_r -= X_phi_r[:-1].t() @ X_r[:-1]
        Gamma_r.diagonal(dim1=-1, dim2=-2).copy_((X_phi_r[:-1] * (1 - X_r[:-1])).sum(0))

        Y_r = ((X_r[1:] - X_r[:-1]) * phi_r[:-1]).sum(0)

        if args.strategy == 'global':
            Gamma += nu_eff_global * Gamma_r
            Y += nu_eff_global * Y_r
        elif args.strategy == 'regional':
            Gamma += nu_eff_r * Gamma_r
            Y += nu_eff_r * Y_r
        elif args.strategy == 'regionalreg':
            nu_eff_reg_r = np.sqrt(nu_effs[r] * nu_eff_global)
            if verbose:
                print("nueff {:.1f} => {:.1f}".format(nu_effs[r], nu_eff_reg_r))
            Gamma += nu_eff_reg_r * Gamma_r
            Y += nu_eff_reg_r * Y_r

    if verbose:
        print("[global nu_eff]  {:.1f}".format(nu_eff_global))
        print("[global nu_eff]  {:.1f} {:.1f} {:.1f}".format(np.min(nu_effs), np.max(nu_effs), np.median(nu_effs)))
        print("Included a total of {} / {} genomes".format(N_kept, int(N_sum.sum().item())))

    assert Y.shape == (num_alleles,)
    assert Gamma.shape == (num_alleles, num_alleles)

    return Y, Gamma


def compute_y_gamma(N, genotype, locations, args, phi=None, window=3, verbose=1):
    if phi is None:
        return _compute_y_gamma(N, genotype, locations, args, phi=None, window=window, verbose=verbose)
    else:
        Y, Gamma = _compute_y_gamma(N, genotype, locations, args, phi=None, window=window, verbose=verbose)
        Y_phi, Gamma_cross = _compute_y_gamma(N, genotype, locations, args, phi=phi, window=window, verbose=0)
        _, Gamma_phi = _compute_y_gamma(N, genotype, locations, args, phi=phi.pow(2.0), window=window, verbose=0)

    A = genotype.size(-1)
    Gamma_full = Gamma.new_zeros(2 * A, 2 * A)

    Gamma_full[:A, :A] = Gamma
    Gamma_full[A:, A:] = Gamma_phi
    Gamma_full[:A, A:] = Gamma_cross
    Gamma_full[A:, :A] = Gamma_cross

    Y_full = torch.cat([Y, Y_phi], dim=-1)

    assert Gamma_full.shape == (2 * A, 2 * A)
    assert Y_full.shape == (2 * A,)

    return Y_full, Gamma_full


def main(args):
    data = torch.load(args.pyrocov_dir + args.filename)
    print("Loading {}".format(args.pyrocov_dir + args.filename))
    features = data['features']     # C F
    print("features.shape: ", features.shape)

    mutations = data['mutations']   # F
    counts = data['weekly_clades']  # T R C
    location_id_inv = np.array(data['location_id_inv'])

    big_regions = counts.sum(0).sum(-1) >= args.min_total_samples

    if args.phi != 'none':  # remove Luxembourg if using vaccination data
        lux_id = data['location_id']['Europe / Luxembourg']
        big_regions[lux_id] = 0

    print("# regions with {} samples: {}".format(args.min_total_samples, big_regions.sum().item()))
    locations = location_id_inv[big_regions.data.cpu().numpy()]
    print(list(set([l.split(' / ')[1] for l in locations])))

    counts = counts[:, big_regions]
    print("total count: ", counts.sum().item())

    if args.phi != 'none':
        if args.phi == 'vaccinated':
            rates = pd.read_csv('vaccine_timeseries.people_vaccinated_per_hundred.csv', index_col=0)
        elif args.phi == 'fully':
            rates = pd.read_csv('vaccine_timeseries.people_fully_vaccinated_per_hundred.csv', index_col=0)
        phi = np.stack([rates[rates.index == loc].values[0] for loc in locations])
        phi = torch.from_numpy(phi).to(counts.device).type_as(counts)
        assert phi.shape == (counts.shape[1], counts.shape[0])
        mutations += ['VAC:' + m for m in mutations]
    else:
        phi = None

    lineage_id_inv = data['lineage_id_inv']
    clade_id_to_lineage_id = data['clade_id_to_lineage_id']
    pango_idx = [lineage_id_inv[clade_id_to_lineage_id[c]] for c in range(clade_id_to_lineage_id.size(0))]
    assert len(lineage_id_inv) == 1544
    assert clade_id_to_lineage_id.size(0) == len(pango_idx) == 3000

    Y, Gamma = compute_y_gamma(counts.transpose(0, 1), features, locations, args, phi=phi)

    lineage_to_clade = defaultdict(list)
    for k, v in data['clade_to_lineage'].items():
        lineage_to_clade[v].append(k)

    features_pt = torch.load(args.pyrocov_dir + 'features.3000.1.pt')
    clades = features_pt['clades']

    data = {'Gamma': Gamma, 'Y': Y, 'num_alleles': Y.size(-1),
            'num_regions': counts.size(1), 'mutations': mutations,
            'genotype': features, 'counts': counts, 'pango_idx': pango_idx}

    day = int(args.filename.split('.')[-2])
    f = 'processed_data.day{}.mts{}k.mbs{}.s{}.{}.{}.pt'
    f = f.format(day, args.min_total_samples // 1000, args.min_biweekly_samples, args.seed,
                 args.strategy, args.phi)
    torch.save(data, f)
    print("Saved output to {}.".format(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulator')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filename', type=str, default='mutrans.data.single.3000.1.50.None.pt')
    parser.add_argument('--pyrocov-dir', type=str, default='/home/mjankowi/pyro-cov/results/')
    parser.add_argument('--min-total-samples', type=int, default=15 * 10 ** 3)
    parser.add_argument('--min-biweekly-samples', type=int, default=100)
    parser.add_argument('--phi', type=str, default='none', choices=['none', 'vaccinated', 'fully'])
    parser.add_argument('--strategy', type=str, default='global',
                        choices=['global', 'regional', 'regionalreg'])
    args = parser.parse_args()

    main(args)

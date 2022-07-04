import argparse

import torch

from bvas import BVASSelector

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


def run(filename, S=50.0, nu_eff_multiplier=0.5, T=500 * 1000, tau=100.0, T_burnin=10 * 1000, seed=0):
    data = torch.load(filename)
    mutations = data['mutations']

    s = "Running inference on {} with {} regions and {} selection coefficients..."
    print(s.format(filename, data['num_regions'], len(mutations)))

    selector = BVASSelector(data['Y'].cuda().double(),
                            data['Gamma'].cuda().double(),
                            mutations,
                            S=S,
                            nu_eff_multiplier=nu_eff_multiplier,
                            tau=tau,
                            genotype_matrix=data['genotype'].cuda().double(),
                            variant_names=data['pango_idx'])

    selector.run(T=T, T_burnin=T_burnin, seed=seed)
    summary, growth_rates = selector.summary, selector.growth_rates

    print(summary.iloc[:20])
    print('[pipsum] {:.2f}'.format(summary.PIP.sum().item()))

    tag = ".S_{}.nueff_{}.tau_{}.T_{}_{}.s{}".format(S, int(100 * nu_eff_multiplier), tau, T, T_burnin, seed)

    f = 'summary.' + filename[15:-3] + tag + '.csv'
    print("saving csv to {}".format(f))
    summary.to_csv(f)

    f = 'growthrates.' + filename[15:-3] + tag + '.csv'
    print("saving csv to {}".format(f))
    growth_rates.to_csv(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BVAS')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--T', type=int, default=500 * 1000)
    parser.add_argument('--Tb', type=int, default=20 * 1000)
    parser.add_argument('--nu-eff', type=float, default=0.5)
    parser.add_argument('--S', type=float, default=200.0)
    parser.add_argument('--tau', type=float, default=50.0)
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    run(args.filename, S=args.S, tau=args.tau, T=args.T, T_burnin=args.Tb,
        nu_eff_multiplier=args.nu_eff, seed=args.seed)

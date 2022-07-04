import argparse

import numpy as np
import pandas as pd
import torch


def run(data, inference, out):
    data = torch.load(data, map_location='cpu')
    df = pd.DataFrame(np.array(data['genotype'], dtype=np.int8), columns=data['mutations'])
    df['PANGO'] = data['pango_idx']

    inf = pd.read_csv(inference, index_col=0)
    assert inf.values.shape[0] == df.values.shape[0]
    df['GrowthRateMean'] = inf['GrowthRate']
    df['GrowthRateStd'] = inf['GrowthRateStd']
    df['ExceedsProb'] = inf['ExceedsProb']
    assert (np.array(data['pango_idx']) == inf['Variant Name']).sum().item() == df.values.shape[0]

    print(df)
    df.to_csv(out + '.csv.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract genotype')
    parser.add_argument('--data', type=str)
    parser.add_argument('--inference', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    run(args.data, args.inference, args.out)

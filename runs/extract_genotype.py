import argparse
import pandas as pd
import torch


def run(filename):
    data = torch.load(filename, map_location='cpu')
    df = pd.DataFrame(data['genotype'], columns=data['mutations'])
    df['PANGO'] = data['pango_idx']
    df.to_csv('genotype.csv.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract genotype')
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    run(args.filename)

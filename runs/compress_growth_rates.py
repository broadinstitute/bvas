import argparse

import numpy as np
import pandas as pd


def run(filename):
    df = pd.read_csv(filename, index_col=0)
    pangos = set(df['Variant Name'].tolist())

    rates = []

    for pango in pangos:
        df_pango = df[df['Variant Name'] == pango]
        means = df_pango.GrowthRate.values
        mean = means.mean()
        var = np.clip(np.square(df_pango.GrowthRateStd.values).mean() +
                      np.square(means).mean() - np.square(mean), a_min=0.0, a_max=None)
        rates.append( (pango, mean, np.sqrt(var)) )

    df = pd.DataFrame({'VariantName':[r[0] for r in rates], 
                       'GrowthRate':[r[1] for r in rates], 
                       'GrowthRateStd':[r[2] for r in rates]})
    df = df.sort_values(by=['GrowthRate'], ascending=False)
    df = df.set_index('VariantName')
    df['Rank'] = 1 + np.arange(df.values.shape[0])
    print(df.iloc[:25])

    df.to_csv('compressed.' + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compress a growth_rates.*.csv file')
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    run(args.filename)

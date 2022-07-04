import argparse
import pandas as pd

from scipy.stats import norm


def run(current, baseline, base_lineage, base_prob):
    assert base_prob > 0.0 and base_prob < 1.0

    df_current = pd.read_csv(current, index_col=0)

    df_baseline = pd.read_csv(baseline, index_col=0)
    df_baseline = df_baseline[df_baseline['Variant Name'] == base_lineage]
    df_baseline = df_baseline.sort_values(by=['GrowthRate'], ascending=False)
    growth_rate_mean = df_baseline.iloc[0]['GrowthRate'].item()
    growth_rate_std = df_baseline.iloc[0]['GrowthRateStd'].item()

    print("Growth rate for {}: {:.3f} +- {:.3f}".format(base_lineage, growth_rate_mean, growth_rate_std))
    baseline = norm.isf(1.0 - base_prob, loc=growth_rate_mean, scale=growth_rate_std)
    print("Growth rate for {} at probability of {:.3f} is {:.3f}".format(base_lineage, base_prob, baseline))

    exceeds_prob = norm.sf(baseline, loc=df_current['GrowthRate'].values, scale=df_current['GrowthRateStd'].values)
    df_current['ExceedsProb'] = exceeds_prob

    print("Minimum exceeds probability: {:.3e}".format(exceeds_prob.min()))
    print("Maximum exceeds probability: {:.3e}".format(exceeds_prob.max()))

    for thresh in [0.1, 0.5, 0.9, 0.99]:
        num = exceeds_prob[exceeds_prob > thresh].shape[0]
        print("# of lineages with an exceed probability of {:.2f} or higher: {}".format(thresh, num))

    print("\nLineages with an exceed probability of 0.9 or higher:")
    print(df_current['Variant Name'].values[exceeds_prob > 0.9])

    df_current.to_csv('annotated.' + current)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute exceeds probability')
    parser.add_argument('--current', type=str)
    parser.add_argument('--baseline', type=str)
    parser.add_argument('--base-lineage', type=str, default='BA.2.12.1')
    parser.add_argument('--base-prob', type=float, default=0.75)
    args = parser.parse_args()

    run(args.current, args.baseline, args.base_lineage, args.base_prob)

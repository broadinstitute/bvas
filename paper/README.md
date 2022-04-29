
## BVAS: Bayesian Viral Allele Selection 

This directory contains some of the raw inference results and main figures from
our paper, `Inferring selections effects in SARS-CoV-2 with Bayesian Viral Allele Selection`.

Raw inference results:
 - [allele_summary.csv](allele_summary.csv): Inference results for allele-level selection coefficients beta, ranked by PIP. The zeroth column is the allele (i.e. amino acid mutation). The first column reports the Posterior Inclusion Probability (PIP) for each allele. The second column reports the posterior mean of beta. The third column reports the posterior standard deviation of beta. The fourth column reports the posterior mean of beta conditioned on the inclusion of that allele in the model (i.e. the corresponding `gamma=1`). The fifth column reports the posterior standard deviation of beta conditioned on the inclusion of that allele in the model (i.e. the corresponding `gamma=1`). The sixth column reports the PIP rank.
 - [growth_rates_summary.csv](growth_rates_summary.csv): Inference results for lineage-level (relative) growth rates. The zeroth column reports the PANGO lineage designation. The first column reports the posterior mean of the growth rate R. The second column reports the posterior standard deviation of the growth rate R.

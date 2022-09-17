
## BVAS: Bayesian Viral Allele Selection 

This directory contains some of the raw inference results and main figures from
our paper, `Inferring selections effects in SARS-CoV-2 with Bayesian Viral Allele Selection`.

Raw inference results:
 - [allele_summary.csv](allele_summary.csv): Inference results for allele-level selection coefficients beta, ranked by PIP. The zeroth column is the allele (i.e. amino acid mutation). The first column reports the Posterior Inclusion Probability (PIP) for each allele. The second column reports the posterior mean of beta. The third column reports the posterior standard deviation of beta. The fourth column reports the posterior mean of beta conditioned on the inclusion of that allele in the model (i.e. the corresponding `gamma=1`). The fifth column reports the posterior standard deviation of beta conditioned on the inclusion of that allele in the model (i.e. the corresponding `gamma=1`). The sixth column reports the PIP rank.
 - [growth_rates_summary.csv](growth_rates_summary.csv): Inference results for lineage-level (relative) growth rates. The zeroth column reports the PANGO lineage designation. The first column reports the posterior mean of the growth rate R. The second column reports the posterior standard deviation of the growth rate R.
 - [accession_ids.txt.xz](accession_ids.txt.xz): Complete list of GISAID accession numbers of viral genomes used in our study. 
 - [selected_growth_rates.png](selected_growth_rates.png): Table of estimated growth rates, relative to Wuhan A, for a few select SARS-CoV-2 lineages. 

More recent inference results are to be found in directories named after the end date of the surveillance data used in the analysis, starting with August 10th: [08.10.22](08.10.22/).

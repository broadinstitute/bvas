[![Build Status](https://github.com/broadinstitute/bvas/workflows/CI/badge.svg)](https://github.com/broadinstitute/bvas/actions)
[![Docs](https://img.shields.io/badge/api-docs-blue)](https://broadinstitute.github.io/bvas/)


<iframe width="50%" src="https://github.com/broadinstitute/bvas/blob/main/paper/manhattan.genome_wide.pdf">


# BVAS: Bayesian Viral Allele Selection 

Welcome to the GitHub repository for [*Inferring selection effects in SARS-CoV-2 with Bayesian Viral Allele Selection*](https://www.biorxiv.org/content/10.1101/2022.05.07.490748v1).


## Requirements

BVAS requires Python 3.8 or later and the following Python packages: [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [pyro](https://github.com/pyro-ppl/pyro).

Note that if you wish to run BVAS on a GPU you need to install PyTorch with CUDA support.
In particular if you run the following command from your terminal it should report True:
```
python -c 'import torch; print(torch.cuda.is_available())'
```


## Installation instructions

Install directly from GitHub:

```pip install git+https://github.com/broadinstitute/bvas.git```

Install from source:
```
git clone git@github.com:broadinstitute/bvas.git
cd bvas 
pip install .
```


## Documentation

The documentation is available [here](https://broadinstitute.github.io/bvas/).


## Repo organization 

This repo is organized as follows:
 - [bvas](bvas/): all the core code: inference algorithms and simulations
 - [notebooks](notebooks/): Jupyter notebooks demonstrating BVAS usage
   - [basic_demo.ipynb](notebooks/basic_demo.ipynb): demo using simulated data 
   - [S_gene_demo.ipynb](notebooks/S_gene_demo.ipynb): demo using GISAID data restricted only to the SARS-CoV-2 S gene
 - [data](data/): pre-processing scripts and (some of the) data used in the analysis
 - [tests](tests/): unit tests for verifying the correctness of inference algorithms and other code

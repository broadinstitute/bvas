# bvas: bayesian viral allele selection 

Welcome to the Github repository for *Inferring selection effects in SARS-CoV-2 with Bayesian Viral Allele Selection*.

This repo is organized as follows:
 - [bvas](bvas/): all the core code: inference algorithms and simulations
 - [notebooks](notebooks/): Jupyter notebooks demonstrate BVAS usage
   - [basic_demo.ipynb](notebooks/basic_demo.ipynb): demo using simulated data 
   - [S_gene_demo.ipynb](notebooks/S_gene_demo.ipynb): demo using GISAID data restricted only to the SARS-CoV-2 S gene
 - [data](data/): pre-processing scripts and (some of the) data used in the analysis
 - [tests](tessts/): unit tests for verifying the correctness of inference algorithms and other code

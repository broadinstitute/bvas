:github_url: https://github.com/broadinstitute/bvas

BVAS Documentation
==================

This is the documentation for BVAS, a package for inferring selection effects
from genomic surveillance data using Bayesian methods, in particular 
Bayesian Viral Allele Selection.
Please see the `GitHub repo <https://github.com/broadinstitute/bvas>`__ for more details.

Requirements
-------------

BVAS requires Python 3.8 or later and the following Python packages:
`PyTorch <https://pytorch.org/>`__,
`pandas <https://pandas.pydata.org>`__, and
`Pyro <https://github.com/pyro-ppl/pyro>`__.

Note that if you wish to run BVAS on a GPU you need to install PyTorch with CUDA support.
In particular if you run the following command from your terminal it should report True:

::

    python -c 'import torch; print(torch.cuda.is_available())'


Installation instructions
-------------------------

Install directly from GitHub:

::

    pip install git+https://github.com/broadinstitute/bvas.git

Install from source:

::

    git clone git@github.com:broadinstitute/bvas.git
    cd bvas 
    pip install .


Basic usage
-----------

The main functionality of BVAS is available through the :class:`BVASSelector` class.
See the Jupyter notebooks in the `notebooks <https://github.com/broadinstitute/bvas/tree/master/notebooks>`__ directory for detailed example usage.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    selector.rst
    sampler.rst
    simulation.rst
    map.rst
    laplace.rst

Index
=====

* :ref:`genindex`

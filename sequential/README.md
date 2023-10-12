# A Parallel Combinatorial Algorithm for Approximating the Optimal Transport

This repository contains the implementation of sequential algorithm in section 2 from our paper and experiment code for the Appendix C.3.

This repository contains three parts: 

1. Implementation of transport and assignment algorithm (section 2): `transport.py`, `matching.py`
2. Experiments compare our method with Sinkhorn (Appendix C.3): `pl_vs_sinkorn_ot_synthetic`

### Dependencies

To use our algorithm or reproduce our experiments, simply install the following dependencies in your python environment and run the code.

For the first part, our algorithm implementation requires:

- [NumPy](https://numpy.org/install/) v1.21 
- [PyTorch](https://pytorch.org/) v1.10

Reproducing our experiments requires:

- [NumPy](https://numpy.org/install/) v1.21
- [POT](https://pythonot.github.io/) v0.8.1
- [PyTorch](https://pytorch.org/) v1.10
- [scikit-learn](https://scikit-learn.org/stable/install.html) v0.24.2


### To Run Experiments Comparing with Sinkhorn
Synthetic Data OT

    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.05
    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.01
    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.005
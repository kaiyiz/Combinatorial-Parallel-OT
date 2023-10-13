# A Parallel Combinatorial Algorithm for Approximating the Optimal Transport

This directory contains the implementation of sequential algorithm in section 2 from our paper and experiment code for the Appendix C.3.

1. Implementation of transport algorithm (section 2): `transport.py`
2. Experiments compare our method with Sinkhorn (Appendix C.3): `pl_vs_sinkorn_ot_synthetic`


### To Run Experiments Comparing with Sinkhorn
Synthetic Data OT

    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.05
    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.01
    python pl_vs_sinkorn_ot_synthetic.py --nexp 10 --n 10000 --delta 0.005
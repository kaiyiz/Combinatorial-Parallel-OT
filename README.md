# A Parallel Combinatorial Algorithm for Approximating the Optimal Transport

This repository contains the implementation of parallel algorithm in section 4 from our paper and experiment code for the section 5 our paper.

This repository contains three parts: 

1. Implementation of transport and assignment algorithm (section 4): `transport.py`, `matching.py`
2. Experiments compare our method and Sinkhorn (section 5): `plgpu_vs_sinkorn_bench.py`,`plgpu_vs_sinkorn_bench_rev.py`
3. Experiments compare our method and DROT (Appendix C.2): `plgpu_vs_drot_bench_step1.py`,`plgpu_vs_drot_bench_step2.py`

### Dependencies

To use our algorithm or reproduce our experiments, simply install the following dependencies in your python environment and run the code.

For the first part, our algorithm implementation requires:

- [NumPy](https://numpy.org/install/) v1.21 
- [PyTorch](https://pytorch.org/) v1.10

Reproducing our experiments requires:

- [NumPy](https://numpy.org/install/) v1.21
- [POT](https://pythonot.github.io/) v0.8.1
- [PyTorch](https://pytorch.org/) v1.10
- [NLTK](https://github.com/nltk/nltk)
- [scikit-learn](https://scikit-learn.org/stable/install.html) v0.24.2
- [pybind11](https://pybind11.readthedocs.io/en/stable/installing.html) v2.7.1

### Datasets

To run the experiments in this repository, you need to download datasets in the anonymized link [here](https://osf.io/njvcb/?view_only=912f7fffa9564926b15988d2ea3c1bdd). 
And also download the glove embedding file from [here](https://nlp.stanford.edu/projects/glove/).

### To Run Experiments Comparing with Sinkhorn
Synthetic Data OT

    python plgpu_vs_sinkhorn_bench.py --nexp 10 --n 10000 --dataset_name synthetic_OT --is_transport 1 --delta_num 10 --delta_low 0.0007 --delta_high 0.1

Synthetic Data OT (reverse)

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 10 --n 10000 --dataset_name synthetic_OT --is_transport 1 --reg_num 10 --reg_low 0.00015 --reg_high 0.01

Synthetic Data Assignment

    python plgpu_vs_sinkhorn_bench.py --nexp 10 --n 10000 --dataset_name synthetic_matching  --delta_num 10 --delta_low 0.0007 --delta_high 0.01 --is_transport 0

Synthetic Data Assignment (reverse)

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 10 --n 10000 --dataset_name synthetic_matching --reg_num 10 --reg_low 0.00045 --reg_high 0.01 --is_transport 0

MNIST Data Assignment

    python plgpu_vs_sinkhorn_bench.py --nexp 10 --n 10000 --dataset_name mnist_matching  --delta_num 10 --delta_low 0.02 --delta_high 0.2 --is_transport 0

MNIST Data Assignment (reverse)

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 10 --n 10000 --dataset_name mnist_matching --reg_num 10 --reg_low 0.002 --reg_high 0.02 --is_transport 0

NLP Data 

the count of monte cristo

    python plgpu_vs_sinkhorn_bench.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name the-count-of-monte-cristo --metric euclidean --nlp_portion_size 2000

IMDB

    python plgpu_vs_sinkhorn_bench.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name IMDB --metric euclidean --nlp_portion_size 100

20NEWS

    python plgpu_vs_sinkhorn_bench.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name 20news --metric euclidean --nlp_portion_size 3000

NLP Data (reverse)

the count of monte cristo

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --reg_num 10 --reg_low 0.001 --reg_high 0.1 --nlp_name the-count-of-monte-cristo --metric euclidean --nlp_portion_size 2000

IMDB

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --reg_num 10 --reg_low 0.001 --reg_high 0.1 --nlp_name 20news --metric euclidean --nlp_portion_size 3000

20NEWS

    python plgpu_vs_sinkhorn_bench_rev.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --reg_num 10 --reg_low 0.001 --reg_high 0.1 --nlp_name IMDB --metric euclidean --nlp_portion_size 100

### To Run Experiments Comparing with DROT

Synthetic Data OT

    python plgpu_vs_drot_bench_step1.py --nexp 10 --n 10000 --dataset_name synthetic_OT --is_transport 1 --delta_num 10 --delta_low 0.0001 --delta_high 0.01
    python plgpu_vs_drot_bench_step2.py --nexp 10 --n 10000 --dataset_name synthetic_OT --is_transport 1 --delta_num 10 --delta_low 0.0001 --delta_high 0.01

Synthetic Data Assignment

    python plgpu_vs_drot_bench_step1.py --nexp 10 --n 10000 --dataset_name synthetic_matching  --delta_num 10 --delta_low 0.0001 --delta_high 0.01 --is_transport 0
    python plgpu_vs_drot_bench_step2.py --nexp 10 --n 10000 --dataset_name synthetic_matching  --delta_num 10 --delta_low 0.0001 --delta_high 0.01 --is_transport 0

MNIST Data Assignment

    python plgpu_vs_drot_bench_step1.py --nexp 10 --n 10000 --dataset_name mnist_matching  --delta_num 10 --delta_low 0.02 --delta_high 0.2 --is_transport 0
    python plgpu_vs_drot_bench_step2.py --nexp 10 --n 10000 --dataset_name mnist_matching  --delta_num 10 --delta_low 0.02 --delta_high 0.2 --is_transport 0

NLP Data

the count of monte cristo

    python plgpu_vs_drot_bench_step1.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name the-count-of-monte-cristo --metric euclidean --nlp_portion_size 2000
    python plgpu_vs_drot_bench_step2.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name the-count-of-monte-cristo --metric euclidean --nlp_portion_size 2000

IMDB

    python plgpu_vs_drot_bench_step1.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name IMDB --metric euclidean --nlp_portion_size 100
    python plgpu_vs_drot_bench_step2.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name IMDB --metric euclidean --nlp_portion_size 100

20NEWS

    python plgpu_vs_drot_bench_step1.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name 20news --metric euclidean --nlp_portion_size 3000
    python plgpu_vs_drot_bench_step2.py --nexp 5 --dataset_name NLP_OT --is_transport 1 --delta_num 10 --delta_low 0.1 --delta_high 1 --nlp_name 20news --metric euclidean --nlp_portion_size 3000
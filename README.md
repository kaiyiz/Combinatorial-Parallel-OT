# A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings

This repository contains the implementation of our algorithms from our paper [OpenReview](https://openreview.net/forum?id=2izFpGERjU) and the experiment code.

Optimal Transport is a popular distance metric for measuring similarity between distributions. Exact and approximate combinatorial algorithms for computing the optimal transport distance are hard to parallelize. This has motivated the development of numerical solvers (e.g. Sinkhorn method) that can exploit GPU parallelism and produce approximate solutions.

We introduce the first parallel combinatorial algorithm to find an additive $\varepsilon$-approximation of the OT distance. The parallel complexity of our algorithm is $O(\log(n)/ \varepsilon^2)$ where $n$ is the total support size for the input distributions. In Massive Parallel Computation (MPC) frameworks such as Hadoop and MapReduce, our algorithm computes an $\varepsilon$-approximate transport plan in $O(\log (\log (n/\varepsilon))/\varepsilon^2)$ rounds with $O(n/\varepsilon)$ space per machine; all prior algorithms in the MPC framework take $\Omega(\log n)$ rounds. 
We also provide a GPU-friendly matrix-based interpretation of our algorithm where each step of the algorithm is row or column manipulation of the matrix. Experiments suggest that our combinatorial algorithm is faster than the state-of-the-art approximate solvers in the GPU, especially for higher values of $n$. 

<p align="center" width="100%">
    <img src="figures/fig1.png"> 
</p>

Current directory contains three parts: 

1. Implementation of parallel transport and assignment algorithm (section 4): `transport.py`, `matching.py`
2. Experiments compare our method and Sinkhorn (section 5): `plgpu_vs_sinkorn_bench.py`,`plgpu_vs_sinkorn_bench_rev.py`
3. Experiments compare our method and DROT (Appendix C.2): `plgpu_vs_drot_bench_step1.py`,`plgpu_vs_drot_bench_step2.py`

## Dependencies

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

## Datasets

To run the experiments in this repository, please download datasets in the anonymized link [here](https://osf.io/njvcb/?view_only=912f7fffa9564926b15988d2ea3c1bdd). 
And also download the glove embedding file from [here](https://nlp.stanford.edu/projects/glove/).

## Citation
If you find this work helpful, please consider citing our paper:

```
@inproceedings{lahn2023combinatorial,
  title={A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings},
  author={Lahn, Nathaniel and Raghvendra, Sharath and Zhang, Kaiyi},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


## Experiments Reproduction
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
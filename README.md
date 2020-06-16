
==============================
# OTAlign

Code for paper "Gromov Wasserstein Alignment of Word Embedding Spaces".

Disclaimer: This codebase borrows some embbedding and evaluation tools from Mikel Artetxe's [vecmap](https://github.com/artetxem/vecmap) repo, and relies on the Gromov-Wasserstein implementation of the Python Optimal Transport [POT](https://github.com/rflamary/POT) from Remi Flamary and colleagues.

## Dependencies

#### Major:
* python (>3.0)
* numpy (>1.15)
* [POT](https://github.com/rflamary/POT) (>0.5)
* (OPTIONAL) [cupy](https://cupy.chainer.org) (for GPU computation)

#### Minor
* tqdm
* matplotlib

## Installation

It's highly recommended that the following steps be done **inside a virtual environment** (e.g., via `virtualenv` or `anaconda`).

Install this package
```
git clone git@github.com:dmelis/otalign.git
cd otalign
pip3 install -e ./
```

## Getting Datasets

Data for the 'Conneau' task can be obtained via the [MUSE](https://github.com/facebookresearch/MUSE) repo, and data for the 'Dinu' task can be obtained via the [VecMap](https://github.com/artetxem/vecmap) repo.

Copy data to local dirs (alternatively, the paths can be explicitly provided via arguments).

```
cp -r /path/to/MUSE/dir/data/* ./data/raw/MUSE/
cp -r /path/to/dinu/dir/data/* ./data/raw/dinu/

```

## How to use

```
python scripts/main_gw_bli.py --task conneau --src en --trg es --maxiter 50
```

## Issues

TODO: POT recently moved from cudamat to cupy for GPU comptuation, which broke this code. It can currently be run on small subsets of the tasks, but will need to fix CUDA dependencies to solve full problems.

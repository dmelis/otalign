
==============================
# OTAlign

Code for paper "Gromov Wasserstein Alignment of Word Embedding Spaces"

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

Copy data to local dirs (Optional, can also be specified via arguments)

```
cp -r /path/to/MUSE/dir/data/* ./data/raw/MUSE/
cp -r /path/to/dinu/dir/data/* ./data/raw/dinu/

```

## How to use

```
python scripts/main_gw_bli.py --src en --trg es
```

## Issues

TODO: POT recently moved from cudamat to cupy for GPU comptuation, which broke this code. It can currently be run on small subsets of the tasks, but will need to fix CUDA dependencies to solve full problems.

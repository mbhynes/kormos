# `kormos`

An interface between [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) and [`keras`](https://keras.io/) for training `keras` models with traditional (deterministic) minimization algorithms.

## Motivation

Tensorflow (and Keras) provide a powerful tool for developing and optimizing parameters in predictive models in a high-level language.
Its most powerful feature is arguably the auto-differentiation utilities to support the use of gradient-based minimization algorithms, which enable rapid prototyping by removing the need to code analytic gradient functions for a given loss function.

While Tensorflow is indeed a framework for large-scale deep learning, its expressiveness and prototyping capabilities make it attractive as a framework to construct small-scale models and optimization problems as well, for which datasets can reasonably fit in memory on a single machine and staple *deterministic* algorithms like [L-BFGS](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb) or [Newton-CG](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html#optimize-minimize-newtoncg) are applicable, and can be preferable to *stochastic* optimization algorithms since: 
  - these algorithms have much faster theoretical convergence to accurate solutions when the dataset is small enough to make computation times feasible
  - deterministic algorithms do not require additional hyperparamter tuning to guarantee convergence (in theory or in practice)

So TL;DR: because Luddites exist even in the field of numerical optimization.

## Why The Name Kormos?

Because `keras` is a powerful and useful tool, and is named after the Greek word *κέρας*, which means *horn*.

This package is related to `keras`, but isn't very powerful or useful. It's named after the Greek word *κορμός*, which means *stump*.

## Installation

Install via the PyPI package [kormos](https://pypi.org/project/kormos/) using pip:

```python
pip3 install kormos
```

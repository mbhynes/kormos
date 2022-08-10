`kormos`
=================================

The `kormos` package provides an interface between `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ and `keras <https://keras.io>`_ for training `keras` models with traditional deterministic minimization algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Motivation
~~~~~~~~~~~

The `keras` framework provides a powerful tool for developing and optimizing parameters in predictive models in a high-level language.
Its most powerful feature is arguably the underlying tensorflow auto-differentiation utilities to support the use of gradient-based minimization algorithms, which enable rapid prototyping by removing the need to code analytic gradient functions for a given loss function.

While `keras` is a framework for large-scale deep learning, its expressiveness and prototyping capabilities make it attractive as a framework to construct small-scale models and optimization problems as well, for which datasets can reasonably fit in memory on a single machine and staple *deterministic* algorithms like `L-BFGS <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_ or `Newton-CG <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-newtoncg.html#optimize-minimize-newtoncg>`_ are applicable, and can be preferable to *stochastic* optimization algorithms since: 

  - these algorithms have much faster theoretical convergence to accurate solutions when the dataset is small enough to make computation times feasible
  - deterministic algorithms do not require additional hyperparamter tuning to guarantee convergence (in theory or in practice)

So TL;DR: because Luddites exist even in the field of numerical optimization.

Why The Name Kormos?
~~~~~~~~~~~~~~~~~~~~

Because `keras` is a powerful and useful tool, and is named after the Greek word *κέρας*, which means *horn*.

This package is related to `keras`, but isn't very powerful or useful. It's named after the Greek word *κορμός*, which means *stump*.

Installation
~~~~~~~~~~~~

Install via the PyPI package [kormos](https://pypi.org/project/kormos/) using pip:

.. code-block:: python

  pip3 install kormos

The `skelo` package is an implementation of the `Elo <https://en.wikipedia.org/wiki/Elo_rating_system>`_ and `Glicko2 <https://en.wikipedia.org/wiki/Glicko_rating_system>`_ rating systems with a `scikit-learn <https://scikit-learn.org/stable>`_ compatible interface.

The `skelo` package is a simple implementation suitable for small-scale rating systems that fit into memory on a single machine.
It's intended to provide a convenient API for creating Elo/Glicko ratings in a data science & analytics workflow for small games on the scale thousands of players and millions of matches, primarily as a means of feature transformation in other `sklearn` pipelines or benchmarking classifier accuracy.

License 
=======
This project is released under the MIT license. Please see the LICENSE header in the source code for more details.

Acknowledgements & Related Work
================================
This package was created as a means to an end in other projects, and has adapted code from the following sources:

- `Pi-Yueh Chuang's <https://pychao.com/contact-us-and-pgp-key/>`_ MIT-licensed `scipy.optimize.minimize_lbfgs` wrapper available on github `here <https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993>`_.
- `Allen Lavoie's <https://github.com/allenlavoie>`_ Hessian-vector-product routines in `tensorflow`, available on github `here <https://github.com/tensorflow/tensorflow/commit/5b37e7ed14eb7dddae8a0e87435595347a315bb7>`_ under the Apache License version 2.

There is also a project `keras-opt <https://github.com/pedro-r-marques/keras-opt>`_  authored by `Pedro Marques <https://github.com/pedro-r-marques>`_ with an identical goal but different implementation and API.

API Reference
=============

Keras Models
~~~~~~~~~~~~
.. automodule:: kormos.model
   :members:

Function Factory for `scipy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: kormos.utils.scipy
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

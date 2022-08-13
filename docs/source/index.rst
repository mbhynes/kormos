`kormos`
=================================

The `kormos` package provides an interface between `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ and `keras <https://keras.io>`_ for training `keras` models with traditional deterministic minimization algorithms.

It provides `keras` users with:

- `keras.Model` subclasses that may be optimized without changes in the API---a model may be trained using *either* the built-in stochastic mini-batch algorithms *or* the deterministic batch algorithms from `scipy.optimize`
- Out-of-the-box Interoperability with the usual `keras` utilities, e.g.:
  - The model `fit <https://keras.io/api/models/model_training_apis/#fit-method>`_ method is still usable by `KerasTuner <https://keras.io/keras_tuner/>`_
  - `fit` still returns a the history object with optimization metadata and validation metrics at each (batch) iteration

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

Requirements
~~~~~~~~~~~~
`kormos` runs on `python3` with `tensorflow 2+` and the respective `keras` engine packaged with it.

Installation
~~~~~~~~~~~~

Install via the PyPI package [kormos](https://pypi.org/project/kormos/) using pip:

.. code-block:: python

  pip3 install kormos

Alternatively, if you like your releases bloody rare you may install from:

.. code-block:: python

  pip3 install git+https://github.com/mbhynes/kormos


License 
=======
This project is released under the MIT license, and contains adaptations of other codes released under the Apache and MIT licenses.
Please see the header in each source file for the applicable license and copyright notices therein. 

Acknowledgements & Related Work
================================
This package has adapted code from the following sources:

- `Pi-Yueh Chuang's <https://pychao.com/contact-us-and-pgp-key/>`_ MIT-licensed `scipy.optimize.minimize_lbfgs` wrapper available on github `here <https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993>`_.
- `Allen Lavoie's <https://github.com/allenlavoie>`_ Hessian-vector-product routines in `tensorflow`, available on github `here <https://github.com/tensorflow/tensorflow/commit/5b37e7ed14eb7dddae8a0e87435595347a315bb7>`_ under the Apache License version 2.

There is also a related project `keras-opt <https://github.com/pedro-r-marques/keras-opt>`_ authored by `Pedro Marques <https://github.com/pedro-r-marques>`_ with a similar goal but unrelated implementation and API. The `kormos` package is recommended over `keras-opt` for users that need interoperability with utilities like `keras_tuner <https://keras.io/keras_tuner/>`_ or are training models with heavier memory requirements and large datasets.

API Reference
=============

Models
~~~~~~~~~~~~
.. automodule:: kormos.models
   :members:

Optimizers
~~~~~~~~~~~~
.. automodule:: kormos.optimizers
   :members:


Caching Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `kormos.optimizers.BatchOptimizer` uses a simple dictionary cache for the `func_and_grad` method with the signature as described below.

.. automodule:: kormos.utils.cache
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

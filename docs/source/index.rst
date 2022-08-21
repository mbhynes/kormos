.. Import the body of this document from the repository README.

.. include:: ../../README.rst

API Reference
=============

Models
------------
.. automodule:: kormos.models
   :members:

Optimizers
------------
.. automodule:: kormos.optimizers
   :members:


Caching Utilities
------------------

The `kormos.optimizers.BatchOptimizer` uses a simple dictionary cache for the `func_and_grad` method with the signature as described below.
Please be aware that this cache utility is bespoke and not intended for downstream usage outside of the `BatchOptimizer`.

.. automodule:: kormos.utils.cache
   :members:

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

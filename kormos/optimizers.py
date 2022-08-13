# MIT License
#
# Copyright (c) 2022 Michael B Hynes
# Copyright (c) 2019 Pi-Yueh Chuang <pychuang@gwu.edu>, where noted.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy
import logging

import numpy as np
import scipy
from scipy.optimize._minimize import MINIMIZE_METHODS

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine import data_adapter

from kormos.utils.cache import OptimizationStateCache

logger = logging.getLogger(__name__)

OPTIMIZER_IDENTIFIERS = set(['scipy'] + [
  method.lower() for method in MINIMIZE_METHODS
] + [
  method.upper() for method in MINIMIZE_METHODS
])


class BatchOptimizer(object):
  """
  An optimizer class providing function callables for minimization by batch optimization algorithms.

  Codes implementing batch minimization routines, such `scipy.optimize.minimize`,
  typically have a call signature like the following:

  .. code-block:: python

    minimize(
      fun,        # callable, the objective function f(x) to minimize 
      ...
      jac=None,   # callable, the jacobian of f(x)
      hessp=None, # callable implementing a hessian/vector product for the hessian of f(x)
      ...
    )

  The `BatchOptimizer` factory class provides methods with the appropriate signature for the above callables:
  
  - `fun`, in the method `func(x)` or `func_and_grad(x)`
  - `jac`, in the method `grad(x)` or `func_and_grad(x)`
  - `hessp`, in the method `hessp(x, vector)`

  This class implements the callables above for a provided `keras.Model` and training dataset,
  in which the execution paradigm is for *full batch* gradient-based optimization, in which
  every sample in a provided dataset is used to compute the loss function and gradient values.

  The `BatchOptimizer` class implements the function and gradient by evaluating them over
  mini-batches of data and accumulating the result of each minibatch such that the computation
  is robust even when the training dataset is large or the model requires lots of RAM or
  GPU memory. To ensure the correct summation, the model's `loss` function attribute
  must have the `reduction=keras.losses.Reduction.SUM` explicitly set (after which
  the total sum will be averaged).

  Child classes extending this base class must implement `minimize`, which is the method
  that implements a gradient-based (or Hessian-based) optimization algorithm.
  """
  
  def __init__(self,
    name="batchoptimizer", 
    dtype=keras.backend.floatx(),
    batch_size=2**10,
    max_cache_entries=5,
  ):
    """
    Construct a `BatchOptimizer` object, but do not build it.
    Please note that the `build` method must be called before parameter
    optimization may be performed on a model.

    The `dtype` of a model's backing tensors when converting to an `numpy.ndarray`
    may be explicitly specified. In general, `tensorflow.float64` will make the
    optimization routine more numerically robust for loss functions with flat
    regions, and as such double precision is recommended for the tensorflow model
    itself. While single precision (`tensorflow.float32`) is often used in deep
    learning applications, numerical minimization routines that use line searches
    may require double precision (`tensorflow.float64`) for Wolfe convergence
    criteria (sufficient decrease) to be numerically resolvable.

    The choice of `batch_size` here is different than in
    stochastic optimization in which the optimizer typically performs best with
    small mini-batches on the order of ``2^4`` or ``2^5``. The `batch_size` here
    should be determined based on the available machine memory, and good values
    will be on the order of ``2^12`` or larger, depending on the memory requirements
    for the evaluation of the model (memory limitations are more likely to be
    encountered if second order algorithsm like the Newton-CG method are used).

    Args:
      dtype: the `tensorflow dtype` to use for weight matrices, defaults to `keras.backend.floatx()`.
      batch_size (int): the number of training examples to process in a single "mini-batch" if the 
        training dataset provided to `build()` must be converted to a `tensorflow.data.Dataset`
      max_cache_entries (int): the number of previous `loss/gradient/hessp` values to cache during training.
    """
    self.name = name
    self.dtype = dtype
    self.batch_size = batch_size

    self._built = False
    self.cache = OptimizationStateCache(max_entries=max_cache_entries)
    self.k = 0
    self.fg_evals = 0
    self.hessp_evals = 0

  def build(self, model, Xyw):
    """
    Dynamically build this `BatchOptimizer` object to prepare for
    a minimization routine for parameter fitting.

    The build process consists of storing the size metadata about this object's `model`'s
    `trainable_weights` such that we may convert between the stacked `numpy.ndarray` vector
    of parameters and the `model`'s list of multidimensional tensors.

    This method has been adapted from Pi-Yueh Chuang's MIT-licensed
    `code <https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993>`_,
    and this file retains the original 2019 copyright notice above.

    Aggs:
      model (keras.Model): the model to be optimized
      Xyw (tensorflow.data.Dataset): the training dataset for optimization. If a type other
        than `Dataset` is provided, this method will attempt to create one using the
        `BatchOptimizer`'s `batch_size` attribute. Providing a `Dataset` is recommended.
    Raises:
      ValueError: if the `model.loss.reduction` is not `kerass.losses.reduction.Reduction.SUM`
  
    """
    def _build_weight_indices():
      # obtain the shapes of all trainable parameters in the model
      shapes = tf.shape_n(model.trainable_weights)
      n_tensors = len(shapes)

      # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
      # prepare required information first
      idx_end = 0
      idx = [] # stitch indices
      part = [] # partition indices

      for part_num, shape in enumerate(shapes):
        num_elements = np.product(shape)
        idx_start = idx_end
        idx_end = idx_start + num_elements
        idx.append(tf.reshape(tf.range(idx_start, idx_end, dtype=tf.int32), shape))
        part.extend(num_elements * [part_num])

      self.part = tf.constant(part)
      self.idx = idx
      self.shapes = shapes
      self.n_tensors = n_tensors

    if not issubclass(type(model.loss), keras.losses.Loss):
      raise ValueError(
        f"Provided model.loss={model.loss} is type: {type(model.loss)}, but a keras.losses.Loss is required"
      )
    if model.loss.reduction != keras.losses.Reduction.SUM:
      raise ValueError(
        f"Provided model.loss.reduction = {model.loss.reduction}, "
        f"but should be: keras.losses.Reduction.SUM (={keras.losses.Reduction.SUM})"
      )
    if not issubclass(type(Xyw), tf.data.Dataset):
      logger.warning(
        f"Provided type of Xyw is '{type(Xyw)}; attempting to create "
        f"a data adapter with batch_size={self.batch_size}. "
        f"It is recommended you provide a tensorflow.data.Dataset directly."
      )
      if type(Xyw) is tuple:
        x, y, w = data_adapter.unpack_x_y_sample_weight(Xyw)
        Xyw = data_adapter.get_data_handler(
          x=x, y=y, sample_weight=w, model=model, batch_size=self.batch_size
        )._dataset
      else:
        Xyw = data_adapter.get_data_handler(x=Xyw, model=model, batch_size=self.batch_size)._dataset
      logger.warning(f"Converted input data to {type(Xyw)}")

    self.model = model
    self.Xyw = Xyw
    self.n = Xyw.cardinality().numpy()
    _build_weight_indices()
    self._built = True
    return self

  def _numpy_to_tensors(self, x):
    """
    Convert a (vector) `numpy.ndarray` of parameters to a list of tensors.

    Args:
      x (numpy.ndarray): vector of parameters used by `scipy.optimize.minimize`

    Returns:
      A list of tensors matching the shapes of this object's `model.trainable_weights`
    """
    assert self._built
    parts = tf.dynamic_partition(x, self.part, self.n_tensors)
    return [tf.reshape(part, self.shapes[k]) for (k, part) in enumerate(parts)]

  def _tensors_to_numpy(self, weights):
    """
    Convert a list of tensors to a (vector) `numpy.ndarray` of parameters.

    Args:
      weights: list of tensors matching the shapes of this object's `model.trainable_weights`

    Returns:
      numpy.ndarray: vector of parameters used by `scipy.optimize.minimize`
    """
    assert self._built
    return tf.dynamic_stitch(self.idx, weights).numpy()

  def get_weights(self):
    """
    Retrieve the `model` weights as a `numpy.ndarray` vector.

    Returns:
      numpy.ndarray: vector of parameters used by `scipy.optimize.minimize`
    """
    assert self._built
    x = self._tensors_to_numpy(self.model.trainable_variables)
    assert type(x) is np.ndarray
    return x

  @tf.function
  def set_weights(self, x, model=None):
    """
    Set a `keras` model's `model.trainable_weights` (by default, use this object's `model`).

    Args:
      x: a `numpy.ndarray` (or similar iterable) parameter vector
    """
    assert self._built
    if model is None:
      model = self.model
    params = tf.dynamic_partition(
      tf.convert_to_tensor(x),
      self.part,
      self.n_tensors,
    )
    for i, (shape, param) in enumerate(zip(self.shapes, params)):
      model.trainable_variables[i].assign(
        tf.cast(tf.reshape(param, shape), model.trainable_variables[i].dtype)
      )

  @OptimizationStateCache.cached(key='fg')
  def func_and_grad(self, x):
    """
    Evaluate the loss function and its gradient.
    
    This method will loop over mini-batches of the training data
    and compute the loss and gradient in a `tensorflow.GradientTape`
    for each mini-batch, such that models that require large amounts
    of memory or large training sets (or both) may be used robustly.
    
    The loop is straightforward, and looks like the following, with
    some simplifications made:

    .. code-block:: python

      def func_and_grad(x):

        # Initialize the function & gradient variables
        loss = 0.0
        gradient = np.zeros(x.shape)
        total_weight = 0.0

        # Loop over subsets [= mini-batches] of the dataset
        for k, data in enumerate(self.Xyw):
          data = data_adapter.expand_1d(data)
          x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
          total_weight += sample_weight   # not actually, but just for simple illustration

          # Compute the gradient for the respective training dat asubset
          with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            batch_loss = self.model.compiled_loss(y, y_pred, sample_weight)

          # Accumulate the gradient and loss values for this subset
          batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
          gradient += tf.dynamic_stitch(self.idx, batch_grad).numpy()
          loss += batch_loss.numpy()

        # Compute the regularization terms separately
        return loss, gradient

    Readers familiar with the standard `keras model training procedure <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py>`_
    will note the absence of *regularization losses* provided to
    ``self.model.compiled_loss``. The regularization terms are computed separately
    from the loop snippet above, and then added to the loss & gradient values.

    Args:
      x (`numpy.ndarray`): vector at which to evaluate the loss function

    Returns:
      (float, `numpy.ndarray`): value of the loss function and its gradient
    """
    assert self._built

    self.set_weights(x)
    loss = 0.0
    gradient = np.zeros(x.shape)
    self.fg_evals += 1
    batch_weights = []
    num_batches = None

    for k, data in enumerate(self.Xyw):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      num_samples = len(x)
      num_batches = num_batches or int(np.ceil(self.n / num_samples))
      batch_weights.append(tf.reduce_sum(sample_weight).numpy() if sample_weight is not None else num_samples)

      with tf.GradientTape() as tape:
        y_pred = self.model(x, training=True)
        batch_loss = self.model.compiled_loss(y, y_pred, sample_weight=sample_weight)

      # calculate gradients and convert to 1D tf.Tensor
      batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
      gradient += tf.dynamic_stitch(self.idx, batch_grad).numpy()
      loss += batch_loss.numpy()

    # Add the regularization terms separately to correctly separate this from sample weight normalization
    with tf.GradientTape() as tape:
      reg_loss = self.model.compiled_loss(
        y, y, sample_weight=(0 * y), regularization_losses=[ls or 0.0 for ls in self.model.losses]
      )
    reg_grad = tape.gradient(reg_loss, self.model.trainable_variables)
    reg_grad = tf.dynamic_stitch(self.idx, reg_grad).numpy()
    total_weight = np.sum(batch_weights)
    return (
      loss / total_weight + reg_loss,
      gradient / total_weight + reg_grad,
    )
  
  def func(self, x):
    """
    Evaluate the loss function at `x`.

    Args:
      x (`numpy.ndarray`): vector at which to evaluate the loss function
    Returns:
      float: the value of the loss function at `x`
    """
    return self.func_and_grad(x)[0]

  def grad(self, x):
    """
    Evaluate the gradient of the loss function at `x`.

    Args:
      x (`numpy.ndarray`): vector at which to evaluate the loss function
    Returns:
      numpy.ndarray: the gradient of the loss function at `x`
    """
    return self.func_and_grad(x)[1]

  def hessp(self, x, vector):
    """
    Evaluate the product of the Hessian matrix of the loss function evaluated at `x`
    with a given `vector` for use in second order optimization methods.

    The implementation here has been adapted from the Hessian-vector-product 
    benchmarking suite in `tensorflow`, available
    `here <https://github.com/tensorflow/tensorflow/commit/5b37e7ed14eb7dddae8a0e87435595347a315bb7>`_.
    
    The computation of the Hessian is performed over mini-batches and accumulated,
    as described in the docstring for the method `func_and_grad`.

    Args:
      x (`numpy.ndarray`): point at which to evaluate the loss function
      vector (`numpy.ndarray`): vector to project the Hessian onto
    Returns:
      numpy.ndarray: the Hessian vector product
    """
    self.hessp_evals += 1
    hvp = np.zeros(x.shape)
    self.set_weights(x)
    vector_part = self._numpy_to_tensors(vector)

    batch_weights = []
    num_batches = None

    for k, data in enumerate(self.Xyw):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      num_samples = len(x)
      num_batches = num_batches or int(np.ceil(self.n / num_samples))
      batch_weights.append(tf.reduce_sum(sample_weight).numpy() if sample_weight is not None else num_samples)

      with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
          y_pred = self.model(x, training=True)
          batch_loss = self.model.compiled_loss(y, y_pred, sample_weight=sample_weight)
        grad_slices = inner_tape.gradient(batch_loss, self.model.trainable_variables)
        grads = [tf.convert_to_tensor(g, dtype=self.dtype) for g in grad_slices]

      batch_hvp = outer_tape.gradient(grads, self.model.trainable_variables, output_gradients=vector_part)
      hvp += tf.dynamic_stitch(self.idx, batch_hvp).numpy()

    # Add the regularization terms separately to correctly separate this from sample weight normalization
    with tf.GradientTape() as outer_tape:
      with tf.GradientTape() as inner_tape:
        reg_loss = self.model.compiled_loss(
          y, y, sample_weight=(0 * y), regularization_losses=[ls or 0.0 for ls in self.model.losses]
        )
      reg_grad_slices = inner_tape.gradient(reg_loss, self.model.trainable_variables)
      reg_grads = [tf.convert_to_tensor(g, dtype=self.dtype) for g in reg_grad_slices]

    reg_hvp = outer_tape.gradient(reg_grads, self.model.trainable_variables, output_gradients=vector_part)
    reg_hvp = tf.dynamic_stitch(self.idx, reg_hvp).numpy()

    total_weight = np.sum(batch_weights)
    return hvp / total_weight + reg_hvp

  def minimize(self, epochs=1, callback=None, pretrain_fn=None, **kwargs):
    raise NotImplementedError


class ScipyBatchOptimizer(BatchOptimizer):
  """
  A `BatchOptimizer` that wraps the `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
  library for multivariate gradient-based optimization.
  """
  DEFAULT_METHOD = "L-BFGS-B"

  def __init__(self, name="scipy", method=None, **kwargs):
    if method is not None and method not in OPTIMIZER_IDENTIFIERS:
      raise ValueError(f"Provided method {method} is not in {OPTIMIZER_IDENTIFIERS}")
    self.method = method
    super().__init__(name=name, **kwargs)

  def minimize(self, epochs=1, callback=None, pretrain_fn=None, method=None, options=None):
    """
    Minimize the loss function for a compiled `model` using an algorithm 
    from `scipy.optimize.minimize`.

    The arguments to this method replicate those of the `scipy.optimize` and may be used analogously.
    However it is also possible to pass *list* values for the following arguments:

    - `epochs`
    - `method`
    - `options`

    When lists are provided, this will result in *multiple* successive calls to `scipy.optimize.minize`
    using the `method` and `options` requested, allowing for combinations of algorithms to be used.
    The result of one method will be passed in a daisy chain as the initial value for the next method.

    Args:
      epochs (int or list[int]): number of maximum iterations to perform.
        This will be passed as `options['maxiters']` to `scipy`.
      callback (callable): univariate callable function to be executed at the end of each iteration.
        The argument to this function is the current `np.ndarray` parameter vector. 
      pretrain_fn (callable): univariate callable function to be executed at the end of each iteration.
        The argument to this function is the `model` instance. This may be used in conjuction with
        a list-like `method` to modify a model or set certain model layers as trainable or not trainable
        in an alternating fashion.
      method (str or list[str]): method to use for optimization
      options (dict or list[dict]): dictionary payload of options to configure an optimization algorithm

    Returns:
      `scipy.optimize.OptimizeResult`
    """
    assert self._built

    # Convert scalars to lists to allow for a suite of optimization methods to be applied successively
    epochs = epochs if type(epochs) is list else [epochs]
    options = options if type(options) is list else len(epochs) * [options or {}]
    method = method if type(method) is list else len(epochs) * [method or self.DEFAULT_METHOD]
    pretrain_fn = pretrain_fn if type(pretrain_fn) is list else len(epochs) * [pretrain_fn]

    # Create a list of config dicts to pass to scipy.optimize.minimize successively
    minimize_configs = []
    for k, e in enumerate(epochs):
      config = {
        'options': deepcopy(options[k]),
        'method': method[k],
        'pretrain_fn': pretrain_fn[k],
      }
      config['options']['maxiter'] = e
      minimize_configs.append(config)

    def sigint_handler():
      x = self.get_weights()
      return scipy.optimize.OptimizeResult(
        x=x,
        success=False,
        status=-1,
        message="User aborted optimization",
        fun=self.func(x),
        grad=self.grad(x),
        nit=-1,
        maxcv=None,
      )

    try:
      for config in minimize_configs:
        if callable(config['pretrain_fn']):
          config['pretrain_fn'](self.model)
          # Rebuild the optimize in case the pretrain_fn has modified the model
          self.build(self.model, self.Xyw)

        result = scipy.optimize.minimize(
          fun=self.func,
          x0=self.get_weights(),
          jac=self.grad,
          hessp=self.hessp,
          method=config['method'],
          options=config['options'],
          callback=callback,
        )
        self.set_weights(result.x)
        if result.nit == 0:
          callback(self.get_weights())
    # This is an advanced implementation of early stopping combined
    # with "human in the loop" interative machine learning.
    except KeyboardInterrupt:
      result = sigint_handler()
    return result


OPTIMIZER_IDENTIFIER_MAP = dict(**{
  'scipy': (ScipyBatchOptimizer, (), {}),
}, **{
  method.lower(): (ScipyBatchOptimizer, (), {'name': method.lower(), 'method': method.upper()}) 
  for method in MINIMIZE_METHODS
}, **{
  method.upper(): (ScipyBatchOptimizer, (), {'name': method.lower(), 'method': method.upper()}) 
  for method in MINIMIZE_METHODS
})

def get(identifier):
  """
  Return an instantiated `kormos.optimizers.BatchOptimizer`.

  Args:
    identifier (str or `kormos.optimizers.BatchOptimizer`): A string identifier
      to use to create a `BatchOptimizer`. May be either 'scipy' or the name of a
      specific `method` implemented by ``scipy.optimize.minimize(method=<name>)``
      
  """
  if isinstance(identifier, BatchOptimizer):
    return identifier
  if isinstance(identifier, str):
    if identifier not in OPTIMIZER_IDENTIFIER_MAP:
      raise ValueError(
        f"Provided identifier='{identifier}' is a valid string. Allowed values: {list(OPTIMIZER_IDENTIFIER_MAP.keys())}"
      )
    (cls, args, kwargs) = OPTIMIZER_IDENTIFIER_MAP[identifier]
    return cls(*args, **kwargs)
  raise ValueError(f"Could not interpret optimizer identifier: {identifier}") 

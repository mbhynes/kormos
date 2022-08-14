# MIT License
#
# Copyright (c) 2022 Michael B Hynes
# Copyright (c) 2019 Pi-Yueh Chuang <pychuang@gwu.edu>, where noted.
# Copyright (c) 2015 The TensorFlow Authors, where noted.
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
    self.results = []

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
    # This code has been modified from a program with the original copyright notice:
    #
    # Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
    # Distributed under terms of the MIT license.

    def _build_weight_indices():
      shapes = tf.shape_n(model.trainable_weights)
      n_tensors = len(shapes)

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
        self.set_weights(x)
        n = 0
        loss = 0.0
        gradient = np.zeros(x.shape)

        if self.model.loss.reduction == keras.losses.Reduction.SUM:
          reweight_batches = False
        else:
          reweight_batches = True

        for k, data in enumerate(self.Xyw):
          data = data_adapter.expand_1d(data)
          x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
          num_samples = len(x)
          n += num_samples

          with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            batch_loss = self.model.compiled_loss(y, y_pred, sample_weight=sample_weight)

          # calculate gradients and convert to 1D tf.Tensor
          batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
          batch_grad = tf.cast(tf.dynamic_stitch(self.idx, batch_grad), dtype=self.dtype).numpy()

          coeff = num_samples if reweight_batches else 1.0
          gradient += coeff * batch_grad
          loss += coeff * batch_loss.numpy()
        
        # Add the regularization terms separately to correctly separate this from sample weight normalization
        with tf.GradientTape() as tape:
          reg_loss = self.model.compiled_loss(
            y, y, sample_weight=(0 * y), regularization_losses=self.model.losses,
          )
        reg_grad = [
          g if g is not None else (0 * w)
          for (g, w) in zip(tape.gradient(reg_loss, self.model.trainable_variables), self.model.trainable_variables)
        ]
        reg_grad = tf.cast(tf.dynamic_stitch(self.idx, reg_grad), dtype=self.dtype).numpy()
        coeff = (1.0 / n) if reweight_batches else 1.0
        return (
          (coeff * loss) + reg_loss.numpy(),
          (coeff * gradient) + reg_grad,
        )

    Readers familiar with the standard `keras model training procedure <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py>`_
    will note the absence of *regularization losses* provided to
    ``self.model.compiled_loss``. The regularization terms are computed separately
    from the loop snippet above, and then added to the loss & gradient values.
    
    Please note that providing a `sample_weight` does not necessarily produce a weighted average
    over the dataset. Rather, like `keras`, the `sample_weight` is applied element-wise to each
    data sample, and the final total is divided by the number of training examples if the
    reduction mechanism is `keras.losses.reduction.Reduction.SUM_OVER_BATCH_SIZE`.
    If a weighted average is desired, it is the caller's responsibility to ensure that
    the sum of the values in `sample_weight` is 1.

    Args:
      x (`numpy.ndarray`): vector at which to evaluate the loss function

    Returns:
      (float, `numpy.ndarray`): value of the loss function and its gradient
    """
    assert self._built

    def _fg(x):
      self.set_weights(x)
      n = 0
      loss = 0.0
      gradient = np.zeros(x.shape)
      self.fg_evals += 1

      # The sample_weight
      if self.model.loss.reduction == keras.losses.Reduction.SUM:
        reweight_batches = False
      else:
        reweight_batches = True

      for k, data in enumerate(self.Xyw):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        num_samples = len(x)
        n += num_samples

        with tf.GradientTape() as tape:
          y_pred = self.model(x, training=True)
          batch_loss = self.model.compiled_loss(y, y_pred, sample_weight=sample_weight)

        # calculate gradients and convert to 1D tf.Tensor
        batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
        batch_grad = tf.cast(tf.dynamic_stitch(self.idx, batch_grad), dtype=self.dtype).numpy()

        coeff = num_samples if reweight_batches else 1.0
        gradient += coeff * batch_grad
        loss += coeff * batch_loss.numpy()
      
      # Add the regularization terms separately to correctly separate this from sample weight normalization
      with tf.GradientTape() as tape:
        reg_loss = self.model.compiled_loss(
          y, y, sample_weight=(0 * y), regularization_losses=self.model.losses,
        )
      reg_grad = [
        g if g is not None else (0 * w)
        for (g, w) in zip(tape.gradient(reg_loss, self.model.trainable_variables), self.model.trainable_variables)
      ]
      reg_grad = tf.cast(tf.dynamic_stitch(self.idx, reg_grad), dtype=self.dtype).numpy()
      coeff = (1.0 / n) if reweight_batches else 1.0
      return (
        (coeff * loss) + reg_loss.numpy(),
        (coeff * gradient) + reg_grad,
      )

    # outputs = self.model.distribute_strategy.run(_fg, args=(x,))
    # loss = reduce_per_replica(outputs[0], self.distribute_strategy, reduction='first') 
    # grad = reduce_per_replica(outputs[0], self.distribute_strategy, reduction='first') 
    return _fg(x)
  
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
      x (numpy.ndarray): point at which to evaluate the loss function
      vector (numpy.ndarray): direction vector to project the Hessian onto
    Returns:
      numpy.ndarray: the Hessian vector product
    """
    # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    self.hessp_evals += 1
    n = 0
    hvp = np.zeros(x.shape)
    self.set_weights(x)
    vector_part = self._numpy_to_tensors(vector)

    if self.model.loss.reduction == keras.losses.Reduction.SUM:
      reweight_batches = False
    else:
      reweight_batches = True

    for k, data in enumerate(self.Xyw):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      num_samples = len(x)
      n += num_samples

      with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
          y_pred = self.model(x, training=True)
          batch_loss = self.model.compiled_loss(y, y_pred, sample_weight=sample_weight)
        grad_slices = [
          g if g is not None else (0.0 * w)
          for (g, w) in zip(inner_tape.gradient(batch_loss, self.model.trainable_variables), self.model.trainable_variables)
        ]
        grads = [tf.cast(tf.convert_to_tensor(g), dtype=self.dtype) for g in grad_slices]

      batch_hvp = outer_tape.gradient(grads, self.model.trainable_variables, output_gradients=vector_part)

      coeff = num_samples if reweight_batches else 1.0
      hvp += coeff * tf.dynamic_stitch(self.idx, batch_hvp).numpy()

    # Add the regularization terms separately to correctly separate this from sample weight normalization
    with tf.GradientTape() as outer_tape:
      with tf.GradientTape() as inner_tape:
        reg_loss = self.model.compiled_loss(
          y, y, sample_weight=(0 * y), regularization_losses=self.model.losses,
        )
      reg_grad_slices = [
        g if g is not None else (0.0 * w)
        for (g, w) in zip(inner_tape.gradient(reg_loss, self.model.trainable_variables), self.model.trainable_variables)
      ]
      reg_grads = [tf.cast(tf.convert_to_tensor(g), dtype=self.dtype) for g in reg_grad_slices]

    reg_hvp = outer_tape.gradient(reg_grads, self.model.trainable_variables, output_gradients=vector_part)
    reg_hvp = tf.dynamic_stitch(self.idx, reg_hvp).numpy()

    coeff = (1.0 / n) if reweight_batches else 1.0
    return (coeff * hvp) + reg_hvp

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
    self.method = method or self.DEFAULT_METHOD
    super().__init__(name=name, **kwargs)

  def minimize(
    self,
    epochs=1,
    callback=None,
    pretrain_fn=None,
    method=None,
    options=None,
    bounds=None,
    constraints=None,
    tol=None):
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
      bounds (sequence or `scipy.optimize.Bounds`):
          Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
          trust-constr methods. Please see the `scipy.optimize.minimize` docstring for details.
      constraints: Constraints definition for the COBYLA, SLSQP and trust-constr solvers. 
          Please see the `scipy.optimize.minimize` docstring for details.
    Returns:
      `scipy.optimize.OptimizeResult`
    """
    assert self._built
    if callback is not None:
      assert callable(callback), "Provided callback is not callable"

    # Convert scalars to lists to allow for a suite of optimization methods to be applied successively
    epochs = epochs if type(epochs) is list else [epochs]
    options = options if type(options) is list else len(epochs) * [options or {}]
    method = method if type(method) is list else len(epochs) * [method or self.method]
    pretrain_fn = pretrain_fn if type(pretrain_fn) is list else len(epochs) * [pretrain_fn]

    # Create a list of config dicts to pass to scipy.optimize.minimize successively
    minimize_configs = []
    for k, e in enumerate(epochs):
      config = {
        'options': deepcopy(options[k]),
        'method': method[k],
        'bounds': bounds,
        'constraints': constraints,
        'tol': tol,
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
        _fn = config.pop('pretrain_fn')
        if callable(_fn):
          _fn(self.model)
          # Rebuild the optimize in case the pretrain_fn has modified the model
          self.build(self.model, self.Xyw)

        result = scipy.optimize.minimize(
          fun=self.func,
          x0=self.get_weights(),
          jac=self.grad,
          hessp=self.hessp,
          callback=callback,
          **config,
        )
        self.set_weights(result.x)
        if result.nit == 0 and callback is not None:
          callback(self.get_weights())

        self.results.append(result)
    # This is an advanced implementation of early stopping combined
    # with "human in the loop" interative machine learning.
    except KeyboardInterrupt:
      result = sigint_handler()
      self.results.append(result)
    finally:
      return self.get_weights()


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

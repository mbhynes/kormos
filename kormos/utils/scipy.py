# MIT License
#
# Copyright (c) 2022 Michael B Hynes
# Copyright (c) 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
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


import logging
from copy import deepcopy

import numpy as np
import scipy

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.keras.engine import data_adapter
from tensorflow import keras

from keras.utils import traceback_utils
from keras.engine import base_layer
from keras import callbacks as callbacks_module

from kormos.utils.cache import OptimizationStateCache

logger = logging.getLogger(__name__)


class FunctionFactory:
  """
  A factory to create callables for `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.
  
  The `optimize.minimize` routine has the following abbreviated signature:
  
  .. code-block:: python

    minimize(
      fun,        # callable, the objective function f(x) to minimize 
      ...
      jac=None,   # callable, the jacobian of f(x)
      hessp=None, # callable implementing a hessian/vector product for the hessian of f(x)
      ...
    )

  This factory class provides methods with the appropriate signature for the above callables:
  
  - `fun`, in the method `func(x)` or `func_and_grad(x)`
  - `jac`, in the method `grad(x)` or `func_and_grad(x)`
  - `hessp`, in the method `hessp(x, vector)`
  """

  def __init__(self, model, Xyw, dtype=keras.backend.floatx(), batch_size=2**12, max_cache_entries=5):
    """
    Construct a `FunctionFactory` object, but do not build it.

    Please note that the `dtype` may be explicitly specified, and in general
    `tensorflow.float64` will make the optimization routine more numerically robust
    for loss functions with flat regions.  While single precision
    (`tensorflow.float32`) is often used in deep learning applications, the `scipy`
    numerical minimization routines that use line searches may require double
    precision (`tensorflow.float64`) for the convergence criteria of the line
    search.

    Please note also the choice of `batch_size` here is different than in
    stochastic optimization in which the optimizer typically performs best with
    small "mini-batches" on the order of ``2^4`` or ``2^5``. The `batch_size` here
    should be determined based on the available machine memory, and good values should
    be on the order of ``2^12`` or larger. This `batch_size` is also used during the
    evaluation of the `hessp` method if you are using a second order algorithm
    like Newton-CG the evaluation of `hessp`

    Args:
      model (tesorflow.keras.Model): a model instance for which to optimize parameters
      Xyw (tensorflow.data.Dataset): a training dataset
      dtype: the `tensorflow dtype` to use for weight matrices, defaults to `keras.backend.floatx()`.
      batch_size (int): the number of training examples to process in a single "batch"      max_cache_entries (int): the number of previous `loss/gradient/hessp` values to cache during training
    """
    if not issubclass(type(model.loss), keras.losses.Loss):
      raise ValueError(
        f"Provided model.loss={model.loss} is type: {type(model.loss)}, but a keras.losses.Loss is required"
      )
    if model.loss.reduction != keras.losses.Reduction.SUM:
      raise ValueError(
        f"Provided model.loss.reduction = {model.loss.reduction}, "
        f"but should be: keras.losses.Reduction.SUM (={keras.losses.Reduction.SUM})"
      )

    self.model = model
    if not issubclass(type(Xyw), tf.data.Dataset):
      logger.warning(f"Provided type of Xyw is '{type(Xyw)}; creating a data adapter with batch_size {batch_size}")
      Xyw = data_adapter.get_data_handler(x=Xyw, model=model, batch_size=batch_size)._dataset
      logger.warning(f"Converted input data to {type(Xyw)}")

    self.Xyw = Xyw
    self.dtype = dtype
    self.batch_size = batch_size

    self.n = Xyw.cardinality().numpy()
    self._built = False
    self.cache = OptimizationStateCache(max_entries=max_cache_entries)
    self.k = 0
    self.fg_evals = 0
    self.hessp_evals = 0

  def build(self):
    """
    Dynamically build this `FunctionFactory` object to prepare its use in a
    optimization routine for parameter fitting.

    The build process consists of storing the size metadata about this object's `model`'s
    `trainable_weights` such that we may convert between the stacked `numpy.ndarray` vector
    of parameters and the `model`'s list of multidimensional tensors.

    This method has been adapted from Pi-Yueh Chuang's MIT-licensed
    `code <https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993>`_,
    and this file retains the original 2019 copyright notice above.

    Returns:
      FunctionFactory: this
    """
    # obtain the shapes of all trainable parameters in the model
    model = self.model
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
    return self._tensors_to_numpy(self.model.trainable_variables)

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
      tf.convert_to_tensor(x, dtype=self.dtype),
      self.part,
      self.n_tensors,
    )
    for i, (shape, param) in enumerate(zip(self.shapes, params)):
      model.trainable_variables[i].assign(tf.reshape(param, shape))

  @OptimizationStateCache.cached(key='fg')
  def func_and_grad(self, x):
    """
    Evaluate the loss function and its gradient.

    This function is cached such that successive calls for the same argument
    will not trigger a recomputation.

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

    for k, data in enumerate(self.Xyw.batch(self.batch_size)):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      num_samples = len(x)
      num_batches = num_batches or int(np.ceil(self.n / num_samples))
      batch_weights.append(tf.reduce_sum(sample_weight).numpy() if sample_weight is not None else num_samples)

      with tf.GradientTape() as tape:
        y_pred = self.model(x, training=True)
        batch_loss = self.model.compiled_loss(
          y,
          y_pred,
          sample_weight=sample_weight,
        )

      # calculate gradients and convert to 1D tf.Tensor
      batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
      gradient += tf.dynamic_stitch(self.idx, batch_grad).numpy()
      loss += batch_loss.numpy()

    # Add the regularization terms separately to correctly isolate the normalization
    with tf.GradientTape() as tape:
      reg_loss = self.model.compiled_loss(
        y,
        y,
        sample_weight=(0 * y),
        regularization_losses=self.model.losses,
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

    for k, data in enumerate(self.Xyw.batch(self.batch_size)):
      data = data_adapter.expand_1d(data)
      x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
      num_samples = len(x)
      num_batches = num_batches or int(np.ceil(self.n / num_samples))
      batch_weights.append(tf.reduce_sum(sample_weight).numpy() if sample_weight is not None else num_samples)

      with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
          y_pred = self.model(x, training=True)
          batch_loss = self.model.compiled_loss(
            y,
            y_pred,
            sample_weight=sample_weight,
          )
        grad_slices = inner_tape.gradient(batch_loss, self.model.trainable_variables)
        grads = [tf.convert_to_tensor(g, dtype=self.dtype) for g in grad_slices]

      batch_hvp = outer_tape.gradient(grads, self.model.trainable_variables, output_gradients=vector_part)
      hvp += tf.dynamic_stitch(self.idx, batch_hvp).numpy()

    # Compute the regularization hessian terms
    with tf.GradientTape() as outer_tape:
      with tf.GradientTape() as inner_tape:
        reg_loss = self.model.compiled_loss(
          y,
          y,
          sample_weight=(0 * y),
          regularization_losses=self.model.losses,
        )
      reg_grad_slices = inner_tape.gradient(reg_loss, self.model.trainable_variables)
      reg_grads = [tf.convert_to_tensor(g, dtype=self.dtype) for g in reg_grad_slices]

    reg_hvp = outer_tape.gradient(reg_grads, self.model.trainable_variables, output_gradients=vector_part)
    reg_hvp = tf.dynamic_stitch(self.idx, reg_hvp).numpy()

    total_weight = np.sum(batch_weights)
    return hvp / total_weight + reg_hvp

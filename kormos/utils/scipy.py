# MIT License
#
# Copyright (c) 2022 Michael B Hynes
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
from tensorflow import keras

from keras.utils import traceback_utils
from keras.engine import base_layer
from keras.engine import data_adapter
from keras import callbacks as callbacks_module

from kormos.utils.cache import OptimizationStateCache

logger = logging.getLogger(__name__)


class FunctionFactory:

  def __init__(self, model, loss, Xyw, Xyw_val=None, dtype=keras.backend.floatx(), batch_size=2**12, max_cache_entries=5):
    self.model = model
    self.loss = loss
    self.Xyw = Xyw
    self.Xyw_val = Xyw_val
    self.dtype = dtype
    self.n = len(Xyw)
    self.batch_size = batch_size
    self._built = False
    self.cache = OptimizationStateCache(max_entries=max_cache_entries)
    self.k = 0
    self.fg_evals = 0
    self.hessp_evals = 0
  """
  A factory to create a function required by tfp.optimizer.lbfgs_minimize.

  Args:
    model: an instance of `tf.keras.Model` or its subclasses.
    loss: a tf.keras.losses.Loss function
  """

  def build(self):
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
    assert self._built
    parts = tf.dynamic_partition(x, self.part, self.n_tensors)
    return [tf.reshape(part, self.shapes[k]) for (k, part) in enumerate(parts)]

  def _tensors_to_numpy(self, weights):
    assert self._built
    return tf.dynamic_stitch(self.idx, weights)

  def get_weights(self):
    assert self._built
    return self._tensors_to_numpy(self.model.trainable_variables)

  @tf.function
  def set_weights(self, params_1d, model=None):
    """A function updating the model's parameters with a 1D tf.Tensor.

    Args:
      params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
    """
    assert self._built
    if model is None:
      model = self.model
    params = tf.dynamic_partition(
      tf.convert_to_tensor(params_1d, dtype=self.dtype),
      self.part,
      self.n_tensors,
    )
    for i, (shape, param) in enumerate(zip(self.shapes, params)):
      model.trainable_variables[i].assign(tf.reshape(param, shape))

  @OptimizationStateCache.cached(key='fg')
  def func_and_grad(self, x):
    assert self._built
  
    def eval_model(params_1d):
      loss = 0.0
      gradient = np.zeros(x.shape)
      self.fg_evals += 1
      num_batches = int(np.ceil(len(self.Xyw) / self.batch_size))
      batch_weights = []

      for k, xyw in enumerate(self.Xyw.batch(self.batch_size)):
        _x, _y, _w = data_adapter.unpack_x_y_sample_weight(xyw)
        num_samples = len(_x)
        batch_weights.append(tf.reduce_sum(_w).numpy() if _w is not None else num_samples)

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
          # update the parameters in the model to evaluate the loss function
          y_pred = self.model(_x, training=True)
          y_true = _y
          weight = None if _w is None else _w
          batch_loss = (
            self.loss(y_true, y_pred, sample_weight=weight)
            + (batch_weights[-1] / num_batches) * tf.reduce_sum(tf.convert_to_tensor(self.model.losses, dtype=self.dtype))
          )

        # calculate gradients and convert to 1D tf.Tensor
        batch_grad = tape.gradient(batch_loss, self.model.trainable_variables)
        gradient += tf.dynamic_stitch(self.idx, batch_grad).numpy()
        loss += batch_loss.numpy()

      total_weight = np.sum(batch_weights)
      return loss / total_weight, gradient / total_weight

    params_1d = tf.convert_to_tensor(x, dtype=self.dtype)
    self.set_weights(params_1d)
    func, grad = eval_model(params_1d) 
    return func, grad
  
  def func(self, x):
    return self.func_and_grad(x)[0]

  def grad(self, x):
    return self.func_and_grad(x)[1]

  def hessp(self, x, vector):
    self.hessp_evals += 1
    hvp = np.zeros(x.shape)
    self.set_weights(x)
    vector_part = self._numpy_to_tensors(vector)
    num_batches = int(np.ceil(len(self.Xyw) / self.batch_size))

    batch_weights = []
    for k, xyw in enumerate(self.Xyw.batch(self.batch_size)):
      _x, _y, _w = data_adapter.unpack_x_y_sample_weight(xyw)
      num_samples = len(_y)
      batch_weights.append(tf.reduce_sum(_w).numpy() if _w is not None else num_samples)
      with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
          y_pred = tf.convert_to_tensor(self.model(_x, training=True), dtype=self.dtype)
          loss = (
            self.loss(_y, y_pred, sample_weight=None if _w is None else _w)
            + (batch_weights[-1] / num_batches) * tf.reduce_sum(tf.convert_to_tensor(self.model.losses, dtype=self.dtype))
          )
        # https://biswajitsahoo1111.github.io/post/indexedslices-in-tensorflow/
        grad_slices = inner_tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.convert_to_tensor(g, dtype=self.dtype) for g in grad_slices]

      batch_hvp = outer_tape.gradient(grads, self.model.trainable_variables, output_gradients=vector_part)
      hvp += tf.dynamic_stitch(self.idx, batch_hvp).numpy()

    total_weight = np.sum(batch_weights)
    return hvp / total_weight

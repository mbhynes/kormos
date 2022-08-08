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

from kormos.utils.scipy import FunctionFactory

logger = logging.getLogger(__name__)


class ScipyFittedModel(keras.Model):

  @traceback_utils.filter_traceback
  def fit(
    self,
    x,
    y,
    batch_size=None,
    epochs=20,
    validation_data=None,
    sample_weight=None,
    verbose=0,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    callbacks=None,
    method="L-BFGS-B",
    options=None,
    pretrain_fn=None,
    validation_freq=1,
  ):
    base_layer.keras_api_gauge.get_cell("fit").set(True)
    self._assert_compile_was_called()
    self._check_call_args("fit")

    # Convert scalars to lists to allow for a suite of optimization methods to be applied successively
    epochs = epochs if type(epochs) is list else [epochs]
    options = options if type(options) is list else len(epochs) * [options or {}]
    method = method if type(method) is list else len(epochs) * [method]
    pretrain_fn = pretrain_fn if type(pretrain_fn) is list else len(epochs) * [pretrain_fn]

    minimize_configs = []
    for k, e in enumerate(epochs):
      config = {
        'options': deepcopy(options[k]),
        'method': method[k],
        'pretrain_fn': pretrain_fn[k],
      }
      config['options']['maxiter'] = e
      minimize_configs.append(config)

    # Container that configures and calls `tf.keras.Callback`s.
    if not isinstance(callbacks, callbacks_module.CallbackList):
      callbacks = callbacks_module.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=(verbose != 0),
        model=self,
        verbose=verbose,
        epochs=np.sum(epochs).astype(int),
        steps=1,
      )
    batch_size = batch_size or len(y)
    logs = None
    self._epoch = 0

    factory = FunctionFactory(
      model=self,
      loss=self.loss,
      Xyw=tf.data.Dataset.from_tensor_slices(keras.utils.pack_x_y_sample_weight(x, y, sample_weight)),
      batch_size=batch_size,
      dtype=tf.keras.backend.floatx(),
    )

    callbacks.on_train_begin()

    def sigint_handler():
      return scipy.optimize.OptimizeResult(
        x=factory.get_weights().numpy(),
        success=False,
        status=-1,
        message="User aborted optimization",
        fun=factory.func(factory.get_weights().numpy()),
        grad=factory.grad(factory.get_weights().numpy()),
        nit=self._epoch,
        maxcv=None,
      )

    def _scipy_minimize_callback(x):
      self._epoch += 1
      epoch = self._epoch
      logs = {
        'loss': factory.func(x),
        'grad': np.linalg.norm(factory.grad(x)),
        'fg_evals': factory.fg_evals,
        'hessp_evals': factory.hessp_evals,
      }

      # Compute metrics on the validation set
      if validation_data is not None and self._should_eval(epoch, validation_freq):
        try:
          val_logs = {
            "val_" + name: val for (name, val) in self.test_step(validation_data).items()
          }
          logs.update(val_logs)
        except Exception as e:
          print(e)

      callbacks.on_train_batch_end(epoch, logs)
      callbacks.on_epoch_end(epoch, logs)

      callbacks.on_epoch_begin(epoch + 1)
      callbacks.on_train_batch_begin(epoch + 1)

      if verbose:
        print(f"{self._epoch}: {logs}")

    callbacks.on_epoch_begin(self._epoch)
    callbacks.on_train_batch_begin(self._epoch)

    try:
      for config in minimize_configs:
        if callable(config['pretrain_fn']):
          config['pretrain_fn'](self)

        # Build or rebuild the factory
        factory = factory.build()
        result = scipy.optimize.minimize(
          fun=factory.func,
          x0=factory.get_weights().numpy(),
          jac=factory.grad,
          hessp=factory.hessp,
          method=config['method'],
          options=config['options'],
          callback=_scipy_minimize_callback,
        )
        if result.nit == 0:
          _scipy_minimize_callback(factory.get_weights().numpy())
    except KeyboardInterrupt:
      result = sigint_handler()

    callbacks.on_train_end(logs=logs)
    self._result = result

    return self.history

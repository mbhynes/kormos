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
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from keras import callbacks as callbacks_module

from kormos.utils.scipy import FunctionFactory

logger = logging.getLogger(__name__)


class ScipyFittedModel(keras.Model):

  @traceback_utils.filter_traceback
  def fit(
    self,
    x=None,
    y=None,
    batch_size=None,
    epochs=20,
    verbose='auto',
    callbacks=None,
    validation_split=0.,
    validation_data=None,
    shuffle=False,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    method="L-BFGS-B",
    options=None,
    pretrain_fn=None,
  ):
    self._assert_compile_was_called()
    self._check_call_args("fit")

    if verbose == 'auto':
      if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        verbose = 2  # Default to epoch-level logging for PSStrategy.
      else:
        verbose = 1  # Default to batch-level logging otherwise.

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y, sample_weight), validation_data = (
          data_adapter.train_validation_split(
              (x, y, sample_weight), validation_split=validation_split))

    if validation_data:
      val_x, val_y, val_sample_weight = (
          data_adapter.unpack_x_y_sample_weight(validation_data))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
          self.distribute_strategy)

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

    with self.distribute_strategy.scope(), \
       training_utils.RespectCompiledTrainableState(self):

      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.get_data_handler(
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=1,
        shuffle=shuffle,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        model=self,
        steps_per_execution=self._steps_per_execution
      )

      factory = FunctionFactory(
        model=self,
        Xyw=data_handler._dataset,
        batch_size=batch_size or data_handler._dataset.cardinality().numpy(),
        dtype=tf.keras.backend.floatx(),
      )

      def sigint_handler():
        return scipy.optimize.OptimizeResult(
          x=factory.get_weights(),
          success=False,
          status=-1,
          message="User aborted optimization",
          fun=factory.func(factory.get_weights()),
          grad=factory.grad(factory.get_weights()),
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
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._eval_data_handler = data_adapter.get_data_handler(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size,
              steps_per_epoch=validation_steps,
              initial_epoch=0,
              epochs=1,
              max_queue_size=max_queue_size,
              workers=workers,
              use_multiprocessing=use_multiprocessing,
              model=self,
              steps_per_execution=self._steps_per_execution,
            )
          val_logs = self.evaluate(
            x=val_x,
            y=val_y,
            sample_weight=val_sample_weight,
            batch_size=validation_batch_size or batch_size,
            steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            return_dict=True,
            _use_cached_eval_dataset=True,
          )
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          logs.update(val_logs)
          # try:
          #   val_logs = {
          #     "val_" + name: val for (name, val) in self.test_step(validation_data).items()
          #   }
          # except Exception as e:
          #   print(e)

        callbacks.on_train_batch_end(epoch, logs)
        callbacks.on_epoch_end(epoch, logs)

        callbacks.on_epoch_begin(epoch + 1)
        callbacks.on_train_batch_begin(epoch + 1)

      logs = None
      self._epoch = 0
      callbacks.on_train_begin()
      callbacks.on_epoch_begin(self._epoch)
      callbacks.on_train_batch_begin(self._epoch)

      try:
        for config in minimize_configs:
          if callable(config['pretrain_fn']):
            config['pretrain_fn'](self)

          # Build or rebuild the factory
          factory.build()
          result = scipy.optimize.minimize(
            fun=factory.func,
            x0=factory.get_weights(),
            jac=factory.grad,
            hessp=factory.hessp,
            method=config['method'],
            options=config['options'],
            callback=_scipy_minimize_callback,
          )
          factory.set_weights(result.x)
          if result.nit == 0:
            _scipy_minimize_callback(factory.get_weights())
      # This is an advanced implementeation of early stopping combined
      # with "human in the loop" interative machine learning.
      except KeyboardInterrupt:
        result = sigint_handler()

      callbacks.on_train_end(logs=logs)
      self._result = result

    factory.set_weights(result.x)
    return self.history


class ScipyFittedSequentialModel(keras.Sequential, ScipyFittedModel):

  def fit(self, **kwargs):
    ScipyFittedModel.fit(self, **kwargs)

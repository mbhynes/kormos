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
# ==============================================================================
#
# Modifications to the original code under license above are subject to:
#
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

import kormos
import kormos.optimizers

logger = logging.getLogger(__name__)


class BatchOptimizedModel(keras.Model):

  def compile(self, **kwargs):
    optimizer_orig = kwargs.pop("optimizer", "rmsprop")
    use_batch_fit = False
    try:
      optimizer = keras.optimizers.get(optimizer_orig)
      logger.warning(f"BatchOptimizedModel compiled with optimizer={optimizer}.")
    except ValueError:
      # A BatchOptimizer or its str identifier will raise a ValueError,
      # in which case we know to fit this model using .fit_batch()
      optimizer = "rmsprop"  # Set a valid default to call .compile()
      use_batch_fit = True

    super().compile(optimizer=optimizer, **kwargs)
    # If we are fitting with a deterministic batch algorithm, reset
    # the optimizer to the desired one after compilation
    if use_batch_fit:
      self.optimizer = kormos.optimizers.get(optimizer_orig)
      self.fit = self.fit_batch

  @traceback_utils.filter_traceback
  def fit_batch(
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
      pretrain_fn=None,
      **kwargs
    ):
    self._assert_compile_was_called()
    self._check_call_args("fit")
    if batch_size is not None:
      logger.warning(
        f"batch_size={batch_size} was provided; this will override the setting:"
        f"BatchOptimizedModel.optimizer.batch_size={BatchOptimizedModel.optimizer.batch_size}."
      )

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

    # Container that configures and calls `tf.keras.Callback`s.
    if not isinstance(callbacks, callbacks_module.CallbackList):
      callbacks = callbacks_module.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=(verbose != 0),
        model=self,
        verbose=verbose,
        epochs=epochs,
        steps=1,
      )

    with self.distribute_strategy.scope(), \
       training_utils.RespectCompiledTrainableState(self):

      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.get_data_handler(
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size or self.optimizer.batch_size,
        steps_per_epoch=None,
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

      def _callback(x):
        self._epoch += 1
        epoch = self._epoch
        logs = {
          'loss': self.optimizer.func(x),
          'grad': np.linalg.norm(self.optimizer.grad(x)),
          'fg_evals': self.optimizer.fg_evals,
          'hessp_evals': self.optimizer.hessp_evals,
        }

        # Compute metrics on the validation set
        if validation_data is not None and self._should_eval(epoch, validation_freq):
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._eval_data_handler = data_adapter.get_data_handler(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size or self.optimizer.batch_size,
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
            batch_size=validation_batch_size or batch_size or self.optimizer.batch_size,
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

    self.optimizer.build(model=self, Xyw=data_handler._dataset)
    result = self.optimizer.minimize(
      epochs=epochs,
      callback=_callback,
      pretrain_fn=pretrain_fn,
      **kwargs
    )
    self.optimizer.set_weights(result.x)
    callbacks.on_train_end(logs=logs)
    return self.history


class BatchOptimizedSequentialModel(keras.models.Sequential, BatchOptimizedModel):
  pass

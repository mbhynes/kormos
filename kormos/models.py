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

from keras.utils import traceback_utils
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils

from tensorflow import keras
from keras import callbacks as callbacks_module

import kormos
import kormos.optimizers

logger = logging.getLogger(__name__)


class BatchOptimizedModel(keras.Model):
    """
    A `keras.Model` allowing parameter optimization by *either* stochastic *or*
    deterministic (full batch) algorithms.

    Stochastic optimization may be performed on a `BatchOptimizedModel` using the
    standard optimizers in `keras.optimizers.*` or by using a `kormos.optimizers.BatchOptimizer`.
    The `optimizer` argument to the model's `compile` method will determine which
    training procedure should be used.

    For instance, the same model may be optimized either by SGD or by L-BFGS-B as follows:

    .. code-block:: python

      from tensorflow import keras
      from kormos.models import BatchOptimizedSequentialModel

      # Create an Ordinary Least Squares regressor
      model = BatchOptimizedSequentialModel()
      model.add(keras.layers.Dense(
        units=1,
        input_shape=(5,),
      ))

      # compile the model for stochastic optimization
      model.compile(loss=keras.losses.MeanSquaredError(), optimizer="sgd")
      model.fit(...)

      # compile the model for deterministic optimization using scipy.optimize.minimize
      model.compile(loss=keras.losses.MeanSquaredError(), optimizer="L-BFGS-B")
      model.fit(...)

    """

    def compile(self, **kwargs):
        """
        Configure the model for training.

        If the `optimizer` argument is specified as one of the `keras.optimizers.*`
        (or a string identifier thereby), then this method will simply call the
        parent method `keras.Model.compile` with the arguments provided.
        Subsequent calls to the `model.fit` will perform training using the
        standard `keras.Model.fit` method.

        If the `optimizer` argument is specified as a valid `kormos.optimizers.BatchOptimizer`
        (or a valid string identifier to create one using `kormos.optimizers.get()`), then
        the model will be configured to map calls to `fit` to the method `fit_batch`.

        Keyword Args:
          optimizer: A `keras.optimizer.Optimizer` or string identifier, or a `kormos.optimizer.BatchOptimizer` or string identifier
          **kwargs: all other `kwargs` as passed to `keras.Model.compile <https://keras.io/api/models/model_training_apis/#compile-method>`_
        """
        optimizer_orig = kwargs.pop("optimizer", "rmsprop")
        use_batch_fit = False
        try:
            optimizer = keras.optimizers.get(optimizer_orig)
            logger.warning(f"BatchOptimizedModel compiled with optimizer={optimizer}.")
        except ValueError:
            # A BatchOptimizer or its string identifier will raise a ValueError,
            # in which case we know to fit this model using .fit_batch()
            optimizer = "rmsprop"  # Set a valid default to call .compile() as a dummy
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
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
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
        **kwargs,
    ):
        """
        Train the model using a batch optimization algorithm for up to the maximum number
        of iterations (`epochs`), and return the optimization history object.

        This method will prepare the training dataset and execution context, and then
        call the model's `optimizer`'s `minimize` method (please note that the `optimizer`
        attribute must implement the `kormos.optimizers.BatchOptimizer` interface).
        The `optimizer` should implement a gradient-based optimization algorithm that uses
        the *entire* set of training data to make parameter updates.

        This is different than the standard `keras model training procedure <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py>`_
        in which mini-batches (sample subsets) of the training dataset are created
        and the optimizer makes updates to the model parameters with a stochastic
        algorithm using only that subset.

        The arguments to this method have the same meaning as for the `keras.models.Model.fit <https://keras.io/api/models/model_training_apis/#fit-method>`_ method; please see the documentation there except
        for the arguments noted below.

        Args:
          batch_size (int): the number of training examples to process in a single "mini-batch" if the
            training dataset provided to `compile` must be converted to a `tensorflow.data.Dataset`.
            If unspecified and the training data provided must be converted to a `tensorflow.data.Dataset`,
            the value in this model's `optimizer` (of type `BatchOptimizer`) will be used.
            Please note the choice of `batch_size` here is different than in
            stochastic optimization in which the optimizer typically performs best with
            small mini-batches on the order of ``2^4`` or ``2^5``. The `batch_size` here
            should be determined based on the available machine memory, and good values
            will be on the order of ``2^12`` or larger, depending on the memory requirements
            for the evaluation of the model (memory limitations are more likely to be
            encountered if second order algorithsm like the Newton-CG method are used).
          pretrain_fn (callable): univariate function that accepts the model as its argument
            and performs an arbitrary operation on it. Or, a list of such functions if the
            model's `optimizer` accepts it. The `pretrain_fn` will be applied prior to
            the start of the optimization routine.
        Keyword Args:
          **kwargs: keyword arguments to be passed to this model's `optimizer.minimize(**kwargs)` method
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.
            ValueError: In case of mismatch between the provided input data
                and what the model expects or when the input data is empty.
        """
        self._assert_compile_was_called()
        self._check_call_args("fit")
        if batch_size is not None:
            logger.warning(
                f"batch_size={batch_size} was provided; this will override the setting:"
                f"BatchOptimizedModel.optimizer.batch_size={BatchOptimizedModel.optimizer.batch_size}."
            )

        if verbose == "auto":
            if (
                self.distribute_strategy._should_use_with_coordinator
            ):  # pylint: disable=protected-access
                verbose = 2  # Default to epoch-level logging for PSStrategy.
            else:
                verbose = 1  # Default to batch-level logging otherwise.

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            val_x, val_y, val_sample_weight = data_adapter.unpack_x_y_sample_weight(
                validation_data
            )

        if (
            self.distribute_strategy._should_use_with_coordinator
        ):  # pylint: disable=protected-access
            self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
                self.distribute_strategy
            )

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

        with self.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(
            self
        ):

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
                steps_per_execution=self._steps_per_execution,
            )

            def _callback(x):
                self._epoch += 1
                epoch = self._epoch
                logs = {
                    "loss": self.optimizer.func(x),
                    "grad": np.linalg.norm(self.optimizer.grad(x)),
                    "fg_evals": self.optimizer.fg_evals,
                    "hessp_evals": self.optimizer.hessp_evals,
                }

                # Compute metrics on the validation set
                if validation_data is not None and self._should_eval(
                    epoch, validation_freq
                ):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, "_eval_data_handler", None) is None:
                        self._eval_data_handler = data_adapter.get_data_handler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size
                            or batch_size
                            or self.optimizer.batch_size,
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
                        batch_size=validation_batch_size
                        or batch_size
                        or self.optimizer.batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                        _use_cached_eval_dataset=True,
                    )
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    logs.update(val_logs)

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
            epochs=epochs, callback=_callback, pretrain_fn=pretrain_fn, **kwargs
        )
        self.optimizer.set_weights(result)
        callbacks.on_train_end(logs=logs)
        return self.history


class BatchOptimizedSequentialModel(keras.models.Sequential, BatchOptimizedModel):
    pass

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

import pytest
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2
from tensorflow.python.keras.engine import data_adapter

from kormos.models import BatchOptimizedModel, BatchOptimizedSequentialModel
from kormos.optimizers import ScipyBatchOptimizer


class TestBatchOptimizedModel:

  def test_stochastic_optimizer(self):
    """
    Ensure that the standard keras fit() interface is used if the optimizer
    provided to compile() is a standard (stochastic) optimizer.
    """
    rank = 5
    model_input = keras.Input(shape=(rank,), name="input")
    model_output = keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-3),
      kernel_initializer="ones",
    )(model_input)
    model = BatchOptimizedModel(
      inputs=model_input,
      outputs=model_output,
    )
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    model.compile(loss=loss, optimizer="adam")
    assert model.fit != model.fit_batch

    np.random.seed(1)
    w = np.random.normal(size=rank)
    X = np.random.normal(size=(1000, rank))
    y = X.dot(w)
    model.fit(X, y, epochs=200)
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, atol=1e-2)

  @pytest.mark.parametrize("optimizer", ["scipy", ScipyBatchOptimizer()])
  def test_compile_scipy_optimizer(self, optimizer):
    rank = 5
    model_input = keras.Input(shape=(None, rank,), name="input")
    model_output = keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-3),
      kernel_initializer="normal",
    )(model_input)
    model = BatchOptimizedModel(
      inputs=model_input,
      outputs=model_output,
    )
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    model.compile(loss=loss, optimizer=optimizer)
    assert model.fit == model.fit_batch

    np.random.seed(1)
    w = np.random.normal(size=rank)
    X = np.random.normal(size=(1000, rank))
    y = np.expand_dims(X.dot(w), axis=1)
    dataset = tf.data.Dataset.from_tensors((X, y))

    # Fit using a Dataset
    model.fit(x=dataset, method="L-BFGS-B", epochs=20, validation_data=dataset, options={'ftol': 1e-9})
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, 1e-2)

    #  Fit using raw numpy values
    model.fit(x=X, y=y, method="L-BFGS-B", epochs=20, validation_data=dataset, options={'ftol': 1e-9})
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, 1e-2)


class TestBatchOptimizedSequentialModel:

  def test_stochastic_optimizer(self):
    """
    Ensure that the standard keras fit() interface is used if the optimizer
    provided to compile() is a standard (stochastic) optimizer.
    """
    rank = 5
    model = BatchOptimizedSequentialModel()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-3),
      kernel_initializer="ones",
    ))
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    model.compile(loss=loss, optimizer="adam")
    assert model.fit != model.fit_batch

    np.random.seed(1)
    w = np.random.normal(size=rank)
    X = np.random.normal(size=(1000, rank))
    y = X.dot(w)
    model.fit(X, y, epochs=200)
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, atol=1e-2)

  @pytest.mark.parametrize("optimizer", ["scipy", ScipyBatchOptimizer()])
  def test_compile_scipy_optimizer(self, optimizer):
    rank = 5
    model = BatchOptimizedSequentialModel()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-3),
      kernel_initializer="ones",
    ))
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    model.compile(loss=loss, optimizer=optimizer)
    assert model.fit == model.fit_batch

    np.random.seed(1)
    w = np.random.normal(size=rank)
    X = np.random.normal(size=(1000, rank))
    y = np.expand_dims(X.dot(w), axis=1)
    dataset = tf.data.Dataset.from_tensors((X, y))

    # Fit using a Dataset
    model.fit(x=dataset, method="L-BFGS-B", epochs=20, validation_data=dataset, options={'ftol': 1e-9})
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, 1e-2)

    #  Fit using raw numpy values
    model.fit(x=X, y=y, method="L-BFGS-B", epochs=20, validation_data=dataset, options={'ftol': 1e-9})
    actual = np.reshape(model.trainable_weights[0].numpy(), (1, -1))
    assert np.allclose(actual, w, 1e-2)

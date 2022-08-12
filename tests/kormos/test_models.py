import pytest
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2
from tensorflow.python.keras.engine import data_adapter

from kormos.models import ScipyFittedModel, ScipyFittedSequentialModel
from kormos.models import BatchOptimizedModel, BatchOptimizedSequentialModel
from kormos.optimizers import ScipyBatchOptimizer


class TestScipyFittedModel:

  def _build_ols_model(self, rank=3, num_samples=1000):
    w = 1 + np.arange(rank)
    X = np.random.normal(size=(num_samples, rank))
    y = X.dot(w)
    sample_weight = np.ones(len(X))
    dataset = tf.data.Dataset.from_tensor_slices(
      tensors=(tf.convert_to_tensor(X), tf.convert_to_tensor(y), tf.convert_to_tensor(sample_weight))
    )

    model = ScipyFittedSequentialModel()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-3),
      kernel_initializer="normal",
    ))
    model.compile(loss=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM), optimizer="adam")
    return model, dataset

  def test_fit_ols(self):
    rank = 5
    model, dataset = self._build_ols_model(rank)
    model.fit(x=dataset, method='L-BFGS-B', epochs=10, options={'gtol': 1e-9, 'ftol': 1e-6})
    expected = 1 + np.arange(rank)
    actual = np.reshape(model.trainable_weights[0], (1, -1))
    assert np.allclose(actual, expected, atol=0.01, rtol=0), \
      f"Best fit parameters {model.trainable_weights[0]} did not match expected {expected}"


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

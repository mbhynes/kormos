import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2
from tensorflow.python.keras.engine import data_adapter

from kormos.models import ScipyFittedModel, ScipyFittedSequentialModel


class TestScipyFittedModel:

  keras.backend.set_floatx("float64")

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


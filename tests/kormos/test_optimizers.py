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
import scipy.optimize as opt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import L1, L2
from tensorflow.python.keras.engine import data_adapter

import kormos.optimizers as kopt
from kormos.optimizers import BatchOptimizer, ScipyBatchOptimizer

def test_get():
  with pytest.raises(ValueError):
    kopt.get('invalid')
  assert type(kopt.get('scipy')) is ScipyBatchOptimizer
  assert type(kopt.get('L-BFGS-B')) is ScipyBatchOptimizer
  assert type(kopt.get('l-bfgs-b')) is ScipyBatchOptimizer


class TestBatchOptimizer(object):

  keras.backend.set_floatx("float64")

  def _build_ols_model(self, rank=3):
    w = 1 + np.arange(rank)
    X = np.ones(shape=(2, rank))
    y = X.dot(w)
    sample_weight = np.array([4.0, 3.0]) # different weights, but should be normalized out
    dataset = tf.data.Dataset.from_tensor_slices(
      tensors=(tf.convert_to_tensor(X), tf.convert_to_tensor(y), tf.convert_to_tensor(sample_weight))
    )

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(rank,),
      activation=None,
      use_bias=False,
      kernel_regularizer=L2(1e-1),
      kernel_initializer="ones",
    ))
    model.compile(loss=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM), optimizer="adam")
    return model, dataset

  def _build_mlp_model(self, train_size=(10, 3)):
    w = np.random.normal(size=train_size[-1])
    X = tf.convert_to_tensor(np.random.normal(size=train_size), dtype=tf.dtypes.float64)
    y = tf.convert_to_tensor((X.numpy().dot(w) > 0.25).astype(int), dtype=tf.dtypes.float64)
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(X, y))

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
      units=3,
      input_shape=(train_size[-1],),
      activation='sigmoid',
      use_bias=False,
      kernel_regularizer=L1(1e-1),
      kernel_initializer="ones",
    ))
    model.add(keras.layers.Dense(
      units=1,
      activation='sigmoid',
      use_bias=False,
      kernel_regularizer=L2(1e-1),
      kernel_initializer="ones",
    ))
    model.compile(loss=keras.losses.BinaryCrossentropy(reduction=keras.losses.Reduction.SUM), optimizer="adam")
    return model, dataset

  def test_grad_analytic(self):
    rank = 3
    model, dataset = self._build_ols_model(rank)
    optimizer = BatchOptimizer( 
      dtype=tf.dtypes.float64,
      batch_size=1,
    ).build(model, dataset)

    # Get the first row from the dataset with a non-zero sample weight
    rows = list(dataset.as_numpy_iterator())
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    print(X, y)

    # Retrieve the model weights, which should be the same as the all-ones initial values
    w = optimizer.get_weights()
    assert w.sum() == len(w)
    bias = 0.0
    y_pred = X.dot(w) + bias 
    reg_coeff = 0.1
    n = X.shape[0]
    expected_loss = (1 / n) * (y - y_pred).dot(y - y_pred) + reg_coeff * w.dot(w)
    expected_grad = (2 / n) * X.T.dot(y_pred - y) + 2 * (reg_coeff * w)
    expected_hess = (2 / n) * X.T.dot(X) + 2 * reg_coeff * np.eye(rank)

    assert optimizer.func(w) == expected_loss, f"optimizer.func(w) = {optimizer.func(w)} != expected = {expected_loss}"
    assert np.allclose(optimizer.grad(w), expected_grad), f"optimizer.grad = {optimizer.grad(w)} did not match expected = {expected_grad}"
    for k in range(rank):
      p = np.eye(rank)[k]
      assert np.allclose(
        optimizer.hessp(w, p),
        expected_hess[:, k],
      ), f"optimizer.hessp for {p} = {optimizer.hessp(w, p)} did not match expected = {expected_hess[:, k]}"

  def test_grad_numeric(self):
    tol = 1e-4
    model, dataset = self._build_mlp_model()
    optimizer = BatchOptimizer( 
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    x = np.array(0 * optimizer.get_weights() + 1.0, dtype=np.float64)
    err = opt.check_grad(optimizer.func, optimizer.grad, x)
    if err > tol:
      g_approx = opt.approx_fprime(x, optimizer.func, epsilon=1e-7)
      g = optimizer.grad(x)
      assert err <= tol, f"Analytical gradient and numerical gradient differ by {err} > {tol}:\n grad: {g}\ngrad_approx: {g_approx}"

  def test_hessp_numeric(self):
    tol = 1e-4
    model, dataset = self._build_mlp_model()
    optimizer = BatchOptimizer( 
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    x = np.array(0 * optimizer.get_weights() + 1.0, dtype=np.float64)

    def _hess(x):
      n = len(x)
      H = np.zeros((n, n))
      grad = lambda x: opt.approx_fprime(x, optimizer.func, epsilon=1e-5)
      for k in range(n):
        h_approx = opt.approx_fprime(x, lambda x: grad(x)[k], epsilon=1e-5)
        H[k, :] = h_approx
      return H
      
    H = _hess(x)
    p = np.zeros(len(x))
    for k in range(len(p)):
      p[k] = 1
      Hp = H.dot(p)
      hessp = optimizer.hessp(x, p)
      assert np.allclose(Hp, hessp, atol=1e-2), f"hessp does not match numerical evaluation for k={k}:\nhessp:{hessp}\napprox: {Hp}"
      p[k] = 0


class TestScipyBatchOptimizer(object):

  keras.backend.set_floatx("float64")

  def _build_ols_model(self, rank=3, num_samples=1000):
    w = 1 + np.arange(rank)
    X = np.random.normal(size=(num_samples, rank))
    y = X.dot(w)
    sample_weight = np.ones(len(X))
    dataset = tf.data.Dataset.from_tensor_slices(
      tensors=(tf.convert_to_tensor(X), tf.convert_to_tensor(y), tf.convert_to_tensor(sample_weight))
    )

    model = keras.models.Sequential()
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
    optimizer = ScipyBatchOptimizer( 
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    result = optimizer.minimize(method='L-BFGS-B', epochs=10, options={'gtol': 1e-9, 'ftol': 1e-6})
    expected = 1 + np.arange(rank)
    actual = np.reshape(model.trainable_weights[0], (1, -1))
    assert np.allclose(actual, expected, atol=0.01, rtol=0), \
      f"Best fit parameters {model.trainable_weights[0]} did not match expected {expected}"

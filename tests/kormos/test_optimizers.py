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

def _build_ols_model(
  rank=3,
  X=None,
  sample_weight=None,
  weights=None,
  num_samples=1000,
  reg_coeff=0.1,
  **kwargs
):
  w = weights if weights is not None else (1.0 + np.arange(rank))
  X = X if X is not None else np.random.normal(size=(num_samples, rank))
  y = X.dot(w)
  sample_weight = sample_weight if sample_weight is not None else np.ones(len(X))
  dataset = (tf.convert_to_tensor(X), tf.convert_to_tensor(y), tf.convert_to_tensor(sample_weight))

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
    units=1,
    input_shape=(rank,),
    activation=None,
    use_bias=False,
    kernel_regularizer=L2(reg_coeff),
    kernel_initializer="ones",
  ))
  model.compile(**{
    **{'loss': keras.losses.MeanSquaredError(), 'optimizer': 'adam'},
    **kwargs
  })
  return model, dataset

def _build_mlp_model(train_size=(10, 3), **kwargs):
  w = np.random.normal(size=train_size[-1])
  X = tf.convert_to_tensor(np.random.normal(size=train_size), dtype=tf.dtypes.float64)
  y = tf.convert_to_tensor((X.numpy().dot(w) > 0.25).astype(int), dtype=tf.dtypes.float64)
  dataset = tf.data.Dataset.from_tensors(tensors=(X, y))

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(
    units=2,
    input_shape=(train_size[-1],),
    activation='sigmoid',
    use_bias=False,
    kernel_regularizer=L1(0e-1),
    kernel_initializer="ones",
  ))
  model.add(keras.layers.Dense(
    units=1,
    activation='sigmoid',
    use_bias=False,
    kernel_regularizer=L2(0e-1),
    kernel_initializer="ones",
  ))
  model.compile(**{
    **{'loss': keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM), 'optimizer': 'adam'},
    **kwargs
  })
  return model, dataset


class TestBatchOptimizer(object):

  def test_grad_analytic(self):
    rank = 3
    reg_coeff = 0.1
    w_true = 1.0 + np.arange(rank)
    model, dataset = _build_ols_model(
      rank, weights=w_true, X=np.ones((2, rank)), reg_coeff=reg_coeff
    )
    optimizer = BatchOptimizer(
      dtype=tf.dtypes.float64,
      batch_size=1,
    ).build(model, dataset)

    # Evaluate the expected values and the model at w = [1, 1, 1]
    (X, y) = (dataset[0].numpy(), dataset[1].numpy())
    w = np.ones(rank)
    bias = 0.0
    y_pred = X.dot(w) + bias
    n = X.shape[0]
    expected_loss = (1 / n) * (y - y_pred).dot(y - y_pred) + reg_coeff * w.dot(w)
    expected_grad = (2 / n) * X.T.dot(y_pred - y) + 2 * (reg_coeff * w)
    expected_hess = (2 / n) * X.T.dot(X) + 2 * reg_coeff * np.eye(rank)

    assert np.allclose(optimizer.func(w), expected_loss), f"optimizer.func(w) = {optimizer.func(w)} != expected = {expected_loss}"
    assert np.allclose(optimizer.grad(w), expected_grad), f"optimizer.grad = {optimizer.grad(w)} did not match expected = {expected_grad}"
    for k in range(rank):
      p = np.eye(rank)[k]
      assert np.allclose(
        optimizer.hessp(w, p),
        expected_hess[:, k],
      ), f"optimizer.hessp for {p} = {optimizer.hessp(w, p)} did not match expected = {expected_hess[:, k]}"

  def test_grad_numeric(self):
    tol = 1e-3
    model, dataset = _build_mlp_model()
    optimizer = BatchOptimizer(
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    w0 = optimizer.get_weights()
    np.random.seed(1)
    x = np.random.normal(size=w0.shape)
    epsilon = tol**2
    err = opt.check_grad(optimizer.func, optimizer.grad, x, epsilon=epsilon)
    if err > tol:
      g_approx = opt.approx_fprime(x, optimizer.func, epsilon=epsilon)
      g = optimizer.grad(x)
      assert err <= tol, f"Analytical gradient and numerical gradient differ by {err} > {tol}:\n grad: {g}\ngrad_approx: {g_approx}"

  def test_hessp_numeric(self):
    tol = 1e-4
    model, dataset = _build_mlp_model()
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
      p[k] = 0
      assert np.allclose(Hp, hessp, atol=1e-2), f"hessp does not match numerical evaluation for k={k}:\nhessp:{hessp}\napprox: {Hp}"


class TestScipyBatchOptimizer(object):

  def test_fit_ols(self):
    rank = 5
    weights = 1 + np.arange(rank)
    model, dataset = _build_ols_model(
      rank,
      weights=weights,
      reg_coeff=1e-5,
      loss=keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM), 
    )
    optimizer = ScipyBatchOptimizer(
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    actual = optimizer.minimize(method='L-BFGS-B', epochs=20, options={'gtol': 1e-9, 'ftol': 1e-6})
    assert np.allclose(actual, weights, atol=0.01, rtol=0), \
      f"Best fit parameters {actual} did not match expected {weights}"
    assert len(optimizer.results) == 1
    assert all([r.success for r in optimizer.results])

  def test_fit_ols_multirun(self):
    rank = 5
    weights = 1 + np.arange(rank)
    model, dataset = _build_ols_model(rank, weights=weights, reg_coeff=1e-5)
    optimizer = ScipyBatchOptimizer(
      dtype=tf.dtypes.float64,
      batch_size=2,
    ).build(model, dataset)
    actual = optimizer.minimize(
      method=['L-BFGS-B', 'Newton-CG'],
      epochs=[2, 5],
      options=[{'gtol': 1e-9, 'ftol': 1e-6}, {}],
    )
    assert np.allclose(actual, weights, atol=0.01, rtol=0), \
      f"Best fit parameters {actual} did not match expected {weights}"
    assert len(optimizer.results) == 2
    # We only check for the success flag in the final solver call,
    # since we haven't run the first solver long enough to converge
    assert optimizer.results[-1].success

  def test_fit_ols_distributed(self):
    with tf.distribute.MirroredStrategy().scope():
      self.test_fit_ols()
